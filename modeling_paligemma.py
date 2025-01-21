from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

from modeling_gemma import GemmaForCausalLM, GemmaConfig, KVCache

# config as per the huggingface implementation of Paligemma
# https://huggingface.co/google/paligemma-3b-pt-224/tree/main -> `config.json` file




class PaliGemmaConfig():
    def __init__(
                self, \
                vision_config = None,
                text_config = None,
                projection_dim = 2048, # Linear projection layer
                ignore_index = -100, 
                image_token_index = 256000, # something inside the vocab_size, but outside usual tokens
                pad_token_id = None,
                vocab_size = 257152,
                hidden_size = 2048, # This is text's embed_dim, see projection_dim is what we want the image tokens to project to
                **kwargs
                ):
                super().__init__()
                self.vision_config = vision_config
                self.text_config = text_config
                self.projection_dim = projection_dim
                self.ignore_index = ignore_index
                self.image_token_index = image_token_index
                self.pad_token_id = pad_token_id
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size

                # set other things
                self.vision_config = SiglipVisionConfig(**vision_config) # sets the values
                self.text_config = GemmaConfig(**text_config, pad_token_id = self.pad_token_id) # sets the values

                self.vocab_size = self.text_config.vocab_size


                self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
                self.vision_config.projection_dim = projection_dim






class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.projection_dim = config.projection_dim
        self.image_emb = config.vision_config.hidden_size
        self.linear = nn.Linear(self.image_emb, self.projection_dim, bias = False)


    def forward(self, \
                pixel_values: torch.FloatTensor\
                ):
        # (bsz, num_patches, image_token_embed_dim)  -> (bsz, num_patches, text_emb i.e proj_dim)
        pixel_values = self.linear(pixel_values)
        return pixel_values
        
         

class PaliGemmaForConditionalGeneration(nn.Module):
    
    def __init__(self, 
                config: PaliGemmaConfig
                ):
        super().__init__()
        self.config = config
        self.vision_config = self.config.vision_config # Recall we upated this in PGConfig and ret the updated config obj
        self.vision_tower = SiglipVisionModel(self.vision_config)

        self.text_config = self.config.text_config
        self.language_model = GemmaForCausalLM(self.text_config)
        #set the vocab_size

        # set the pad_token_id 
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        # get the dummy image token id
        self.dummy_image_token_id = self.config.image_token_index

        # Linear-Projector
        self.multi_modal_projector = PaliGemmaMultiModalProjector(self.config)


    def _get_masks(self, input_ids):
        pad_token_mask = (input_ids == self.pad_token_id)
        image_token_mask = (input_ids == self.dummy_image_token_id)
        text_token_mask = (input_ids != self.dummy_image_token_id) & (input_ids != self.pad_token_id)
        return pad_token_mask, text_token_mask, image_token_mask

    def _create_final_embedding(
                                self, \
                                text_token_mask, 
                                pad_token_mask,
                                image_token_mask,
                                final_embedding,
                                projected_image_features,
                                input_embeds
                                ):
                # replace at the necessary places
        # can do now, as text_token_mask has same shape as input_embeds and final_embedding
        # src: input_embeds, tgt: final_embeds have same shape -> can copy this way
        final_embedding = torch.where( text_token_mask, input_embeds, final_embedding )

        # place image_tokens -> after Linear Projection
        # 1st we scale to make the values out of Linear_Projection, numerically stable
        # for stable training
        scale = self.config.projection_dim**-0.5
        scaled_projected_image_embeds = projected_image_features * scale
        # scenario: Shape of tgt and src don't match as seq_len of image_emb are diff
        #   [T, F, T, F], tgt = [1, 2, 3, 4], src: [10, 12, 13, 14]
        #                 tgt = [10, 2, 13, 4]      
        final_embedding = final_embedding.masked_scatter( mask = image_token_mask, 
                                                         source = scaled_projected_image_embeds )

        # populate 0s at padding token positions
        final_embedding = torch.where( pad_token_mask, 
                                      torch.zeros_like(final_embedding),
                                      final_embedding )
        return final_embedding

    def _get_causal_mask_and_position_ids(self, 
                                          kv_cache: Optional[KVCache] = None, 
                                          attention_mask: torch.Tensor = None, 
                                          input_embeds: torch.Tensor = None):
        '''
            Create Causal Mask and Position-Ids of the incoming query-token for:-
                - Prefilling stage
                    - Only the <image_token_embs><bos><text_token_embs><"\n"> i.e gemma_string_tokenised
                - Generation stage
                    - [original_query][genarated_token] -> predict new token i.e Next ttoken prediction task
        '''

        # unpack necessary things when required
        q_len = input_embeds.shape[1] # (B, seq_len, embed_dim)
        batch_size = input_embeds.shape[0] # (B, seq_len, embed_dim)
        device, dtype = input_embeds.device, input_embeds.dtype

        causal_mask, position_ids = None, None

        if (kv_cache is None) or (kv_cache.num_items() == 0):
            print(f"Prefilling Phase")
            # prefilling stage
            # imagine a mask matrix just for one sequence i.e batch_idx = 0
            # square -> [ gemma_string_tokens_len, gemma_string_tokens_len ]
            causal_mask = torch.full( size = (batch_size, q_len, q_len), 
                                     dtype = dtype, device = device,
                                    fill_value=0  )

        else:
            # generation phase
            print(f"Generation Phase")
            assert q_len == 1, "Generation Phase more than one token CAN'T be input"
            # Single token interacts with [all previous + current token's key and values]
            # Recall important viz
            kv_len = kv_cache.num_items() + q_len # [prev + current] key_states and value_states
            causal_mask = torch.full( (batch_size, q_len, kv_len), \
                                      device = device,
                                      dtype = dtype,
                                      fill_value=0,
                                    ) # recall attention_mask for future pad is 0.
            position_pre_sum = attention_mask.cumsum(dim=-1)


        # shape expected for causal mask has to contain kv_heads
        # NEED TO REGISTER SCENARIOS WHERE .EXPAND TRICK AND WHERE UNSQUEEZE
        # (bsz, kv_heads, seq_len, embed_dim)
        # causal_mask is used at the Self-Attention after splitting into heads
        causal_mask = causal_mask.unsqueeze( 1 ) 

                    
            # # What's the idx/ start_pos(recall in LLaMa), for this incoming query_token
            # # This info can be derived from attention_mask
            # # attention_mask: (Batch_size, seq_len) -> Need to know num 1s in seq_len dim in each batch_idx
            # positional_ids = attention_mask.cumsum(dim=-1).masked_fill_( mask = (attention_mask == 0),
                                       # last item of the prefix sum tells position of incoming q_token
            # position_ids = position_pre_sum[:, -1]                                              value = 1 ).to(device)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device) 

        print( position_ids, type(position_ids) )
        return causal_mask, position_ids

 
    def _merge_input_ids_with_image_features(self, \
                                             input_ids, \
                                             input_embeds, \
                                             attention_mask, 
                                             kv_cache,
                                             projected_image_features
                                            ):
        '''
            - Here, we have the input_embeds containing
                - <dummy_image_tokens><dummy_image_tokens>....<bos_token><text_token1><text_token2>...<"\n">
                - This string was tokenized and then an embedding was created
            - input_embed shape: (bsz, seq_len, embed_dim)
        '''

        # We need to know where the text_tokens, image_tokens, pad_tokens are
        # (Batch_sz, seq_len)
        pad_token_mask, text_token_mask, image_token_mask = self._get_masks(input_ids)
        # shape of these masks?
        # (Batch_size, seq_len)


        # based on the above we can store the input_embeds out of Linear Projector
        # instead of modifying input_embeds we create a final_embedding
        dtype, device = input_embeds.dtype, input_embeds.device
        batch_size, seq_len = input_ids.shape
        _, _, embed_dim = input_embeds.shape # (Batch_size, seq_len, embed_dim)
        final_embedding = torch.zeros( size = (batch_size, seq_len, embed_dim),
                                      device = device, 
                                      dtype = dtype )

        # populate embeddings -> shape mismatch
        text_token_mask = text_token_mask[:, :, None].expand(-1, -1, embed_dim)
        image_token_mask = image_token_mask[:, :, None].expand(-1, -1, embed_dim)
        pad_token_mask = pad_token_mask[:, :, None].expand(-1, -1, embed_dim)

        # populate at respective places
        final_embedding = self._create_final_embedding(
                                                       text_token_mask, 
                                                       pad_token_mask,
                                                       image_token_mask,
                                                       final_embedding,
                                                       projected_image_features,
                                                       input_embeds
                                                       )

        # attention_mask, causal_mask
        # decision to send input_embeds: Has batch_size, Has q_len, Has dtype and Has device
        causal_mask, position_ids = self._get_causal_mask_and_position_ids(kv_cache, 
                                                                          attention_mask,
                                                                          input_embeds)
        return final_embedding, causal_mask, position_ids


    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(self, \
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None
                ):
        """
            - Here, the processing_paligemma is already done
            - We now, have the actual pixel_values for a given image
                - It can be for a batch of image, as well, (for our case, just 1 image and prompt)
            - So, the tokenizer was called for "gemma_string"
            - Recall: gemma_string -> {dummy_image_tokens}{bos_token}{prefix_prompt}{"\n"} 
                - input_ids, attention_mask = tokenizer(gemma_string)
            - The args "input_ids" and "attention_mask" are as the ones
            o/p from the tokenizer
            
            - This method, just projects the pixel_values after going through siglip
            - to the intended_dim as that of text_tokens
            - Then it places these actual values at the "dummy_image_tokens"

            - After merging, this concatenated <prompt_token_emb, image_token_emb> are sent to the language model
        """

        # We have the raw pixel values from the processor for a given image, transformed as torch
        pixel_values = self.vision_tower(pixel_values)
        projected_image_features = self.multi_modal_projector(pixel_values)

        # we need the image_embeds for gemma_string
        # get_input_embeddings()_in_GemmaModel = self.language_model.get_input_embeddings()
        # recall: GemmaModel -> GemmaForCausalLM (Both of these will have this "get_input_embeddings()" )
        # We need to fetch, the placeholder_embeds_after_tokenisation_of_gemma_string
        input_embeds = self.language_model.get_input_embeddings()(input_ids) # nn.Embedding(input_ids)

        # Now we can combine/merge by placing them at the placeholder in the 
        # tokenised gemma_string
        gemma_string_embed, attention_mask, position_ids = self._merge_input_ids_with_image_features( \
                                                                input_ids,
                                                                input_embeds,
                                                                attention_mask,
                                                                kv_cache,
                                                                projected_image_features
                                                                )

        # outputs for multiple [ <image, prefix_prompt>, <>, <>, ] for a batch
        outputs = self.language_model( 
                                        attention_mask = attention_mask,
                                        kv_cache = kv_cache,
                                        position_ids = position_ids,
                                        input_embeds = gemma_string_embed
                                     ) 

        return outputs