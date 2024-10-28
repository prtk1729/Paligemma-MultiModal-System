import torch
import torch.nn as nn

from modeling_paligemma import KVCache
from typing import Optional


class GemmaConfig():
    def __init__(
                self, \
                rope_theta: float = 10000.0, # recall pos / 10000.0**( 2i ), i = [0, 1, ..., dim/2], sin or cos
                max_position_encodings: int = 8192, # in position encodings
                rms_norm_eps: float = None, # RMSNorm in Llama / Gemma x / RMS(x), RMS(x).. can u recall?
                hidden_size: int = None,  # embed_dim
                num_hidden_layers: int = None, # Nx of DecoderBlock
                num_attention_heads: int = None, # num of query heads
                num_key_value_heads: int = None, # We will use Multi-Query Attention
                head_dim: int = 256, # for any head, query/key_value
                intermediate_size: int = None,  # in FFN / MLP -> proj -> rev_proj
                attention_bias: bool = False, # all the wk, wv, wq, wo matrices bias
                attention_dropout: float = 0.0, # Inference: no dropout, only needed during training for reg
                pad_token_id: int = None,
                **kwargs
                ):
                super().__init__()
                self.rope_theta = rope_theta
                self.max_position_encodings = max_position_encodings
                self.rms_norm_eps = rms_norm_eps
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers
                self.num_attention_heads = num_attention_heads
                self.num_key_value_heads = num_key_value_heads
                self.head_dim = head_dim
                self.intermediate_size = intermediate_size
                self.attention_bias = attention_bias
                self.attention_dropout = attention_dropout
                self.pad_token_id = pad_token_id

class DecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        # This along with Attention are the main component


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        # self.gamma = nn.Parameter( torch.zeros(self.dim) ) # recall this takes a tensor -> soln: dummy init tensor like zeros
        self.weight = nn.Parameter( torch.zeros(self.dim) ) # recall this takes a tensor -> soln: dummy init tensor like zeros

    def _norm(self, x):
        # Implements the RMSNorm as in the original RMS Norm Paper
        # Recall: center-invariance don't imppact just scale invariants
        # But repacement for std can be this RMS(a) as a proxy for that
        # Similar effect as LayerNorm with less learnables
        # RMS(x) = 1 / sqrt( summation(x_i**2)/dim )
        # rqrt(x) => 1/sqrt(x), since this is deno, for div by 0, we add small eps for smoothing
        smoothed_term = x.pow(2).mean(dim = -1, keepdim = True) + self.eps
        term = torch.rsqrt( smoothed_term )
        rms_term = x * term
        return rms_term

    def forward(self, x):
        x = self._norm(x.float())   
        # In LLaMa, they implement it as x * gamma i.e x * weight
        # Here, x * (1.0 + weight)  
        out = x * (1.0 + self.weight.float())  
        return out

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad_token_id

        self.num_hidden_layers = config.num_hidden_layers

        # padding_idx is used to tell the model, this special idx in the embed-dict
        # say this sepcial id is 0 (say for padding
        # Now, irrespective of whatever this learns, zero this out
        # i.e The embedding_vector correspnding to this idx is 0s
        self.embed_tokens = nn.Embedding( self.vocab_size, \
                                          self.hidden_size,
                                          padding_idx = self.pad_token_id )

        # Decoder Layer
        self.layers = nn.Sequential( \
                                    [ DecoderLayer(config) for layer_idx in range(self.num_hidden_layers) ]
                                    )

        # post decoder norm layer
        self.rms_norm_eps = config.rms_norm_eps
        self.norm = GemmaRMSNorm(dim = self.hidden_size, \
                                 eps = self.rms_norm_eps)

    def tie_weights(self):
        return self.embed_tokens
    
    def forward(self, \
                input_embeds: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None 
                ):
        hidden_states = input_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(input_embeds = hidden_states,
                                          position_ids = position_ids,
                                          attention_mask = attention_mask,
                                          kv_cache = kv_cache
                                          )

        # Finally After all the DecoderLayers there's a post_norm layer in decoder
        # (Batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states) 
        
        return hidden_states
        


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        '''
        Any <>ForCausalLM genrally has all other parts implemented except
        The linear layer that converts into logits for softmax and next token prediction
        '''
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)# language head: converts the embed_dim -> vocab_size

        self.text_config = config
        self.model = GemmaModel(self.text_config) # this is language_model

    def tie_weights(self):
        # share the "weight" params NOT the buffers
        # Why? 
            # The "output embveddings" and "LinearLinear" have opposite funcationalities (invese of each other)
            # We share the same weight matrix to perform the opposite direction functionalities
            # It has been seen, these layers in Decoder-only models constitute 10% of entire weights
        self.lm_head.weight = self.model.embed_tokens.weight
        return
    
    def forward(self, \
                input_embeds: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None
                ):
        
        device, dtype = input_embeds.device, input_embeds.dtype
        # need to scale down the input_embeds
        normalizer = torch.tensor( data = self.hidden_size**0.5, dtype = dtype, device = device )
        input_embeds = input_embeds * normalizer

        # We have the merged input and text embeds as input_embeds
        # inputs_ids after merging reflecting pad indices and actual tokens
        outputs = self.model( 
                            attention_mask = attention_mask,
                            kv_cache = kv_cache,
                            position_ids = position_ids,
                            input_embeds = input_embeds
                            ) 

        # Need to solve the next predicted token
        vocab_logits = self.lm_head(outputs)

        vocab_logits.float()

        # prepare the return data
        return_data = { "logits": vocab_logits }

        # store the updated kv_cache
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache  

        return return_data