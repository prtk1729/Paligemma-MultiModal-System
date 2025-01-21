import torch
import torch.nn as nn

from typing import Optional, Tuple
import math


class KVCache():
    def __init__(self):
        '''
            The max size of k_cache can go till "N"
            Where "N", is number of decoder layers i.e "Nx" in diagram
        '''
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []


    def update(self, \
               key_states: torch.Tensor, 
               value_states: torch.Tensor, 
               layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            - We get the computed key_states after interacting with
            - Wk, Wv for each head
            - We store it in respective caches.
            - NOTE: 
                For each layer_idx in DecoderLayer, we store the
                key and value after computation
            - Recall:
                Key and Value Viz
        """
        # Need to know if the layer_idx, is being computed for first time
        if layer_idx >= len(self.k_cache):
            # say, there are 32 DecoderLayers in the Decoder Block
            # But, currently I have processed in the 10 layers
            # i.e key_states and value_states for [1, 2, 3, ..., 10] are in cache
            # 11th layer 1st token comes i.e start_pos = 0 (recall: LLaMa)
            # we need to create a new list_item to populate the 11th layer
            self.k_cache.append(key_states)
            self.v_cache.append(value_states)        

            # Recall viz:
            #   row_view: [ [...], [..new_key_state..] ], new row added
            #   in col view: [ [...], [..new_value_state..] ], new col added           
        else:
            # (Batch, num_kv_heads, seq_len, embed_dim)
            # For each head, the self-attention happens parallely
            # new token's key and value states.
            # Since, new token, get's appended where, i.e at which dim?
            # 2nd last dim => dim = -2
            # say, 11th layer was already present and layer_idx is 11
            # Means this is some token with pos >= 2, for 11th layer
            # Where to place this?
            self.k_cache[layer_idx] = torch.cat([ self.k_cache[layer_idx], key_states ], dim = -2)
            self.v_cache[layer_idx] = torch.cat([ self.v_cache[layer_idx], value_states ], dim = -2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def num_items(self) -> int:
        if len(self.k_cache) == 0:
            return 0
        else:
            # The shape of the k_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.k_cache[0].shape[-2]



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
                vocab_size: int = None,
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
                self.vocab_size = vocab_size



class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed





class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # self.gamma = nn.Parameter( torch.zeros(self.dim) ) # recall this takes a tensor -> soln: dummy init tensor like zeros
        self.weight = nn.Parameter( torch.zeros(dim) ) # recall this takes a tensor -> soln: dummy init tensor like zeros

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


def repeat_kv(x: torch.Tensor, group_size: int):
    # If 1-1 correposndence i.e usual self-attention w/o group/mulitiquery
    if group_size == 1:
        return x

    # Here, some >=2 mapping i.e num_q_heads : num_key_heads = K : 1 , K >= 2
    bsz, num_heads, seq_len, hidden_size = x.shape
    x_expanded = x[:, :, None, :, :].expand( bsz, num_heads, group_size, seq_len, hidden_size )
    # reshape ( bsz, num_heads, group_size, seq_len, hidden_size ) ->
    # bsz, num_heads * group_size, seq_len, hidden_size
    x_expanded = x_expanded.reshape( bsz, num_heads*group_size, seq_len, hidden_size )
    return x_expanded

class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_proj = nn.Linear( self.hidden_size, self.intermediate_size, bias = False )
        self.up_proj = nn.Linear( self.hidden_size, self.intermediate_size, bias = False )
        self.down_proj = nn.Linear( self.intermediate_size, self.hidden_size, bias = False )


    def forward(self, x):
        # (bsz, seq_len, hidden_size)
        y = self.gate_proj(x)
        # apply non-linearity
        y = nn.functional.gelu( y, approximate="tanh" )
        x = self.up_proj(x)
        # Hadamard product
        out = y * x
        return self.down_proj(out)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim # 128 as per HF

        self.attention_dropout = config.attention_dropout
        self.rms_norm_eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_encodings
        self.rope_theta = config.rope_theta
        self.is_causal = True

    
        # RoPE
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


        # For Multi/Grouoped Query Attention
        self.num_heads = config.num_attention_heads # num of heads of Query in total
        self.num_key_value_heads = config.num_key_value_heads # num of heads of Key/Value in total
        # Group size? How many query_heads gets assoc to 1 head of query?
        assert self.num_heads % self.num_key_value_heads == 0, "number of Key/Value heads donot divide Number of Query Heads"
        self.key_value_groups = self.num_heads // self.num_key_value_heads

        # HF uses Multi/Query Attention not evern Grouped-Query as per config. Why?
        # config.num_key_value_heads = 1, as per HF => 1 head in key/value in total
        # Mapping: num_query_heads : 1, Can u viz the assoc?
        self.k_proj = nn.Linear( self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias )
        self.v_proj = nn.Linear( self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias )
        self.q_proj = nn.Linear( self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias )

        self.o_proj = nn.Linear( self.hidden_size, self.hidden_size, bias = config.attention_bias )




    def forward(self,
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs
                ):
        
        # Project them
        # (bsz, seq_len, hidden_szie) -> (bsz, seq_len, num_key_value_heads * head_dim )
        key_states = self.k_proj(hidden_states)
        # (bsz, seq_len, hidden_szie) -> (bsz, seq_len, num_key_value_heads * head_dim )
        value_states = self.v_proj(hidden_states)
        # (bsz, seq_len, hidden_szie) -> (bsz, seq_len, num_heads * head_dim )
        query_states = self.q_proj(hidden_states)

        # info for split to heads
        bsz, seq_len, hidden_size = hidden_states.shape

        # split into heads 
        # 1st reshape and then for parallelising so that each head can work in parallel
        key_states = key_states.view( bsz, seq_len, self.num_key_value_heads, self.head_dim ).transpose(1, 2)
        value_states = value_states.view( bsz, seq_len, self.num_key_value_heads, self.head_dim ).transpose(1, 2)
        query_states = query_states.view( bsz, seq_len, self.num_heads, self.head_dim ).transpose(1, 2)


        #For each head we want to assoc the positional emb info
        # Apply RoPE
        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        # After applying positional info on the key/value states
        # we now have key_states / value_states for all positiona uptil start_pos
        # [  ]
        if kv_cache is not None:
            key_states, value_states = kv_cache.update( key_states, value_states, self.layer_idx )

        # Multi-Query Attention
        # Since, the custom cuda kernel isn't available, we just make `group_size` copies
        # And make 1-1 assoc
        key_states = repeat_kv(key_states, self.key_value_groups)
        value_states = repeat_kv(value_states, self.key_value_groups)


        # Perform the similarity
        # (bsz, num_heads, seq_len, hidden_size ) x (bsz, num_heads, hidden_size, seq_len) 
        # -> (bsz, num_heads, seq_len, seq_len )
        attn_weights = torch.matmul( query_states, key_states.transpose(-2, -1) ) / math.sqrt( self.head_dim )


        # mask stuff
        # In prefilling phase or in generation phase?
        # This answer is stored in the attention_mask
        # - If 0s => don't mask, => prefilling phase => [image][image][bos][text_toks][text_toks][\n_token] => gemma_string
        # Here, the prompt encodes info about the task and it's generally smaller
            # We don't mask the future tokens, they look at each other irrespeive of the positions
        # If generation phase, we mask the future tokens( during training ) => there we add -inf in attention_mask -> e**-inf = 0 after softmax
        # SInce, this is inference, we don't know the gt => usual auto-regressively does it.
        assert attention_mask is not None, "Attention Mask needss to be provided"
        attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.device)
        # dropout, here in inference is 0.
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # print(f"Shape of attn_weights: {attn_weights.shape}")
        # print(f"Shape of value_states: {value_states.shape}")


        # calc. attention_output 
        # (bsz, num_heads, seq_len, seq_len ) x (bsz, num_heads, seq_len, head_dim) ->  (bsz, num_heads, seq_len, head_dim)
        attn_output = torch.matmul( attn_weights, value_states )

        if attn_output.size() !=  (bsz, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Size Mismatch: Expected ( { bsz, self.num_heads, seq_len, self.head_dim } ), instead got: ",
                f"{attn_output.shape} "
            )
        
        # Heads are indepedently calculated based on specific portion on the embed_dim/hidden_size
        # Mix the info, as currently they are independent
        # attn_output = self.o_proj(attn_output) -> Can wer do this? No. Hint: Think of the shape compat

        # Shape of attn_output: (bsz, num_heads, seq_len, head_dim)
        # Shape of o_proj matrix: (hidden_size, hidden_size)
        # So, we need to combine that dim
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape( bsz, seq_len, -1 )
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights



            

class DecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        '''
        For evbery interm layer, we will have kv_cache for k and v
        And DecoderLayer for every layer_idx
        Although, these are interm, we store layer_idx info to persist
        info.
        '''
        super().__init__()
        # This along with Attention are the main components

        # Diagram in mind
        # Only the Decoder Part
        # embs -> [ DecoderLayer ] -> [ DecoderLayer ] -> .... -> contextualised_embs
        self.layer_idx = layer_idx
        self.input_layernorm = GemmaRMSNorm(dim=config.hidden_size)
        self.self_attn = GemmaAttention(config, layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(dim=config.hidden_size)
        self.mlp = GemmaMLP(config)


    def forward(
                self, \
                hidden_states: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None
                ) -> Tuple[ torch.FloatTensor, Optional[Tuple[ torch.FloatTensor, torch.FloatTensor ]] ]: # in come cases attention_scores might be returned
        # (bsz, seq_len, hidden_size)
        residual = hidden_states
        # (bsz, seq_len, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)

        # (bsz, seq_len, hidden_size) -> # (bsz, seq_len, hidden_size)
        hidden_states, _ = self.self_attn( 
                                        hidden_states = hidden_states, \
                                        position_ids = position_ids, \
                                        attention_mask = attention_mask, \
                                        kv_cache = kv_cache
                                        )

        # (bsz, seq_len, hidden_size) -> # (bsz, seq_len, hidden_size)
        hidden_states = residual + hidden_states

        # save for the nxt residual
        residual = hidden_states

        # (bsz, seq_len, hidden_size) -> # (bsz, seq_len, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # (bsz, seq_len, hidden_size) -> # (bsz, seq_len, hidden_size)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states





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
        self.layers = nn.ModuleList( \
                                    [ DecoderLayer(config, layer_idx) for layer_idx in range(self.num_hidden_layers) ]
                                    )

        # post decoder norm layer
        self.rms_norm_eps = config.rms_norm_eps
        self.norm = GemmaRMSNorm(dim = self.hidden_size)

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
            hidden_states = decoder_layer(hidden_states = hidden_states,
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

    def get_input_embeddings(self):
        return self.model.embed_tokens

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

        vocab_logits = vocab_logits.float()

        # prepare the return data
        return_data = { "logits": vocab_logits }

        # store the updated kv_cache
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache  

        return return_data