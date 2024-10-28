import torch
import torch.nn as nn


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



class GemmaForCausalLM(nn.Module):
    pass