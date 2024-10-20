from typing import Optional

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

# config as per the huggingface implementation of Paligemma
# https://huggingface.co/google/paligemma-3b-pt-224/tree/main -> `config.json` file

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
                self.text_config = GemmaConfig(**text_config) # sets the values

                self.vocab_size = self.text_config.vocab_size

                self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
                self.vision_config.projection_dim = projection_dim



