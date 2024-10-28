from typing import Tuple, Optional
import torch
import torch.nn as nn

# Architecture of SigLIP
# (VisionEmbeddings, Encoder) -> VisionTransformer -> VisionModel
# (Attention, LayerNorm, MLP) -> EncoderLayer -> Encoder


class SiglipVisionConfig:
    def __init__(self, \
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 hidden_size: int = 768, # embed_dim of image, not text_tokens(we use Linear Projection for this)
                 intermediate_size: int = 3072, # hidden dim of FFN that it projects to.
                 num_hidden_layers: int = 12, # Nx of Encoder Block
                 num_attention_heads: int = 12, # Num of heads of MH-A
                 attention_dropout: float = 0.0,
                 layer_norm_eps: float = 1e-6,
                 num_image_tokens: int = None,
                 **kwargs
                 ):
        '''
            Paligemma comes in different configs
            The above are the default configs of the one we will use
        '''
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        ''' 
            (LayerNorm) -> Attention -> (residual concat) -> (LayerNorm) -> (MLP)
            Idea:
            - Make 3 copies xk, xq, xv
            - Interact with corresponding weight matrices
            - 
        '''
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = 1 / (self.head_dim ** 0.5)

        self.dropout = config.attention_dropout

        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, x):
        batch_size, num_patches, _ = x.shape

        # Contextualize the embddings i.e
        # Each patch needs to interact with key/query/value-learned_matrices 
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        xk = self.key_proj(x)
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        xq = self.query_proj(x)
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        xv = self.value_proj(x)

        # split into heads, so that each head just focusses on a specific part of the embed_dim
        # (B, num_patches, embed_dim) -> (B, num_patches, num_heads, head_dim)
        xk = xk.reshape(batch_size, num_patches, self.num_heads, self.head_dim)
        xq = xq.reshape(batch_size, num_patches, self.num_heads, self.head_dim)
        xv = xv.reshape(batch_size, num_patches, self.num_heads, self.head_dim)

        # Now, since learning of each head can be parallelized, we transpose them
        # For a particular head, all the patches' slice corresponding to that head
        # can be trained indep'tly from other heads
        # (B, num_patches, num_heads, head_dim) -> (B, num_heads, num_patches, head_dim)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        queries = xq.transpose(1, 2)

        # How strongly are ith-query and jth-keys related.
        # Higher the score, they are linked more closely
        # In ow, we are contextualising the embeddings
        # (B, num_heads, num_patches, head_dim) -> (B, num_heads, num_patches, num_patches)
        attn_weights = torch.matmul(queries, keys.transpose(2, 3) ) 

        # scaling down by sqrt(head_dim)
        # (B, num_heads, num_patches, num_patches) -> (B, num_heads, num_patches, num_patches)
        attn_weights = attn_weights * self.scale


        if attn_weights.size() != (batch_size, self.num_heads, num_patches, num_patches):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, num_patches, num_patches)}, but is"
                f" {attn_weights.size()}"
            )    


        # Now, we apply softmax
        # In case of causal mask i.e Language Transf., after calc the attn_weights
        # we mask the future tokens by making the weights as -inf. Why?
        # So, that the softmax,which has e**x will make e**(-inf) = 0, i.e
        # => Thereby making the influence of future tokens 0. (no contribution)

        # Not so, in the case of VisionTransformer, here. Why?
        # An image isn't sequencial, 
        # Argument: "any patch of an image contain info about other patches"
        # weighted-sum argument when interacting attn_weights with values

        # (B, num_heads, num_patches, num_patches)
        attn_outputs = torch.softmax(attn_weights, dim = -1, dtype = torch.float32).to(queries.dtype)

        ##### CAN IGNORE as it is done only during training
        # Apply dropout only during training -> for nference ew are passing dropout = 0.0
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        ################

        # interact with values, so that the weighted_sum of a attn_score can be treated as how much
        # weight/importance each contextualised emb gives to each value inorder to predict
        # the next image_token

        # important funda -> register
        # (B, num_heads, num_patches, num_patches) * (B, num_heads, num_patches, head_dim) 
        #   -> (B, num_heads, num_patches, head_dim)
        attn_outputs = torch.matmul(attn_outputs, values)

        if attn_outputs.size() != (batch_size, self.num_heads, num_patches, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, num_patches, self.head_dim)}, but is"
                f" {attn_outputs.size()}"
            )

        # concat all the heads
        # (B, num_heads, num_patches, head_dim)
        # rearrange info abt heads i.e aggregate all heads to make embed_dim for a given patch
        # (B, num_heads, num_patches, head_dim) -> (B, num_patches, num_heads, head_dim)
        attn_outputs = attn_outputs.transpose(1, 2)

        # concat
        # (B, num_patches, num_heads, head_dim) -> (B, num_patches, num_heads * head_dim)
        # (B, num_patches, num_heads * head_dim) === (B, num_patches, embed_dim)
        attn_outputs = attn_outputs.reshape(batch_size, num_patches, self.num_heads * self.head_dim)

        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        attn_outputs =  self.out_proj(attn_outputs)
        return attn_outputs, attn_weights


class SiglipMLP(nn.Module):
    """
        Idea behind this layer:-
            - Increases degress of freedom of the model
            - i.e more params to learn -> more flexible
            - Has non-linear `gelu` activation function
                - Therefore, model can be more flexible to learn non-linear transformations
                making it more powerful, than to just learn linear-transformations [argument]
            - Also, it generally is used, between 2 main blocks of an architecture
                - To reshape tensors before it enters another block, 
                - The projections in MLP make it compatible for it to be fed 
                to the next block
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.embed_dim = config.hidden_size
        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.embed_dim)


    def forward(self, x):
        # (B, num_patches, embed_dim) 
        x = self.fc1(x)
        x = nn.functional.gelu( x, approximate = "tanh" )
        x = self.fc2(x)
        return x


class SiglipEncoderLayer(nn.Module):
    '''
        Recall:
          (Attention, LayerNorm, MLP) ->  SiglipEncoderLayer
    '''
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layer_norm_eps = config.layer_norm_eps
        self.embed_dim = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(normalized_shape = self.embed_dim, 
                                 eps = self.layer_norm_eps) # normalise across, feature-space to solve covariate-shift problem
        self.self_attn = SiglipAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(normalized_shape = self.embed_dim, 
                                 eps = self.layer_norm_eps)

    def forward(self, x):
        
        residual = x
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        x = self.layer_norm1(x)
        # # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        x, _ = self.self_attn(x)
        # (B, num_patches, embed_dim) -> # (B, num_patches, embed_dim)
        x = residual + x

        residual = x
        # (B, num_patches, embed_dim) -> # (B, num_patches, embed_dim)
        x = self.layer_norm2(x)
        # (B, num_patches, embed_dim) -> # (B, num_patches, embed_dim)
        x = self.mlp(x)
        return residual + x


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers): 
            self.layers.append(SiglipEncoderLayer(config))

    def forward(self, x):
        # "x" comes after final_emb i.e concat( img_embs, pos_embs )
        # (B, num_patches, hidden_size or embed_dim ) -> (B, num_patches, embed_dim) ... N times
        for layer in self.layers:
            x = layer(x)
        return x        

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        '''
            Gets an image as an np-ndarray
            Each Image in a batch gets partitioned into patches
            Each patch is then projected to a `embed_dim` dimesnsion
        '''
        super().__init__()

        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = ( self.image_size // self.patch_size ) ** 2
        self.num_channels = config.num_channels # 3 RGB
        self.embed_dim = config.hidden_size

        # Partition each image into num_patches s.t
        # Each patch has (patch_size, patch_size) shape
        self.patch_embedding = nn.Conv2d( in_channels = self.num_channels, \
                                     out_channels = self.embed_dim, \
                                     stride = self.patch_size, # recall viz in H, W dim. Just this?
                                     kernel_size = self.patch_size, # in conjunction to stride, makes the intended patch
                                     padding = "valid", # no padding applied
                                      )

        # Positional Encodings (to be learned unlike Vanilla Transformer)
        # Recall: In Vanilla Transformer, the positional Encodings were deterministic sinosuids
        self.positional_embeddings = nn.Embedding( self.num_patches, self.embed_dim )

        # For book-keeping we store the postion_ids in buffer for later
        self.position_ids = torch.arange(self.num_patches) # These many postions for each patch
        self.position_ids = self.position_ids.expand((1, -1)) # Adds the batch_dim
        # register in buffer <since we don't learn this.>
        # Also, since it's easy to reconstruct, we don't want to store in state_dict
        # i.e when loading state_dict, we intend not to retain this buffer, <register_buffer is in nn.Module>
        self.register_buffer( name = "postion_ids",
                             tensor = self.position_ids,
                              persistent= False )
        

    def forward(self, x: torch.FloatTensor):
        # x is a numpy tensor/ nd-array
        # (B, C, H, W) -> (B, embed_dim, num_patches_H, num_patches_W)
        # where, num_patches_H = H // patch_size
        # and, num_patches_W = W // patch_size
        patch_embedding = self.patch_embedding(x) 


        # Flatten as in viz ViT before concat, so that same shape as position_enc
        # (B, embed_dim, num_patches_H, num_patches_W) -> (B, embed_dim, num_patches_H * num_patches_W)
        image_embeddings = patch_embedding.flatten( start_dim = 2 ) 

        # (B, num_patches) -> (B, num_patches, embed_dim)
        positional_encodings = self.positional_embeddings( self.position_ids )

        # Align the image_embs to correctly assoc with positional_encodings
        # (B, embed_dim, num_patches_H * num_patches_W) -> (B, num_patches_H * num_patches_W, embed_dim)
        image_embeddings = image_embeddings.transpose(1, 2)
        final_embeddings = image_embeddings + positional_encodings
        return final_embeddings



class SiglipTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        # (VisionEmbeddings, Encoder)
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        # x is a np-ndarray
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.embeddings(x)
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        x = self.encoder(x)
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        x = self.post_layernorm(x)
        return x



class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        # sets the config and model
        self.config = config
        self.model = SiglipTransformer(self.config)

    def forward(self, x):
        # x is of type numpy
        # (Batch, C, H, W)
        return self.model(x)


if __name__ == '__main__':
    batch_size = 2
    num_channels = 3
    height, width = 224, 224

    x = torch.rand(batch_size, num_channels, height, width)
    config = SiglipVisionConfig( num_channels=3, \
                                image_size=224,
                                patch_size=16,
                                hidden_size=768,
                                intermediate_size=3072,
                                num_hidden_layers=12,
                                num_attention_heads=12,
                                attention_dropout=0.0,
                                layer_norm_eps=1e-6,
                                num_image_tokens=None
                                         )

    # print( config.num_channels )

    model = SiglipVisionModel(config)
    out = model.forward(x)
    print( out.shape )  # torch.Size([2, 196, 768])
    # num_patches = (224//16)**2 = 14**2 = 196
    # verified!