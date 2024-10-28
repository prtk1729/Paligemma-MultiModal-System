from typing import Optional, List, Union

from PIL import Image
import numpy as np



IMAGENET_STANDARD_MEAN = [ 0.5, 0.5, 0.5 ]
IMAGENET_STANDARD_STD = [ 0.5, 0.5, 0.5 ]


def resize(image: Image, \
           resampling: Image.Resampling,
           image_size: int,
           reducing_gap: Optional[int] = None) -> Image:
    image = image.resize( (image_size, image_size), 
                         resample=resampling, 
                         reducing_gap=reducing_gap )
    return image

def rescale(image: np.ndarray, scale_factor: float, dtype: np.dtype = np.float32):
    image = (image * scale_factor).astype(dtype)
    return image

def normalise(image: np.ndarray, 
              mean: Union[ float, List[float] ],
              std: Union[ float, List[float] ]
                ):
    
    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)

    return (image - mean) / std



def process_images(images: List[Image], \
                   image_size: int,
                   scale_factor: float, 
                   resampling: Image.Resampling = None,
                   reducing_gap: Optional[int] = None
                   ) -> List[np.ndarray]:

    # resize
    images = [ 
                resize(image=image, image_size=image_size, resampling = resampling) 
                for image in images 
            ]
    
    # remaing can be done by np
    images = [ np.array(image) for image in images ]
    
    # scale down
    images = [ 
                rescale(image, scale_factor) 
                for image in images 
            ]   
    # print( np.max(images[0]), np.min(images[0]) ) # (1, 0)
    

    # normalise
    images = [ 
                normalise(image, IMAGENET_STANDARD_MEAN,
                          IMAGENET_STANDARD_STD) 
                for image in images 
            ]      

    return images



def create_gemma_string(prefix_prompt, 
                        image_seq_len, 
                        image_token,
                        bos_token
                        ):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally. 
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
    



class PaligemmaProcessor(): # no inheritance

    def __init__(self, \
                 tokenizer, 
                 num_image_tokens: int, 
                 image_size: int):
        '''
            Idea:
                This block interacts with raw images, raw text
                - With Images
                    - (image) -> resize[Keep PIL. Why?] -> numpy[Why?] -> rescale -> normalize -> torchify[Why?]
                        - [Keep PIL. Why?], Easier to resize images, comes from PIL resize()
                        - numpy[Why?], Easier to (rescale, normalize, operations)
                - With Text
                    - Keep the prompt_string as it is
                    - According to the blog: https://huggingface.co/blog/paligemma#detailed-inference-process 
                        - A partuclr format needs to prepared for Gemma LM
            Returns:
                - dict of {pixel_values, dict() of merged_dummy_imageTokens_tokensizedText }
        '''
        

        self.tokenizer = tokenizer
        self.image_seq_len = num_image_tokens
        self.IMAGE_TOKEN = "<image>"
        self.image_size = image_size

        
        ### PREPARE THE TOKENIZER BY ADDING THINGS
        self._add_new_tokens_to_tokenizer()

        # we will set the eos and bos tokens ourselves and not use the ones in the tokenizer
        self.tokenizer.add_eos_token = False # later tokenizer.eos_token_id = setting
        self.tokenizer.add_bos_token = False

    def _add_new_tokens_to_tokenizer(self):

        # Special tokens (Image tokens are special for Gemma LM, as it is a LM)
        special_tokens = {"additional_special_tokens": [self.IMAGE_TOKEN]}    
        self.tokenizer.add_special_tokens(special_tokens)

        # Gemma can also, carry out seg and OD tasks
        # We add these extra tokens
        # seg task tokens    -> [seg0, seg1, .., seg128] upto 3 decimal places
        EXTRA_TOKENS = [ f"<seg{i:03d}>" for i in range(128) ]
        # OD task   -> [loc0, loc1, .., loc1024] upto 3 decimal places
        EXTRA_TOKENS += [ f"<loc{i:04d}>" for i in range(1024) ]
        self.tokenizer.add_tokens(EXTRA_TOKENS)

        # Also, we need to set image_token_id in tokenizer, to the spceial token's id
        self.tokenizer.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        return


    def __call__(self,\
                 images: List[Image],
                 text: List[str], # List of strings, 1-1 correspondence with the list of images above
                 padding, # For each batch element to be same sz max_seq_len, [prompts can have varied sizes]
                 truncation: bool # If prompt exceeds max_seq_len, truncate it!
                 ): 
        '''
            - Process images
            - Prepare a "gemma_string" that merges [ image_tokens + bos_token + text_prompt ] 
                - This is the synatx as per the paper
            - Then we tokenize this gemma_string
            - Finally, prepare a dict to store the:-
                - pixel_values with actual image pixels
                - [ <dummy_image_token> <dummy_image_token>, ..., <bos_token>, [ prompt_tokens ] ]
                - Later we set the dummy_image_tokens with the output from SigLIP after processing
                    - the image_pixels
        '''

        # Aside[ Recalling my Python concepts ]: Ignore!
        # Need for call, Python101: p = Dog(), then p(...) instance can be used as a function call
        # Overloads the __call__ of object class in python

        assert len(self.images) == 1 and len(self.text) == 1, "Working with only 1 image and prompt, to test, got more than one"

        pixel_values = process_images(images, 
                                        self.image_size, 
                                        scale_factor= 1 / 255.0,
                                        resampling= Image.Resampling.BICUBIC,
                                        ) # list of np.ndarray

        # create a single item for this, to create the batch_dim 
        pixel_values = np.stack(pixel_values, axis = 0) # [ [image1], [image2], [image3], [image4].. ] 
        
        # ("<image>" * seq_len) + "{bos_token}" + prefix_prompts i.e text/prompts
        # Later SigLIP(pixel_values) -> (B, num_patches, embed_dim) -> Projector  
        #   Projector -> aligns it with the gemma_string_tokenized for campatible shape

        # NOTE: num_image_tokens = 256 for gemma
        # NOTE: num_text_tokens = 128 for gemma
        # Hence, there's a need for "Linear Projection Layer"

        gemma_string = create_gemma_string(prefix_prompt = text, 
                                           image_seq_len = self.image_seq_len, 
                                           image_token = self.IMAGE_TOKEN,
                                           bos_token = self.tokenizer.bos_token)

        # str -> tokens i.e encode method
        # Returns, input_ids(i.e tokenised values for seq)
        # and attention_mask
        gemma_string_tokenized = self.tokenizer( gemma_string,
                                                 return_tensors = "pt",
                                                 truncation = truncation, 
                                                 padding = padding
                                               )

        return_data = {"pixel_values": pixel_values, **gemma_string_tokenized}
        return return_data



if __name__ == '__main__':

    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image)
    pixel_values = process_images( [image], 
                                  image_size=224, 
                                  scale_factor=1/255.0,
                                  resampling=Image.Resampling.BICUBIC) # list of np.ndarray
    print(pixel_values[0].shape) # (224, 224, 3)
    print( np.max(pixel_values[0]), np.min(pixel_values[0]) ) # 1.0 -1.0