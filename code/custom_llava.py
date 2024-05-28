import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, CLIPImageProcessor
import torch.nn as nn

"""
Custom LLaVa implementation that enables to run from custom vision embedding
"""

class CustomLlava(nn.Module):
    def __init__(self, llava_checkpoint_name="llava-hf/llava-1.5-7b-hf"):
        super(CustomLlava, self).__init__()

        # Load base LLaVa
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(llava_checkpoint_name)
        llava_pretrained_processor = AutoProcessor.from_pretrained(llava_checkpoint_name)

        # Modify processor to not have center crop, so that it's the same as Owl
        self.llava_processor = LlavaProcessor(
                image_processor=CLIPImageProcessor(
                    size={"height": 336, "width": 336}, 
                    crop_size={"height": 336, "width": 336}, 
                    do_convert_rgb=True, 
                    do_center_crop=False, 
                    processor_class="LlavaProcessor"
                    ),
                tokenizer = llava_pretrained_processor.tokenizer
            )
        
        self.device = "cpu"
        
    def to(self, device):
        self.device = device
        self.llava_model.to(device)
    
    def cuda(self):
        self.device = "cuda"
        self.llava_model.to("cuda")
    
    def cpu(self):
        self.device = "cpu"
        self.llava_model.to("cpu")

    def run_single(self, prompt, image, custom_vision=None):
        """
        Generates the next token
        If custom_vision is none, returns the original LLaVa output
        If custom_vision is the aligned output of Owl-ViT, 
        """
        llava_inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")
        llava_inputs = {key: value.to(self.device) for key, value in llava_inputs.items()}
        if custom_vision is not None:
            llava_inputs["custom_vision_tokens"] = custom_vision.to(self.device)
        llava_out = self.llava_model(**llava_inputs)
        return llava_out
    
    def run_generate(self, prompt, image, custom_vision=None, max_length=100):
        """
        Generates text using the model
        If custom_vision is none, returns the original LLaVa output
        If custom_vision is the aligned output of Owl-ViT, 
        """
        if image is not None:
            llava_inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")
        else:
            llava_inputs = self.llava_processor(text=prompt, return_tensors="pt")

        llava_inputs = {key: value.to(self.device) for key, value in llava_inputs.items() if value is not None}
        # Add custom vision if present
        if custom_vision is not None:
            llava_inputs["custom_vision_tokens"] = custom_vision.to(self.device)
        # Generate
        generate_ids = self.llava_model.generate(**llava_inputs, max_length=max_length)
        return generate_ids

    
    def generate(self, prompt, image, custom_vision=None, max_length=100):
        """
        Generates using the LLaVa model. 
        """
        generate_ids = self.run_generate(prompt, image, custom_vision=custom_vision, max_length=max_length)
        return self.llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_token_logit(self, llava_outputs, token_id, token_position=-1):
        assert llava_outputs.logits.shape[0] == 1, "Batch must be 1"
        last_logit = llava_outputs.logits[0,token_position,token_id]
        return last_logit

    def get_last_token_logit(self, llava_outputs, token_id):
        return self.get_token_logit(llava_outputs, token_id, token_position=-1)