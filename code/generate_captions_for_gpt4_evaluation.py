import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

MODE = "owlvit_llava_deepermlp_includecls"  # "original_llava", "owlvit_llava", "owlvit_llava_includecls", "owlvit_llava_deepermlp_includecls"

owl_vit_checkpoint_name = "google/owlvit-base-patch32"
llava_checkpoint_name = "llava-hf/llava-1.5-7b-hf"
openai_clip_checkpoint_name = "openai/clip-vit-large-patch14-336"
BATCH_SIZE = 16
PROJECTOR_LR = 1*1e-4
ADAMW_WD = 1*1e-2
device = "cuda"
N_EPOCHS = 1

# Load LLaVa
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, CLIPImageProcessor

llava_model = LlavaForConditionalGeneration.from_pretrained(llava_checkpoint_name)
llava_pretrained_processor = AutoProcessor.from_pretrained(llava_checkpoint_name)

# Modify processor to not have center crop, so that it's the same as Owl
llava_processor = LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"height": 336, "width": 336}, 
            crop_size={"height": 336, "width": 336}, 
            do_convert_rgb=True, 
            do_center_crop=False, 
            processor_class="LlavaProcessor"
            ),
        tokenizer = llava_pretrained_processor.tokenizer
    )

# Load OwlVit
from transformers import OwlViTProcessor, OwlViTForObjectDetection

owl_model = OwlViTForObjectDetection.from_pretrained(owl_vit_checkpoint_name)
owl_processor = OwlViTProcessor.from_pretrained(owl_vit_checkpoint_name)

# Create new token projection module
from projection_modules import DeeperMLPProjectionModule

new_token_projection = DeeperMLPProjectionModule()

# Move models to GPUs
new_token_projection.to(device)
llava_model.to(device)
llava_model.multi_modal_projector.to(device)
_ = owl_model.to(device)

# Function to generate image caption
def run_llava(image_url, original_vision=True):
    # Get Owl image embedding
    image = Image.open(requests.get(image_url, stream=True).raw)
    text_queries = ["foo"]
    detection_inputs = owl_processor(text=text_queries, images=image, return_tensors="pt")
    detection_inputs = {key: value.to(device) for key, value in detection_inputs.items()}
    out_owl_detection = owl_model(**detection_inputs, output_hidden_states=True)

    # Prepare LLaVa
    prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
    llava_inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    llava_inputs = {key: value.to("cuda") for key, value in llava_inputs.items()}
    selected_image_feature = out_owl_detection.vision_model_output.hidden_states[-2][:, 1:]  # TODO revert to [-2][:, 1:]
    image_features_fromowl = new_token_projection(selected_image_feature)
    # Update vision with Owl embedding if original_vision set to False
    if not original_vision:
        llava_inputs["custom_vision_tokens"] = image_features_fromowl.to("cuda")
    
    # Run LLaVa
    generate_ids = llava_model.generate(**llava_inputs, max_length=300)
    return llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Load data
import pickle

with open("../data/gpt_evaluation/selected_images.pickle", "rb") as f:
    coco_image_data = pickle.load(f)

# Run for all images
from tqdm import tqdm
for data in tqdm(coco_image_data):
    out = run_llava(data["url"], original_vision=(MODE=="original_llava"))
    data["caption_"+MODE] = out.split("ASSISTANT: ")[-1]  # TODO _oi? _coco?

coco_image_data

with open("../data/gpt_evaluation/selected_images.pickle", "wb") as f:
    pickle.dump(coco_image_data, f)


