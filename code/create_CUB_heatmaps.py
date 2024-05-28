import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

owl_vit_checkpoint_name = "google/owlvit-base-patch32"
llava_checkpoint_name = "llava-hf/llava-1.5-7b-hf"
openai_clip_checkpoint_name = "openai/clip-vit-large-patch14-336"
BATCH_SIZE = 16
PROJECTOR_LR = 1*1e-4
ADAMW_WD = 1*1e-2
device_owl = "cuda"
device_llava = "cuda:1"
N_EPOCHS = 1

MAX_IMG = -1

import pickle
cub_val_folder = "../datasets/CUB/CUB_200_2011/images/"
wsolright_cub_test_file = "../datasets/CUB/image_ids.txt"
wsolright_cub_label_file = "../datasets/CUB/class_labels.txt"

# Create list of images to be loaded
with open(wsolright_cub_test_file) as f:
    cub_test_image_ids = f.readlines()

test_image_paths = [cub_val_folder + x.strip() for x in cub_test_image_ids]
test_image_ids = [x.strip() for x in cub_test_image_ids]

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
import copy
new_token_projection = copy.deepcopy(llava_model.multi_modal_projector)
new_token_projection.linear_1 = torch.nn.Linear(in_features=768, out_features=4096, bias=True)

for param in new_token_projection.parameters():
    try:
        torch.nn.init.xavier_uniform_(param)
    except ValueError:
        torch.nn.init.normal_(param)


new_token_projection.load_state_dict(torch.load("../models/llava_projector/projector_deepermlp.pth"))
_ = new_token_projection.eval()

# %%
# Move models to GPUs and copy llava_2 for testing
new_token_projection.to(device_llava)
llava_model.to(device_llava)
llava_model.multi_modal_projector.to(device_llava)
owl_model.to(device_owl)
_ = llava_model.half()  # Otherwise gradients don't fit

# Helper function to draw boxes on image
from PIL import ImageDraw
def draw_image_boxes(image, outputs, text_queries):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for box, score, label in outputs:
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")
    image.show()

from model_utils import run_owl_model_from_custom_embedding
import numpy as np

def last_token_prob(llava_outputs, token_id):
    assert llava_outputs.logits.shape[0] == 1, "Batch must be 1"
    last_logit = llava_outputs.logits[0,-1,token_id]
    return last_logit

def get_heatmap(image, out_size=None, classname="bird"):
    if out_size is None:
        height = image.height
        width = image.width
    else:
        height = out_size
        width = out_size

    out_cam = np.zeros((height, width))

    text_queries = [f"a photo of a {classname}"]
    prompt = "<image>\nUSER: What's in the image?\nASSISTANT: The image shows a"

    # Run Owl to get embeddings to be fed to LLaVa
    detection_inputs = owl_processor(text=text_queries, images=image, return_tensors="pt")
    detection_inputs = {key: value.to(device_owl) for key, value in detection_inputs.items()}
    out_owl_detection = owl_model(**detection_inputs, output_hidden_states=True)

    # Run LLaVa
    token_id = llava_processor.tokenizer.encode(classname)[-1]

    llava_inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    llava_inputs = {key: value.to(device_llava) for key, value in llava_inputs.items()}
    llava_inputs["pixel_values"] = llava_inputs["pixel_values"].half()
    # As per LLaVa, take the output to the penultimate transformer layer, then remove the CLS token from the features
    in_embedding = out_owl_detection.vision_model_output.hidden_states[-2].detach().requires_grad_(True)

    selected_image_feature = in_embedding[:, 1:]
    image_features_fromowl = new_token_projection(selected_image_feature.to(device_llava)) 
    llava_inputs["custom_vision_tokens"] = image_features_fromowl  # Comment this line if you want the original LLaVa output without the Owl-vit projected vision tokens (but careful the projection layer is now changed)
    llava_inputs["custom_vision_tokens"] = llava_inputs["custom_vision_tokens"].half()
    llava_inputs["custom_vision_tokens"].retain_grad()
    llava_model.requires_grad_(True)
    llava_model.zero_grad()

    llava_out = llava_model(**llava_inputs)  # llava_out.logits has long sequence since the first 576 are just the image tokens!
    # last_logit = llava_out.logits[:,-1,:]
    llava_out.logits.retain_grad()

    loss = last_token_prob(llava_out, token_id)

    loss.backward()

    llava_grad = in_embedding.grad.detach().float()  # Gradient of output token prob w.r.t. custom_vision_tokens 
    llava_grad.to(device_owl)

    # Clear memory and gradients or else they don't fit in memory
    import gc

    try:
        del llava_inputs
        del llava_out
        del loss
        llava_model.requires_grad_(False)
        gc.collect()
        # print(torch.cuda.memory_summary())
    except NameError:
        print(torch.cuda.memory_summary())
        pass
    
    # Compute gradients for all features in Owl-ViT (takes a few seconds)
    owl_model.requires_grad_(True)
    owl_model.zero_grad()

    text_queries = ["a photo of a bird"]  # It can be any string list, but keep this short or it takes a long time and OOM since the gradients get large

    in_embedding = out_owl_detection.vision_model_output.hidden_states[-2].detach().requires_grad_(True)
    owlvit_outputs = run_owl_model_from_custom_embedding(owl_model, 
                                                        owl_processor, 
                                                        in_embedding,
                                                        text_queries
                                                        )

    num_classes = owlvit_outputs["logits"].shape[2]
    num_out_tokens = owlvit_outputs["logits"].shape[1]

    all_owl_grads = torch.zeros([num_out_tokens, num_classes + 4] + list(in_embedding.shape))

    for token_id in range(owlvit_outputs["logits"].shape[1]):
        for coord in range(4):
            grad = torch.autograd.grad(owlvit_outputs["pred_boxes"][0,token_id,coord], in_embedding, retain_graph=True)[0]
            all_owl_grads[token_id, num_classes + coord] = grad.detach()

        for class_id in range(owlvit_outputs["logits"].shape[2]):
            grad = torch.autograd.grad(owlvit_outputs["logits"][0,token_id,class_id], in_embedding, retain_graph=True)[0]
            all_owl_grads[token_id, class_id] = grad.detach()

    try:
        del in_embedding, grad
        owl_model.requires_grad_(False)
        gc.collect()
        # print(torch.cuda.memory_summary())
    except NameError:
        print(torch.cuda.memory_summary())
        pass

    # Normalize grads
    llava_grad = llava_grad / torch.norm(llava_grad)
    norms_all_owl_grads = all_owl_grads.pow(2).sum(keepdim=True, dim=[-3, -2, -1]).sqrt()
    all_owl_grads = all_owl_grads / norms_all_owl_grads

    # Compute cosine similarity between all_owl_grads and llava_grad
    grad_similarities = torch.tensordot(all_owl_grads.to(device_owl), llava_grad, dims=[[-3, -2, -1], [-3, -2, -1]])

    grad_similarities[:,:-4] = (grad_similarities[:,:-4] - torch.min(grad_similarities[:,:-4])) / (torch.max(grad_similarities[:,:-4]) - torch.min(grad_similarities[:,:-4]))
    grad_similarities[:,-4:] = (grad_similarities[:,-4:] - torch.min(grad_similarities[:,-4:])) / (torch.max(grad_similarities[:,-4:]) - torch.min(grad_similarities[:,-4:]))

    # Generate all boxes and print them on the saliency
    class_id=[0]
    logits = owlvit_outputs["logits"].detach()
    pred_boxes = owlvit_outputs["pred_boxes"].detach()
    cell_size = 32
    grid_size = [24, 24]
    max_thick = 10
    cmap_scale = 256

    # Scan out featuremap
    for token_id in range(logits.shape[1]):
        # Extract box and grad similarity data for this feature map token
        box_coords = pred_boxes[0,token_id]
        # grad_similarities_coords = grad_similarities[token_id, -4:]
        grad_similarities_class = grad_similarities[token_id, :-4]
        
        # Draw box
        x0 = np.clip(int(width * (box_coords[0]-(box_coords[2]/2))), 0, width)
        x1 = np.clip(int(width * (box_coords[0]+(box_coords[2]/2))), 0, width)
        y0 = np.clip(int(height * (box_coords[1]-(box_coords[3]/2))), 0, height)
        y1 = np.clip(int(height * (box_coords[1]+(box_coords[3]/2))), 0, height)

        # Class weight
        cweight = (torch.max(grad_similarities_class[class_id])).cpu()

        # Set box pixels in saliency map
        overlay_mask = np.zeros_like(out_cam)
        overlay_mask[y0:y1,x0:x1] = cweight

        out_cam = np.max(np.concatenate((overlay_mask[:,:,np.newaxis], out_cam[:,:,np.newaxis]), axis=2), axis=2)
    
    return out_cam

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from os import path

base_save_folder = "../data/wsol_right_heatmaps/CUB/"
heatmap_resolution = 224

for idx, path in tqdm(enumerate(test_image_paths)):
    image = Image.open(path).convert("RGB")
    heatmap = get_heatmap(image, out_size=heatmap_resolution).astype(float)
    image_array = np.array(image.resize((heatmap_resolution,heatmap_resolution)))

    target_folder = "/".join((base_save_folder + test_image_ids[idx]).split("/")[:-1])
    os.makedirs(target_folder, exist_ok=True)

    filename_noext = path.split("/")[-1].split(".")[0]
    filename_ext = path.split("/")[-1]

    np.save(target_folder + "/" + filename_ext + ".npy", heatmap)

# Build wsol_right metadata to enable running evaluation.py on fewer images
out_meta = "\n".join(test_image_ids[:])  # :MAX_IMG

with open(wsolright_cub_test_file, "w") as f:
    f.write(out_meta)
