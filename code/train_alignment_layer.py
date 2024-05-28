import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

owl_vit_checkpoint_name = "google/owlvit-base-patch32"
llava_checkpoint_name = "llava-hf/llava-1.5-7b-hf"
openai_clip_checkpoint_name = "openai/clip-vit-large-patch14-336"
BATCH_SIZE = 16
PROJECTOR_LR = 1*1e-4
ADAMW_WD = 1*1e-2
device_llava = "cuda:1"
device_owl = "cuda:0"
device_projection = "cuda:1"
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

from projection_modules import DeeperMLPProjectionModule

new_token_projection = DeeperMLPProjectionModule()

# Load openimages
from data_utils import OpenImagesOnlyImage
from torch.utils.data import Dataset, DataLoader

oi_root_folder = '../datasets/OpenImages'

def collate_to_list(original_batch):
    return original_batch

coco_dataset = OpenImagesOnlyImage(oi_root_folder)
dataloader = DataLoader(coco_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_to_list)

# Create otimizer
import torch.optim as optim
from torch.optim import AdamW

param_groups = [
            {'params': new_token_projection.parameters(), 'lr': PROJECTOR_LR},
        ]

optimizer = AdamW(param_groups, lr=PROJECTOR_LR, weight_decay=ADAMW_WD)

new_token_projection.train()
new_token_projection.to(device_projection)
llava_model.vision_tower.to(device_llava)
llava_model.multi_modal_projector.to(device_llava)
_ = owl_model.to(device_owl)

# Train projection
for epoch in range(N_EPOCHS):
    step = 0
    cum_loss = 0
    for img in dataloader:
        # Assuming you have input_ids, attention_mask defined
        owl_inputs = owl_processor(images=img, return_tensors="pt")
        owl_inputs = {key: value.to(device_owl) for key, value in owl_inputs.items()}
        out_owl = owl_model.owlvit.vision_model(**owl_inputs, output_hidden_states=True)
        selected_image_feature = out_owl.hidden_states[-2]  # As per LLaVa, take the output to the penultimate transformer layer
        selected_image_feature = selected_image_feature[:, 1:]  # As per LLaVa, remove the CLS token from the features
        image_features_fromowl = new_token_projection(selected_image_feature.to(device_projection))
        
        llava_inputs = llava_processor(text="", images=img, return_tensors="pt")
        llava_inputs = {key: value.to(device_llava) for key, value in llava_inputs.items()}
        image_features_fromllava = llava_model.get_vision_tokens(**llava_inputs)

        loss = torch.nn.functional.mse_loss(image_features_fromowl, image_features_fromllava.to(device_projection), reduction='mean')

        cum_loss += loss.item()
        step += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"\rStep {step}, loss {cum_loss/step}", end="")

    print(f'Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

new_token_projection.eval()
new_token_projection.to("cpu")
llava_model.vision_tower.to("cpu")
llava_model.multi_modal_projector.to("cpu")
_ = owl_model.to("cpu")

torch.save(new_token_projection.state_dict(), "/home/lorisg96/llm_draw/models/llava_projector/projector_deepermlp_oi.pth")


