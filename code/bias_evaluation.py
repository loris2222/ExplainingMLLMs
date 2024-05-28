import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda:0"

SPLIT = 0  # 0 female/Caucasian, 1 male/African-American
TYPE = "ethnicity"  # "ethnicity", "gender"

PROPERTY = ""  
original_characteristic = ""  
adversarial_characteristic = ""  


# Load LLaVa
from custom_llava import CustomLlava

custom_llava = CustomLlava()
custom_llava_2 = CustomLlava()

# Load Owl
from custom_owl import CustomOwl

custom_owl = CustomOwl()

# Load projection module
from projection_modules import CopyProjectionModule, TransformerProjectionModule, DeeperMLPProjectionModule

new_token_projection = DeeperMLPProjectionModule()  # CopyProjectionModule()

new_token_projection.to(device)
custom_llava.to(device)
custom_llava.llava_model.multi_modal_projector.to(device)
_ = custom_owl.to(device)

# Run owlvit on an image and save the hidden states, then visualize the result starting from that hidden state
from model_utils import run_owl_model_from_custom_embedding
from PIL import Image
import requests

from adversarial_perturbations import perturb_owl_embedding

from glob import glob
import numpy as np

np.random.seed(0)
portraits_folder = "../data/portraits_sdxl_" + TYPE
portraits_paths = glob(portraits_folder+"/*.jpg")
random_order = np.random.permutation(len(portraits_paths))
portraits_paths = [portraits_folder + f"/portrait_{i}.jpg" for i in range(50*SPLIT,50*(SPLIT+1))]

from pathlib import Path
from adversarial_perturbations import get_delta_characteristic, get_characteristic_score
from tqdm import tqdm
from PIL import Image

max_images = 50
deltas = []
verbose = False

for idx, img_path in tqdm(enumerate(portraits_paths)):
    if idx >= max_images:
        break

    image = Image.open(img_path).convert("RGB")

    try:
        delta, perturbed_embedding = get_delta_characteristic(custom_llava, 
                                                              custom_owl, 
                                                              new_token_projection, 
                                                              image, 
                                                              PROPERTY, 
                                                              original_characteristic, 
                                                              adversarial_characteristic, 
                                                              verbose=verbose, 
                                                              adversarial_lr=1, # 0.005 adam, 1 FSGM
                                                              adversarial_iter=200  # 200 adam, ignored in FSGM
                                                              )
    except ValueError:
        print("Error parsing output, probably not a score. Skipping.")

    deltas.append(delta)

print(f"Average delta: {np.mean(deltas)}")
