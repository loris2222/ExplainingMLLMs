N_IMAGES = 100

# Load COCO 
from pycocotools.coco import COCO
import numpy as np

annotation_file = '../datasets/COCO/annotations/captions_train2017.json'

coco = COCO(annotation_file)
annotations = coco.anns
per_image_annotations = {}

for ann in annotations.values():
    try:
        per_image_annotations[str(ann["image_id"])].append(ann["caption"])
    except KeyError:
        per_image_annotations[str(ann["image_id"])] = [ann["caption"]]

assert N_IMAGES <= len(per_image_annotations)

random_idx = np.random.permutation(len(per_image_annotations))
random_idx = random_idx[:N_IMAGES]
random_coco_ids = [list(per_image_annotations.keys())[x] for x in random_idx]

coco_image_data = [{"id": elem,
                    "url": f"https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/{elem.zfill(12)}.jpg",
                    "caption": per_image_annotations[elem][0]
                    } for elem in random_coco_ids]

# Check that all images have valid URLs
import requests
from tqdm import tqdm

for data in tqdm(coco_image_data):
    requests.get(data["url"], stream=True).raw

# Save data
import pickle

with open("../data/gpt_evaluation/selected_images.pickle", "wb") as f:
    pickle.dump(coco_image_data, f)


