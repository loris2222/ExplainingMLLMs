# Create COCO dataloader
from typing_extensions import override
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
import json
from glob import glob
import xml.etree.ElementTree as ET

class CocoDataset(Dataset):
    def __init__(self, root_folder, annotation_file, labelmap_file, processor, imsize=(224,224), out_preds_count=576, keep_iscrowd=False):
        self.root_folder = root_folder
        self.coco = COCO(annotation_file)
        self.label_map = labelmap_file
        self.ids = list(self.coco.imgs.keys())
        self.transform = processor
        self.keep_iscrowd = keep_iscrowd
        self.outsize = imsize
        self.n_preds = out_preds_count

        with open(labelmap_file) as f:
            class_mapping = json.load(f)

        class_mapping = {int(key): value for key, value in class_mapping.items()}
        self.class_mapping = class_mapping

        # COCO classes are not all used. There are 90 class_ids but only 80 are used. This remaps from coco_class_id to an int in [0,79]
        coco_id_to_ordered_mapping = {x: i for (i, x) in zip(range(len(class_mapping)), class_mapping.keys())}

        # And the opposite
        ordered_id_to_coco_mapping = {value: key for key, value in coco_id_to_ordered_mapping.items()}

        self.n_classes = len(class_mapping)

        def coco_id_to_ordered_id(coco_id):
            return coco_id_to_ordered_mapping[coco_id]

        def ordered_id_to_coco_id(ordered_id):
            return ordered_id_to_coco_mapping[ordered_id]
        
        self.coco_id_to_ordered_id = coco_id_to_ordered_id
        self.ordered_id_to_coco_id = ordered_id_to_coco_id

    def __len__(self):
        return len(self.ids)
    
    def get_rescaled_annotations(self, imsize, out_size, original_anns):
        # TODO check that all images are just rescaled and never cropped
        """
        Returns the rescaled annotations as two lists boxes, class_ids in format xyxy between 0 and 1
        """

        scale_factor = np.tile((np.array(out_size)/np.array(imsize)) / np.array(out_size), 2)  # Output is 0-1 scaled and takes into account the image rescaling to IM_SIZE

        out_boxes = torch.zeros([self.n_preds, 4])  # Initialize n_preds boxes to zero (so that they are not ragged and padded to the right n of boxes)
        out_class_ids = torch.full([self.n_preds], self.n_classes)  # Initialize all preds as 'background' class 80 (COCO n_classes)
        out_n_boxes = 0

        for id, ann in enumerate(original_anns):
            # Skip crowd annotations
            if not self.keep_iscrowd and ann["iscrowd"]:
                pass

            # Scale box to new dim
            box = torch.tensor(ann["bbox"] * scale_factor)
            # Get class id
            class_coco_id = ann["category_id"]
            class_ordered_id = torch.tensor(self.coco_id_to_ordered_id(class_coco_id))
            out_boxes[id] = box
            out_class_ids[id] = class_ordered_id
            out_n_boxes += 1
        
        # COCO boxes are in xywh, transform to xyxy
        out_boxes[..., 2:] += out_boxes[..., :2]

        out_n_boxes = torch.tensor(out_n_boxes)
        
        return out_boxes, out_class_ids, out_n_boxes

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root_folder, path)).convert("RGB")

        boxes, class_ids, n_boxes = self.get_rescaled_annotations(image.size, self.outsize, anns)
        img = self.transform(images=image, return_tensors="pt")["pixel_values"][0]

        return img, boxes, class_ids, n_boxes
    
class CocoDatasetOnlyImage(CocoDataset):
    def __init__(self, root_folder, annotation_file, labelmap_file, processor=None, imsize=(224,224), out_preds_count=576, keep_iscrowd=False):
        super().__init__(root_folder, annotation_file, labelmap_file, processor, imsize, out_preds_count, keep_iscrowd)

    @override
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root_folder, path)).convert("RGB")

        return image

class OpenImagesOnlyImage(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.image_paths = glob(self.root_folder + "*.jpg")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        path = self.image_paths[idx]
        image = Image.open(os.path.join(self.root_folder, path)).convert("RGB")

        return image

class LAIONImagesOnlyImage(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.image_paths = glob(self.root_folder + "/**/*.jpg")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        path = self.image_paths[idx]
        image = Image.open(os.path.join(self.root_folder, path)).convert("RGB")

        return image

def load_voc_dataset(data_dir):
    images_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')
    
    image_files = os.listdir(images_dir)
    annotations_files = os.listdir(annotations_dir)
    
    dataset = []
    
    for img_file in image_files:
        img_id = os.path.splitext(img_file)[0]
        annotation_file = img_id + '.xml'
        
        if annotation_file not in annotations_files:
            continue
        
        image_path = os.path.join(images_dir, img_file)
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        # Load image
        image = Image.open(image_path)
        
        # Parse XML annotation
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Extract bounding boxes and labels
        objects = root.findall('object')
        boxes = []
        labels = []
        
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            label = obj.find('name').text
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        dataset.append({
            'image': image,
            'boxes': boxes,
            'labels': labels
        })
    
    return dataset