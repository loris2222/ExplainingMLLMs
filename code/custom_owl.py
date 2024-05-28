import torch
from PIL import Image, ImageDraw
import requests
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch.nn as nn
from model_utils import run_owl_model_from_custom_embedding

"""
Custom Owl-ViT implementation that enables to run the model from arbitrarily perturbed embeddings
"""

class CustomOwl(nn.Module):
    def __init__(self, owl_vit_checkpoint_name="google/owlvit-base-patch32"):
        super(CustomOwl, self).__init__()
        self.owl_model = OwlViTForObjectDetection.from_pretrained(owl_vit_checkpoint_name)
        self.owl_processor = OwlViTProcessor.from_pretrained(owl_vit_checkpoint_name)
        
    def to(self, device):
        self.device = device
        self.owl_model.to(device)
    
    def cuda(self):
        self.device = "cuda"
        self.owl_model.to("cuda")
    
    def cpu(self):
        self.device = "cpu"
        self.owl_model.to("cpu")
    
    def get_vision_features(self, image):
        """
        Given an image, returns the Owl-ViT vision embedding at the penultimate CLIP layer
        """
        detection_inputs = self.owl_processor(text=["foo"], images=image, return_tensors="pt")
        detection_inputs = {key: value.to(self.device) for key, value in detection_inputs.items()}
        out_owl_detection = self.owl_model(**detection_inputs, output_hidden_states=True)
        return out_owl_detection.vision_model_output.hidden_states[-2]

    def run_from_vision_features(self, text_queries, vision_features):
        return run_owl_model_from_custom_embedding(self.owl_model, 
                                                   self.owl_processor, 
                                                   vision_features,
                                                   text_queries
                                                   )
    
    def post_process_detection(self, out_detection, threshold, target_sizes):
        """
        Given the raw Owl-Vit outputs, returns bounding boxes rescaled to target_sizes image sizes
        """
        return self.owl_processor.post_process_object_detection(out_detection, threshold=threshold, target_sizes=target_sizes)[0]
    
    def draw_image_boxes(self, image, outputs, text_queries, font=None):
        """
        Given outputs = (boxes, scores, labels), returns an image where bboxes are drawn
        """
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for box, score, label in outputs:
            xmin, ymin, xmax, ymax = box
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            xmin_text = max(0, xmin)
            ymin_text = max(0, ymin)
            print(f"drawing {round(score,2)}")
            draw.text((xmin_text, ymin_text), f"{text_queries[label]}: {round(score,2)}", fill="green", font=font)
        return image

    def draw_detections(self, image, text_queries, out_detection, threshold):
        """
        Given raw outputs, returns an image with detections drawn
        """
        target_sizes = torch.tensor([image.size[::-1]])
        processed_out = self.post_process_detection(out_detection, threshold, target_sizes)
        scores = processed_out["scores"].tolist()
        labels = processed_out["labels"].tolist()
        boxes = processed_out["boxes"].tolist()
        return self.draw_image_boxes(image, zip(boxes, scores, labels), text_queries)