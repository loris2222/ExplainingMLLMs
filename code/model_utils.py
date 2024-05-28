from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput
import numpy as np
import torch
from PIL import Image
import torch.nn as nn

def run_owl_model_from_custom_embedding(owl_model, owl_processor, penultimate_vision_hidden_state, text_queries):
    """
    Run Owl-ViT from the feature used by LLaVa (this is useful after we have perturbed it)
    """
    # Copy owl output at the embedding used by LLaVa, which is at the penultimate transformer layer
    device = penultimate_vision_hidden_state.device

    detection_inputs = owl_processor(text=text_queries, images=[Image.new("RGB", size=(768,768))], return_tensors="pt")
    detection_inputs = {key: value.to(device) for key, value in detection_inputs.items()}   
    
    original_text_embedding = owl_model(**detection_inputs).text_embeds

    # Run the remainder of the vision tower in Owl-ViT
    last_transformer_layer_input = {"hidden_states": penultimate_vision_hidden_state, "attention_mask": None, "causal_attention_mask": None, "output_attentions": False}
    last_transformer_layer_output = owl_model.owlvit.vision_model.encoder.layers[-1](**last_transformer_layer_input)[0]
    image_embeds = owl_model.owlvit.vision_model.post_layernorm(last_transformer_layer_output)

    # NOTE this is inspired from modeling_owlvit.OwlVitForObjectDetection.image_embedder
    # Resize class token
    new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
    class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

    # Merge image embedding with class tokens
    image_embeds = image_embeds[:, 1:, :] * class_token_out
    image_embeds = owl_model.layer_norm(image_embeds)

    # Resize to [batch_size, num_patches, num_patches, hidden_size]
    new_size = (
        image_embeds.shape[0],
        int(np.sqrt(image_embeds.shape[1])),
        int(np.sqrt(image_embeds.shape[1])),
        image_embeds.shape[-1],
    )
    image_embeds = image_embeds.reshape(new_size)

    # NOTE this is inspired from modeling_owlvit.OwlVitForObjectDetection.forward
    batch_size, num_patches, num_patches, hidden_dim = image_embeds.shape  # feature_map is image_embeds returned by image_text_embedder
    image_feats = torch.reshape(image_embeds, (batch_size, num_patches * num_patches, hidden_dim))
    input_ids = detection_inputs["input_ids"]

    # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
    max_text_queries = input_ids.shape[0] // batch_size
    original_text_embedding = original_text_embedding.reshape(batch_size, max_text_queries, original_text_embedding.shape[-1])

    # If first token is 0, then this is a padded query [batch_size, num_queries].
    input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
    query_mask = input_ids[..., 0] > 0

    # Predict object classes [batch_size, num_patches, num_queries+1]
    (pred_logits, class_embeds) = owl_model.class_predictor(image_feats, original_text_embedding, query_mask)

    # Predict object boxes
    pred_boxes = owl_model.box_predictor(image_feats, image_embeds)

    return OwlViTObjectDetectionOutput(
                image_embeds=image_embeds,
                text_embeds=original_text_embedding,
                pred_boxes=pred_boxes,
                logits=pred_logits,
                class_embeds=class_embeds,
                text_model_output=None,
                vision_model_output=None,
            )