from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput
import numpy as np
import torch
from PIL import Image
import torch.nn as nn

from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from transformers.models.llava.modeling_llava import LlavaConfig

"""
These projection modules take the output of the vision tower of Owl-Vit and align it to LLaVa's vision input.
"""

class CopyProjectionModule(nn.Module):
    """
    This is a projection module that copies the structure of the LLaVa projection module used to align OpenAI CLIP to LLaMa.
    """
    def __init__(self, pth_path = "/home/lorisg96/llm_draw/models/llava_projector/projector_oi.pth"):
        super(CopyProjectionModule, self).__init__()
        config = LlavaConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
        new_token_projection = LlavaMultiModalProjector(config=config)
        new_token_projection.linear_1 = torch.nn.Linear(in_features=768, out_features=4096, bias=True)
        new_token_projection.load_state_dict(torch.load(pth_path))
        self.token_projection = new_token_projection

    def forward(self, *args, **kwargs):
        return self.token_projection(*args, **kwargs)

class TransformerProjectionModule(nn.Module):
    """
    This is a projection module that uses a small transformer (performance seems to be poor)
    """
    def __init__(self, input_dim=768, hidden_dim=4096, num_layers=2, output_dim=4096, pth_path = "/home/lorisg96/llm_draw/models/llava_projector/projector_laion.pth"):
        super(TransformerProjectionModule, self).__init__()
    
        # Linear layer to reduce input dimension to hidden_dim
        self.linear_in = nn.Linear(input_dim, hidden_dim)
    
        # Transformer Encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
    
        # Final Linear layer to project to output_dim
        self.linear_out = nn.Linear(hidden_dim, output_dim)

        # Load state dict
        self.load_state_dict(torch.load(pth_path))
    
    def forward(self, x):
        # Input projection
        x = self.linear_in(x)
    
        # Transformer layers
        x = self.transformer_encoder(x)
    
        # Output projection
        x = self.linear_out(x)
    
        return x

class DeeperMLPProjectionModule(nn.Module):
    """
    This is a projection module that uses a deeper MLP than the original from LLaVa
    """
    def __init__(self, input_dim=768, hidden_dim=8192, num_hidden_layers=2, output_dim=4096, pth_path = "/home/lorisg96/llm_draw/models/llava_projector/projector_deepermlp.pth"):
        super(DeeperMLPProjectionModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Define input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        
        # Define output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Load state dict
        self.load_state_dict(torch.load(pth_path))
        
    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))  # Activation function for input layer
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))  # Activation function for hidden layers
        
        x = self.output_layer(x)  # No activation function for output layer
        
        return x