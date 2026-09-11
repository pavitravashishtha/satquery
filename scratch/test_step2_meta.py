import sys, os, torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor
from accelerate import init_empty_weights

class VisionModelWrapper(nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)

cfg = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336")

with init_empty_weights():
    vm = CLIPVisionModel(cfg)
    wrapper = VisionModelWrapper(vm)

print("Created skeleton in init_empty_weights successfully!")
print("Skeleton keys count:", len(wrapper.state_dict()))
print("First 5 keys:", list(wrapper.state_dict().keys())[:5])
