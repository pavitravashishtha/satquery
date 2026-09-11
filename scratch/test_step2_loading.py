import sys, os, torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.insert(0, "/home/pavitra/satquery")
from GeoChat.geochat.model.language_model.geochat_llama import GeoChatLlamaForCausalLM, GeoChatConfig
from GeoChat.geochat.model.multimodal_encoder.clip_encoder import CLIPVisionTower

class VisionModelWrapper(nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)

def test_loading():
    print("Testing Step 2 loading mechanism...")
    config = GeoChatConfig.from_pretrained("MBZUAI/geochat-7B")
    
    # 1. Vanilla CLIP inspection
    vanilla_clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", use_safetensors=True)
    vanilla_keys = list(vanilla_clip.state_dict().keys())
    print(f"Vanilla CLIPVisionModel total keys: {len(vanilla_keys)}")
    print(f"Vanilla CLIPVisionModel first 5 keys: {vanilla_keys[:5]}")

    # 2. Check wrapped vision tower keys
    vt = CLIPVisionTower(config.mm_vision_tower, args=config, delay_load=True)
    vt.image_processor = CLIPImageProcessor.from_pretrained(config.mm_vision_tower)
    vt.vision_tower = VisionModelWrapper(vanilla_clip)
    vt.is_loaded = True

    vt_keys = list(vt.state_dict().keys())
    print(f"\nWrapped CLIPVisionTower total keys: {len(vt_keys)}")
    print(f"Wrapped CLIPVisionTower first 10 keys:")
    for k in vt_keys[:10]:
        print(f"  {k}")

    # Verify matching with checkpoint keys
    import glob, json
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--MBZUAI--geochat-7B")
    with open(glob.glob(cache_dir + "/**/pytorch_model.bin.index.json", recursive=True)[0]) as f:
        weight_map = json.load(f)["weight_map"]
    
    ckpt_vt_keys = [k for k in weight_map if "vision_tower" in k]
    print(f"\nTotal vision_tower keys in checkpoint: {len(ckpt_vt_keys)}")

    expected_full_keys = [f"model.vision_tower.{k}" for k in vt_keys]
    mismatch = set(ckpt_vt_keys) ^ set(expected_full_keys)
    print(f"Key mismatch count between checkpoint and wrapped model: {len(mismatch)}")
    assert len(mismatch) == 0, f"Mismatched keys: {mismatch}"
    print("SUCCESS: Every single vision_tower key in the checkpoint matches the wrapped model!")

if __name__ == "__main__":
    test_loading()
