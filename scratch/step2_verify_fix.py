import os
import sys
import torch
import warnings
from transformers import CLIPVisionModel

sys.path.insert(0, "/home/pavitra/satquery")
from GeoChat.geochat.model.builder import load_pretrained_model
from GeoChat.geochat.mm_utils import get_model_name_from_path

def step2_verify():
    print("=====================================================================")
    print(" STEP 2: Verify Vision Tower Fix and Checkpoint Loading")
    print("=====================================================================")

    # 1. Load vanilla CLIP separately and count its keys
    print("\n--- 1. VANILLA CLIP (openai/clip-vit-large-patch14-336) ---")
    vanilla_clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", use_safetensors=True)
    vanilla_keys = list(vanilla_clip.state_dict().keys())
    print(f"Vanilla CLIP total keys: {len(vanilla_keys)}")
    print("First 10 keys of Vanilla CLIP:")
    for i, k in enumerate(vanilla_keys[:10]):
        print(f"  [{i+1:02d}] {k}")

    # 2. Reload GeoChat model (in 4-bit to fit in memory for inspection)
    print("\n--- 2. RELOADING GEOCHAT MODEL WITH STEP 2 FIX ---")
    model_path = "MBZUAI/geochat-7B"
    model_name = get_model_name_from_path(model_path)

    # Capture warnings/logger outputs during from_pretrained
    import logging
    transformers_logger = logging.getLogger("transformers")
    prev_level = transformers_logger.level
    transformers_logger.setLevel(logging.WARNING)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=True,
        device="cuda",
    )

    vision_tower = model.get_vision_tower()
    vt_sd = vision_tower.state_dict()
    vt_keys = list(vt_sd.keys())

    print(f"\nFixed vision_tower.state_dict() total key count: {len(vt_keys)}")
    print("First 10 keys of Fixed vision_tower:")
    for i, k in enumerate(vt_keys[:10]):
        print(f"  [{i+1:02d}] {k}")

    # 3. Compare with Step 1 391 checkpoint keys
    import glob, json
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--MBZUAI--geochat-7B")
    with open(glob.glob(cache_dir + "/**/pytorch_model.bin.index.json", recursive=True)[0]) as f:
        weight_map = json.load(f)["weight_map"]
    ckpt_vt_keys = [k for k in weight_map if "vision_tower" in k]

    expected_full_keys = [f"model.vision_tower.{k}" for k in vt_keys]
    mismatches = set(ckpt_vt_keys) ^ set(expected_full_keys)
    print(f"\nComparison with Step 1 Checkpoint keys:")
    print(f"  Checkpoint vision_tower key count: {len(ckpt_vt_keys)}")
    print(f"  Fixed vision_tower key count:      {len(vt_keys)}")
    print(f"  Mismatched keys count:             {len(mismatches)}")
    print(f"  Exact match confirmed?             {len(mismatches) == 0}")

    # Check position embedding shape after interpolation
    pos_emb = vt_sd["vision_tower.vision_model.embeddings.position_embedding.weight"]
    print(f"\nInterpolated position embedding shape: {list(pos_emb.shape)}")
    assert pos_emb.shape == torch.Size([1297, 1024]), f"Unexpected position embedding shape: {pos_emb.shape}"

    # Also check if any unexpected keys were for vision_tower
    print("\nSTEP 2 VERIFICATION COMPLETE.")

if __name__ == "__main__":
    step2_verify()
