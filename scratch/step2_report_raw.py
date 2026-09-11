import os
import sys
import glob
import json
import torch
from transformers import CLIPVisionModel

sys.path.insert(0, "/home/pavitra/satquery")
from GeoChat.geochat.model.builder import load_pretrained_model
from GeoChat.geochat.mm_utils import get_model_name_from_path

def run_step2_verification():
    print("=====================================================================")
    print(" STEP 2: RAW REPORT - Vision Tower Fix Verification")
    print("=====================================================================")

    # 1. Vanilla CLIP inspection
    print("\n--- 1. VANILLA CLIP (openai/clip-vit-large-patch14-336) ---")
    vanilla_clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", use_safetensors=True)
    vanilla_keys = list(vanilla_clip.state_dict().keys())
    print(f"Vanilla CLIP total key count: {len(vanilla_keys)}")
    print("Vanilla CLIP first 10 keys:")
    for i, k in enumerate(vanilla_keys[:10]):
        print(f"  [{i+1:02d}] {k}")

    # 2. Reload GeoChat model with Step 2 fix in FP16 with offload
    print("\n--- 2. RELOADING GEOCHAT MODEL WITH STEP 2 FIX ---")
    offload_dir = "/home/pavitra/satquery/scratch/offload_geochat"
    os.makedirs(offload_dir, exist_ok=True)

    model_path = "MBZUAI/geochat-7B"
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device="cuda",
        offload_folder=offload_dir,
    )

    vt_sd = model.get_vision_tower().state_dict()
    vt_keys = list(vt_sd.keys())

    print(f"\nFixed vision_tower.state_dict() total key count: {len(vt_keys)}")
    print("Fixed vision_tower first 10 keys:")
    for i, k in enumerate(vt_keys[:10]):
        print(f"  [{i+1:02d}] {k}")

    # 3. Compare with Checkpoint Keys from Step 1
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--MBZUAI--geochat-7B")
    with open(glob.glob(cache_dir + "/**/pytorch_model.bin.index.json", recursive=True)[0]) as f:
        weight_map = json.load(f)["weight_map"]
    ckpt_vt_keys = [k for k in weight_map if "vision_tower" in k]

    expected_full_keys = [f"model.vision_tower.{k}" for k in vt_keys]
    mismatches = set(ckpt_vt_keys) ^ set(expected_full_keys)

    print("\n--- 3. COMPARISON WITH STEP 1 CHECKPOINT KEYS ---")
    print(f"Checkpoint vision_tower key count: {len(ckpt_vt_keys)}")
    print(f"Fixed vision_tower key count:      {len(vt_keys)}")
    print(f"Mismatched keys count:             {len(mismatches)}")
    print(f"All 391 checkpoint keys match?     {len(mismatches) == 0}")

    # Clean up offload
    import shutil
    shutil.rmtree(offload_dir, ignore_errors=True)

if __name__ == "__main__":
    run_step2_verification()
