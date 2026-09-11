import os
import sys
import glob
import json
import torch

sys.path.insert(0, "/home/pavitra/satquery")

def step1_raw_checkpoint_investigation():
    print("=====================================================================")
    print(" STEP 1: Direct State Dict Analysis of MBZUAI/geochat-7B Checkpoint")
    print("=====================================================================")

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--MBZUAI--geochat-7B")
    index_files = glob.glob(f"{cache_dir}/**/pytorch_model.bin.index.json", recursive=True)
    assert index_files, f"Index file not found in {cache_dir}"
    index_path = index_files[0]
    checkpoint_dir = os.path.dirname(index_path)

    print(f"Checkpoint index path: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    all_keys = list(weight_map.keys())

    # Filter keys belonging to the vision tower
    vision_tower_keys = [k for k in all_keys if "vision_tower" in k]
    total_vt_keys = len(vision_tower_keys)

    print(f"\nTotal keys in checkpoint: {len(all_keys)}")
    print(f"Total vision_tower keys in checkpoint: {total_vt_keys}")

    print("\n--- FIRST 20 VISION TOWER KEYS IN CHECKPOINT ---")
    for i, k in enumerate(vision_tower_keys[:20]):
        shard = weight_map[k]
        print(f"[{i+1:02d}] {k} (Shard: {shard})")

    # Select 3 distinct representative keys across embeddings, encoder layers, and layernorms
    sample_keys = [
        "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_tower.vision_tower.vision_model.post_layernorm.weight",
    ]

    print("\n--- TENSOR STATS FOR 3 REPRESENTATIVE VISION TOWER KEYS ---")
    shards_to_load = {}
    for k in sample_keys:
        shard_file = weight_map[k]
        if shard_file not in shards_to_load:
            shards_to_load[shard_file] = []
        shards_to_load[shard_file].append(k)

    for shard_file, keys in shards_to_load.items():
        shard_path = os.path.join(checkpoint_dir, shard_file)
        print(f"\nLoading shard: {shard_file} ({shard_path}) ...")
        # Load shard directly
        state_dict = torch.load(shard_path, map_location="cpu")
        for k in keys:
            t = state_dict[k]
            print(f"  Key:  {k}")
            print(f"  Type: {t.dtype} | Device: {t.device}")
            print(f"  Shape: {list(t.shape)}")
            print(f"  Mean:  {t.float().mean().item():.8f}")
            print(f"  Std:   {t.float().std().item():.8f}")
            print(f"  Min:   {t.float().min().item():.8f}")
            print(f"  Max:   {t.float().max().item():.8f}")
            print(f"  All-zero? {(t == 0).all().item()} | Contains NaN? {torch.isnan(t).any().item()}")

    # Now verify with transformers from_pretrained unexpected keys list
    print("\n--- CATCHING UNEXPECTED KEYS FROM AutoModel / GeoChatLlamaForCausalLM ---")
    from GeoChat.geochat.model.language_model.geochat_llama import GeoChatLlamaForCausalLM, GeoChatConfig

    config = GeoChatConfig.from_pretrained("MBZUAI/geochat-7B")
    # instantiate model skeleton with current delay_load=True architecture
    with torch.device("meta"):
        skeleton_model = GeoChatLlamaForCausalLM(config)

    model_submodule_keys = set(skeleton_model.state_dict().keys())
    unexpected_keys = [k for k in all_keys if k not in model_submodule_keys]
    unexpected_vt_keys = [k for k in unexpected_keys if "vision_tower" in k]

    print(f"Total unexpected keys compared to skeleton: {len(unexpected_keys)}")
    print(f"Total unexpected vision_tower keys: {len(unexpected_vt_keys)}")
    print(f"Matches 391 count? {len(unexpected_vt_keys) == 391}")

if __name__ == "__main__":
    step1_raw_checkpoint_investigation()
