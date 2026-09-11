import os
import sys
import torch
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery")
from GeoChat.geochat.model.builder import load_pretrained_model
from GeoChat.geochat.mm_utils import get_model_name_from_path, process_images_demo, tokenizer_image_token, KeywordsStoppingCriteria
from GeoChat.geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from GeoChat.geochat.conversation import conv_templates

# Global telemetry tracking
hook_data = {
    "vision_tower_calls": [],
    "projector_calls": [],
    "multimodal_prepare_calls": [],
    "spliced_features": [],
}

def attach_hooks(model):
    # STEP A Hook: Vision Tower forward
    vt = model.get_vision_tower()
    # The inner vision model inside VisionModelWrapper
    inner_vm = getattr(vt.vision_tower, "vision_model", vt.vision_tower)

    def vt_hook(module, inputs, output):
        # output is BaseModelOutputWithPoolingAndCrossAttentions or similar
        # hidden_states are selected by feature_select
        last_hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        stats = {
            "call_idx": len(hook_data["vision_tower_calls"]) + 1,
            "shape": list(last_hidden.shape),
            "mean": last_hidden.float().mean().item(),
            "std": last_hidden.float().std().item(),
            "has_nan": torch.isnan(last_hidden).any().item(),
        }
        hook_data["vision_tower_calls"].append(stats)
        print(f"\n[HOOK Step A] Vision Tower forward call #{stats['call_idx']}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean:  {stats['mean']:.8f}")
        print(f"  Std:   {stats['std']:.8f}")
        print(f"  NaN?:  {stats['has_nan']}")

    inner_vm.register_forward_hook(vt_hook)

    # STEP B Hook: mm_projector forward
    proj = model.get_model().mm_projector

    def proj_hook(module, inputs, output):
        stats = {
            "call_idx": len(hook_data["projector_calls"]) + 1,
            "shape": list(output.shape),
            "mean": output.float().mean().item(),
            "std": output.float().std().item(),
            "has_nan": torch.isnan(output).any().item(),
            "has_inf": torch.isinf(output).any().item(),
        }
        hook_data["projector_calls"].append(stats)
        print(f"\n[HOOK Step B] mm_projector forward call #{stats['call_idx']}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean:  {stats['mean']:.8f}")
        print(f"  Std:   {stats['std']:.8f}")
        print(f"  NaN?:  {stats['has_nan']} | Inf?: {stats['has_inf']}")

    proj.register_forward_hook(proj_hook)

    # STEP C Hooks on prepare_inputs_labels_for_multimodal
    orig_prepare = model.prepare_inputs_labels_for_multimodal

    def instrumented_prepare(input_ids, attention_mask, past_key_values, labels, images):
        call_num = len(hook_data["multimodal_prepare_calls"]) + 1
        print(f"\n[HOOK Step C] prepare_inputs_labels_for_multimodal() CALLED (Call #{call_num})")
        print(f"  input_ids.shape:       {list(input_ids.shape) if input_ids is not None else None}")
        print(f"  images is None?:       {images is None}")
        print(f"  past_key_values None?: {past_key_values is None}")
        hook_data["multimodal_prepare_calls"].append(call_num)

        ret = orig_prepare(input_ids, attention_mask, past_key_values, labels, images)
        new_ids, new_mask, new_pkv, new_embeds, new_labels = ret
        if new_embeds is not None:
            stats = {
                "embeds_shape": list(new_embeds.shape),
                "embeds_mean": new_embeds.float().mean().item(),
                "embeds_std": new_embeds.float().std().item(),
            }
            hook_data["spliced_features"].append(stats)
            print(f"[HOOK Step C] Multimodal embeddings output:")
            print(f"  new_embeds.shape: {stats['embeds_shape']}")
            print(f"  new_embeds.mean:  {stats['embeds_mean']:.8f}")
            print(f"  new_embeds.std:   {stats['embeds_std']:.8f}")
        else:
            print(f"[HOOK Step C] new_embeds is NONE (input_ids returned unchanged)")
        return ret

    model.prepare_inputs_labels_for_multimodal = instrumented_prepare

def run_isolated_test(image_path, query, label):
    print(f"\n=====================================================================")
    print(f" FRESH INSTANCE TEST: {label} ({image_path})")
    print(f"=====================================================================")

    # Reset telemetry for this run
    hook_data["vision_tower_calls"].clear()
    hook_data["projector_calls"].clear()
    hook_data["multimodal_prepare_calls"].clear()
    hook_data["spliced_features"].clear()

    model_path = "MBZUAI/geochat-7B"
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=True,
        device="cuda",
    )
    model = model.eval()
    attach_hooks(model)

    # Check token embeddings range for Step B
    tok_emb = model.get_model().embed_tokens.weight
    print(f"\n[Language Model embed_tokens Baseline Stats]:")
    print(f"  Shape: {list(tok_emb.shape)}")
    print(f"  Mean:  {tok_emb.float().mean().item():.8f}")
    print(f"  Std:   {tok_emb.float().std().item():.8f}")

    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = process_images_demo([pil_image], image_processor)
    vision_device = next(model.get_vision_tower().parameters()).device
    image_tensor = image_tensor.to(device=vision_device, dtype=torch.float16)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
    conv.messages[-1][1] = conv.messages[-1][1] + " " + query
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    first_device = next(model.parameters()).device
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device=first_device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.where(
        input_ids == IMAGE_TOKEN_INDEX,
        torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
        input_ids,
    )

    stop_str = conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    print("\nCalling model.generate()...")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            past_key_values=None,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=60,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    generated_ids = outputs[0, input_ids.shape[1]:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if raw_answer.endswith(stop_str):
        raw_answer = raw_answer[:-len(stop_str)].strip()

    print(f"\nDECODED OUTPUT: \"{raw_answer}\"")

    # Capture snapshot
    snapshot = {
        "label": label,
        "image": image_path,
        "answer": raw_answer,
        "vt": list(hook_data["vision_tower_calls"]),
        "proj": list(hook_data["projector_calls"]),
        "prep_calls": list(hook_data["multimodal_prepare_calls"]),
        "spliced": list(hook_data["spliced_features"]),
    }

    # Clean up model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return snapshot

def main():
    query = "What objects and features are visible in this satellite image?"
    img1 = "data/SECOND/im2/00003.png"
    img2 = "data/SECOND/im2/00011.png"

    # Step A & B & C on FRESH instances
    res1 = run_isolated_test(img1, query, "Image 1 (Fresh Instance)")
    res2 = run_isolated_test(img2, query, "Image 2 (Fresh Instance)")

    print("\n" + "=" * 70)
    print("--- COMPARISON: STEP A (Vision Tower Output Stats) ---")
    print("=" * 70)
    vt1 = res1["vt"][0] if res1["vt"] else None
    vt2 = res2["vt"][0] if res2["vt"] else None
    print(f"Image 1 Vision Tower: Mean={vt1['mean']:.8f}, Std={vt1['std']:.8f}, Shape={vt1['shape']}")
    print(f"Image 2 Vision Tower: Mean={vt2['mean']:.8f}, Std={vt2['std']:.8f}, Shape={vt2['shape']}")
    diff_mean = abs(vt1['mean'] - vt2['mean'])
    diff_std = abs(vt1['std'] - vt2['std'])
    print(f"Mean Difference: {diff_mean:.8f} | Std Difference: {diff_std:.8f}")
    print(f"Are stats different? {diff_mean > 1e-6 or diff_std > 1e-6}")

    print("\n" + "=" * 70)
    print("--- COMPARISON: STEP B (mm_projector Output Stats) ---")
    print("=" * 70)
    p1 = res1["proj"][0] if res1["proj"] else None
    p2 = res2["proj"][0] if res2["proj"] else None
    print(f"Image 1 Projector: Mean={p1['mean']:.8f}, Std={p1['std']:.8f}, Shape={p1['shape']}")
    print(f"Image 2 Projector: Mean={p2['mean']:.8f}, Std={p2['std']:.8f}, Shape={p2['shape']}")
    p_diff_mean = abs(p1['mean'] - p2['mean'])
    p_diff_std = abs(p1['std'] - p2['std'])
    print(f"Mean Difference: {p_diff_mean:.8f} | Std Difference: {p_diff_std:.8f}")
    print(f"Are stats different? {p_diff_mean > 1e-6 or p_diff_std > 1e-6}")

    print("\n" + "=" * 70)
    print("--- STEP C: Multimodal Splicing in generate() ---")
    print("=" * 70)
    print(f"Image 1 prepare_inputs_labels_for_multimodal calls count: {len(res1['prep_calls'])}")
    print(f"Image 2 prepare_inputs_labels_for_multimodal calls count: {len(res2['prep_calls'])}")
    print(f"Image 1 spliced shape: {res1['spliced'][0]['embeds_shape'] if res1['spliced'] else 'None'}")
    print(f"Image 2 spliced shape: {res2['spliced'][0]['embeds_shape'] if res2['spliced'] else 'None'}")

if __name__ == "__main__":
    main()
