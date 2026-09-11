import os
import sys
import argparse
import torch
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path, process_images_demo, tokenizer_image_token, KeywordsStoppingCriteria
from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from geochat.conversation import conv_templates

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to image file")
parser.add_argument("--image-label", type=str, default="Image", help="Label for display")
parser.add_argument("--mode", type=str, default="generate", choices=["generate", "direct_encode", "both"])
args = parser.parse_args()

print("=" * 70)
print(f"DIAGNOSTIC RUN: {args.image_label} ({args.image})")
print(f"Mode: {args.mode}")
print("=" * 70)

# Hook telemetry
telemetry = {
    "vt_calls": 0,
    "vt_stats": [],
    "proj_calls": 0,
    "proj_stats": [],
    "prep_calls": 0,
    "prep_stats": [],
    "spliced_calls": 0,
    "spliced_stats": [],
}

model_path = "MBZUAI/geochat-7B"
model_name = get_model_name_from_path(model_path)

print(f"\n[LOAD] Loading model from {model_path} in 4-bit...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name,
    load_8bit=False,
    load_4bit=True,
    device="cuda",
)
model = model.eval()

# Check language model embedding baseline
embed_tokens_w = model.get_model().embed_tokens.weight
print(f"\n[BASELINE] Language Model embed_tokens.weight:")
print(f"  Shape: {list(embed_tokens_w.shape)}")
print(f"  Mean:  {embed_tokens_w.float().mean().item():.8f}")
print(f"  Std:   {embed_tokens_w.float().std().item():.8f}")
print(f"  Min:   {embed_tokens_w.float().min().item():.8f}")
print(f"  Max:   {embed_tokens_w.float().max().item():.8f}")

# Hook Step A: Vision Tower forward
vt = model.get_vision_tower()
inner_vm = getattr(vt.vision_tower, "vision_model", vt.vision_tower)

def vt_hook(module, inputs, output):
    telemetry["vt_calls"] += 1
    # output can be BaseModelOutputWithPoolingAndCrossAttentions or tuple
    feat = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    stats = {
        "call": telemetry["vt_calls"],
        "shape": list(feat.shape),
        "mean": feat.float().mean().item(),
        "std": feat.float().std().item(),
        "has_nan": torch.isnan(feat).any().item(),
        "has_inf": torch.isinf(feat).any().item(),
    }
    telemetry["vt_stats"].append(stats)
    print(f"\n>>> [HOOK Step A] Vision Tower forward call #{stats['call']}:")
    print(f"    Shape:   {stats['shape']}")
    print(f"    Mean:    {stats['mean']:.8f}")
    print(f"    Std:     {stats['std']:.8f}")
    print(f"    Has NaN: {stats['has_nan']} | Has Inf: {stats['has_inf']}")

inner_vm.register_forward_hook(vt_hook)

# Hook Step B: mm_projector forward
proj = model.get_model().mm_projector

def proj_hook(module, inputs, output):
    telemetry["proj_calls"] += 1
    stats = {
        "call": telemetry["proj_calls"],
        "shape": list(output.shape),
        "mean": output.float().mean().item(),
        "std": output.float().std().item(),
        "has_nan": torch.isnan(output).any().item(),
        "has_inf": torch.isinf(output).any().item(),
        "min": output.float().min().item(),
        "max": output.float().max().item(),
    }
    telemetry["proj_stats"].append(stats)
    print(f"\n>>> [HOOK Step B] mm_projector forward call #{stats['call']}:")
    print(f"    Shape:   {stats['shape']}")
    print(f"    Mean:    {stats['mean']:.8f}")
    print(f"    Std:     {stats['std']:.8f}")
    print(f"    Min:     {stats['min']:.8f} | Max: {stats['max']:.8f}")
    print(f"    Has NaN: {stats['has_nan']} | Has Inf: {stats['has_inf']}")

proj.register_forward_hook(proj_hook)

# Hook Step C: prepare_inputs_labels_for_multimodal
orig_prepare = model.prepare_inputs_labels_for_multimodal

def instrumented_prepare(input_ids, attention_mask, past_key_values, labels, images):
    telemetry["prep_calls"] += 1
    call_idx = telemetry["prep_calls"]
    input_shape = list(input_ids.shape) if input_ids is not None else None
    has_images = images is not None
    pkv_type = type(past_key_values).__name__ if past_key_values is not None else "None"
    pkv_len = len(past_key_values) if past_key_values is not None else 0
    pkv_seq = getattr(past_key_values, "get_seq_length", lambda: None)() if past_key_values is not None else None

    print(f"\n>>> [HOOK Step C Top] prepare_inputs_labels_for_multimodal() Call #{call_idx}:")
    print(f"    input_ids shape: {input_shape}")
    print(f"    images is None:  {not has_images}")
    print(f"    past_key_values: {pkv_type} (len={pkv_len}, seq_length={pkv_seq})")

    ret = orig_prepare(input_ids, attention_mask, past_key_values, labels, images)
    new_ids, new_mask, new_pkv, new_embeds, new_labels = ret

    if new_embeds is not None:
        telemetry["spliced_calls"] += 1
        stats = {
            "call": telemetry["spliced_calls"],
            "shape": list(new_embeds.shape),
            "mean": new_embeds.float().mean().item(),
            "std": new_embeds.float().std().item(),
            "min": new_embeds.float().min().item(),
            "max": new_embeds.float().max().item(),
        }
        telemetry["spliced_stats"].append(stats)
        print(f"    [HOOK Step C Splicing] Image features spliced into cur_new_input_embeds:")
        print(f"      new_embeds shape: {stats['shape']}")
        print(f"      new_embeds mean:  {stats['mean']:.8f}")
        print(f"      new_embeds std:   {stats['std']:.8f}")
        print(f"      new_embeds min:   {stats['min']:.8f} | max: {stats['max']:.8f}")
    else:
        print(f"    [HOOK Step C Splicing] new_embeds is NONE (multimodal processing SKIPPED)")

    return ret

model.prepare_inputs_labels_for_multimodal = instrumented_prepare

# Process Image
pil_image = Image.open(args.image).convert("RGB")
image_tensor = process_images_demo([pil_image], image_processor)
vision_device = next(model.get_vision_tower().parameters()).device
image_tensor = image_tensor.to(device=vision_device, dtype=torch.float16)

print(f"\n[IMAGE TENSOR]:")
print(f"  Shape: {list(image_tensor.shape)}")
print(f"  Mean:  {image_tensor.float().mean().item():.8f}")
print(f"  Std:   {image_tensor.float().std().item():.8f}")
print(f"  Min:   {image_tensor.float().min().item():.8f}")
print(f"  Max:   {image_tensor.float().max().item():.8f}")

if args.mode in ("direct_encode", "both"):
    print("\n" + "=" * 50)
    print("DIRECT ENCODE PASS (model.encode_images)")
    print("=" * 50)
    with torch.no_grad():
        encoded = model.encode_images(image_tensor)
    print(f"\nDirect encode output shape: {list(encoded.shape)}")
    print(f"Direct encode output mean:  {encoded.float().mean().item():.8f}")
    print(f"Direct encode output std:   {encoded.float().std().item():.8f}")

if args.mode in ("generate", "both"):
    print("\n" + "=" * 50)
    print("FULL model.generate() PASS")
    print("=" * 50)

    query = "What objects and features are visible in this satellite image?"
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

    print(f"Initial input_ids shape before generate: {list(input_ids.shape)}")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            past_key_values=None,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=100,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    generated_ids = outputs[0, input_ids.shape[1]:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if raw_answer.endswith(stop_str):
        raw_answer = raw_answer[:-len(stop_str)].strip()

    print(f"\nDECODED OUTPUT: \"{raw_answer}\"")

print("\n" + "=" * 50)
print("FINAL TELEMETRY SUMMARY")
print("=" * 50)
print(f"Vision tower calls:    {telemetry['vt_calls']}")
print(f"mm_projector calls:    {telemetry['proj_calls']}")
print(f"prepare_inputs calls:  {telemetry['prep_calls']}")
print(f"Features spliced count: {telemetry['spliced_calls']}")
