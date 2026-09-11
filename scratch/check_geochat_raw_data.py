import os
import sys
import torch
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery")
from scripts.geochat_specialist import GeoChatSpecialist
from GeoChat.geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from GeoChat.geochat.conversation import conv_templates
from GeoChat.geochat.mm_utils import tokenizer_image_token, process_images_demo

def inspect_run(image_path, query, specialist):
    print(f"\n=======================================================")
    print(f"IMAGE: {image_path}")
    print(f"QUERY: {query}")
    print(f"=======================================================")

    # 1. Load image and process image tensor
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = process_images_demo([pil_image], specialist.image_processor)
    image_tensor = image_tensor.to(device=f"cuda:{specialist.GPU_ID}", dtype=torch.float16)

    # Question 1:
    has_nan = torch.isnan(image_tensor).any().item()
    is_all_zero = (image_tensor == 0).all().item()
    print("--- 1. IMAGE TENSOR STATS ---")
    print(f"image_tensor.shape: {list(image_tensor.shape)}")
    print(f"image_tensor.mean(): {image_tensor.mean().item():.6f}")
    print(f"image_tensor.std(): {image_tensor.std().item():.6f}")
    print(f"contains NaN: {has_nan}")
    print(f"is all-zero: {is_all_zero}")

    # Prepare prompt and input_ids as GeoChatSpecialist does
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
    conv.messages[-1][1] = conv.messages[-1][1] + " " + query
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    raw_input_ids = tokenizer_image_token(
        prompt, specialist.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device=f"cuda:{specialist.GPU_ID}")

    # Check position of IMAGE_TOKEN_INDEX before replacement
    image_token_positions_before = (raw_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).tolist()

    pad_token_id = specialist.tokenizer.pad_token_id if specialist.tokenizer.pad_token_id is not None else 0
    input_ids = torch.where(
        raw_input_ids == IMAGE_TOKEN_INDEX,
        torch.tensor(pad_token_id, device=raw_input_ids.device, dtype=raw_input_ids.dtype),
        raw_input_ids,
    )

    print("\n--- 2. INPUT_IDS & SPLICING DATA ---")
    print(f"raw_input_ids shape: {list(raw_input_ids.shape)}")
    print(f"final input_ids sequence length (passed to generate): {input_ids.shape[1]}")
    print(f"IMAGE_TOKEN_INDEX position(s) in input_ids before sentinel replace: {image_token_positions_before}")

    # Inspect the multimodal embedding splicing from Fix 4 code path
    # We call model.prepare_inputs_labels_for_multimodal directly to inspect exact splicing
    attention_mask = torch.ones_like(input_ids)
    (
        mm_input_ids,
        mm_attention_mask,
        mm_past_key_values,
        mm_inputs_embeds,
        mm_labels,
    ) = specialist.model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
    )

    if mm_inputs_embeds is not None:
        print(f"Multimodal inputs_embeds shape after splicing: {list(mm_inputs_embeds.shape)}")
        print(f"Spliced sequence length: {mm_inputs_embeds.shape[1]}")
    else:
        print(f"Multimodal inputs_embeds: None (input_ids returned: {mm_input_ids.shape})")

    # Run specialist.run to get the exact output from the model
    res = specialist.run(image=image_path, query=query, task_type="vqa")
    return res["answer"]

def main():
    specialist = GeoChatSpecialist()
    specialist.load()

    query = "What objects and features are visible in this satellite image?"
    img1 = "data/SECOND/im2/00003.png"
    img2 = "data/SECOND/im2/00011.png"

    output1 = inspect_run(img1, query, specialist)
    output2 = inspect_run(img2, query, specialist)

    print("\n=======================================================")
    print("--- 3. UNMODIFIED DECODED OUTPUTS SIDE BY SIDE ---")
    print("=======================================================")
    print(f"[Image 1: {img1}]")
    print(f"{repr(output1)}")
    print(f"\n[Image 2: {img2}]")
    print(f"{repr(output2)}")

if __name__ == "__main__":
    main()
