import os
import sys
import torch
from PIL import Image
from accelerate import infer_auto_device_map
from transformers import AutoConfig

sys.path.insert(0, "/home/pavitra/satquery")
from GeoChat.geochat.model.builder import load_pretrained_model
from GeoChat.geochat.mm_utils import get_model_name_from_path, process_images_demo, tokenizer_image_token, KeywordsStoppingCriteria
from GeoChat.geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from GeoChat.geochat.conversation import conv_templates
from GeoChat.geochat.model.language_model.geochat_llama import GeoChatLlamaForCausalLM

def test_fp16_run():
    print("Testing FP16 generation with vision tower on GPU...")
    offload_dir = "/home/pavitra/satquery/scratch/offload_geochat"
    os.makedirs(offload_dir, exist_ok=True)

    model_path = "MBZUAI/geochat-7B"
    config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        skeleton = GeoChatLlamaForCausalLM(config)

    max_mem = {0: "4500MiB", "cpu": "14000MiB"}
    dmap = infer_auto_device_map(skeleton, max_memory=max_mem, no_split_module_classes=["LlamaDecoderLayer", "CLIPEncoderLayer"])
    dmap["model.vision_tower"] = 0
    dmap["model.mm_projector"] = 0

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=dmap,
        device="cuda",
        offload_folder=offload_dir,
    )
    print("Model loaded successfully!")
    print("Vision tower device:", next(model.get_vision_tower().parameters()).device)

    # Test single generation
    image_path = "data/SECOND/im2/00003.png"
    query = "What objects and features are visible in this satellite image?"

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

    print("Generating...")
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
    print("DECODED ANSWER:")
    print(raw_answer)

if __name__ == "__main__":
    test_fp16_run()
