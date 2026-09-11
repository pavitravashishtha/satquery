import os
import sys
import json
import torch
import gc
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from scripts.geochat_specialist import GeoChatSpecialist, KeywordsStoppingCriteria, process_images_demo, conv_templates, DEFAULT_IMAGE_TOKEN
from geochat.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX

image_paths_10 = [
    "data/SECOND/im2/00013.png",
    "data/SECOND/im2/00015.png",
    "data/SECOND/im2/00016.png",
    "data/SECOND/im2/00017.png",
    "data/SECOND/im2/00018.png",
    "data/SECOND/im2/00020.png",
    "data/SECOND/im2/00021.png",
    "data/SECOND/im2/00025.png",
    "data/SECOND/im2/00026.png",
    "data/SECOND/im2/00028.png",
]
query = "What objects and features are visible in this satellite image?"

def run_test():
    sp = GeoChatSpecialist()
    sp.load()

    print("=" * 80)
    print("RUNNING 10-IMAGE SWEEP WITH repetition_penalty=1.15")
    print("=" * 80)

    results = []
    for i, img_path in enumerate(image_paths_10, 1):
        pil_image = Image.open(img_path).convert("RGB")
        image_tensor = process_images_demo([pil_image], sp.image_processor)
        image_tensor = image_tensor.to(device=f"cuda:{sp.GPU_ID}", dtype=torch.float16)

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
        conv.messages[-1][1] = conv.messages[-1][1] + " " + query
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, sp.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(sp.model.device)
        input_ids[input_ids < 0] = 0

        stop_str = conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], sp.tokenizer, input_ids)

        with torch.inference_mode():
            outputs = sp.model.generate(
                input_ids,
                images=image_tensor,
                past_key_values=None,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.15,
                max_new_tokens=300,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[0, input_ids.shape[1]:]
        raw_answer = sp.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if raw_answer.endswith(stop_str):
            raw_answer = raw_answer[:-len(stop_str)].strip()

        confidence = sp._compute_confidence(outputs.scores)
        tokens = sp.tokenizer.encode(raw_answer, add_special_tokens=False)
        bldg_cnt = raw_answer.count("building")

        # Cleanup tensors immediately to keep VRAM minimal
        del outputs, image_tensor, input_ids
        torch.cuda.empty_cache()

        print(f"\n[{i:02d}/10] {img_path}")
        print(f"  Confidence: {confidence:.4f} | Tokens: {len(tokens)} | 'building' count: {bldg_cnt}")
        print(f"  Output: {repr(raw_answer)}")

        results.append({
            "index": i,
            "image_path": img_path,
            "decoded_output": raw_answer,
            "tokens": len(tokens),
            "building_count": bldg_cnt,
            "confidence": round(confidence, 4),
        })

    print("\n" + "=" * 80)
    print("TESTING 00018.png 3 CONSECUTIVE TIMES WITH repetition_penalty=1.15")
    print("=" * 80)
    runs_18 = []
    for r in range(1, 4):
        pil_image = Image.open("data/SECOND/im2/00018.png").convert("RGB")
        image_tensor = process_images_demo([pil_image], sp.image_processor).to(device=f"cuda:{sp.GPU_ID}", dtype=torch.float16)

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
        conv.messages[-1][1] = conv.messages[-1][1] + " " + query
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, sp.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(sp.model.device)
        input_ids[input_ids < 0] = 0

        stopping_criteria = KeywordsStoppingCriteria([conv.sep2], sp.tokenizer, input_ids)

        with torch.inference_mode():
            outputs = sp.model.generate(
                input_ids,
                images=image_tensor,
                past_key_values=None,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.15,
                max_new_tokens=300,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[0, input_ids.shape[1]:]
        raw_answer = sp.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if raw_answer.endswith(conv.sep2):
            raw_answer = raw_answer[:-len(conv.sep2)].strip()

        confidence = sp._compute_confidence(outputs.scores)
        tokens = sp.tokenizer.encode(raw_answer, add_special_tokens=False)
        bldg_cnt = raw_answer.count("building")

        del outputs, image_tensor, input_ids
        torch.cuda.empty_cache()

        print(f"\nRun {r}/3 for 00018.png:")
        print(f"  Confidence: {confidence:.4f} | Tokens: {len(tokens)} | 'building' count: {bldg_cnt}")
        print(f"  Output: {repr(raw_answer)}")

        runs_18.append({
            "run": r,
            "decoded_output": raw_answer,
            "tokens": len(tokens),
            "building_count": bldg_cnt,
            "confidence": round(confidence, 4),
        })

    sp.unload()

    with open("scratch/results_repetition_penalty_115.json", "w") as f:
        json.dump({"sweep_10": results, "runs_00018": runs_18}, f, indent=2)

if __name__ == "__main__":
    run_test()
