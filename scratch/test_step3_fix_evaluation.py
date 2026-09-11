import os
import sys
import json
import torch
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from scripts.geochat_specialist import GeoChatSpecialist, KeywordsStoppingCriteria, process_images_demo, conv_templates, DEFAULT_IMAGE_TOKEN
from geochat.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX

def run_single(specialist, image_path, query, no_repeat_ngram_size=3, repetition_penalty=1.05):
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = process_images_demo([pil_image], specialist.image_processor)
    image_tensor = image_tensor.to(device=f"cuda:{specialist.GPU_ID}", dtype=torch.float16)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
    conv.messages[-1][1] = conv.messages[-1][1] + " " + query
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, specialist.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(specialist.model.device)
    input_ids[input_ids < 0] = 0

    stop_str = conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], specialist.tokenizer, input_ids)

    kwargs = {
        "input_ids": input_ids,
        "images": image_tensor,
        "past_key_values": None,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": 300,
        "use_cache": True,
        "stopping_criteria": [stopping_criteria],
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.inference_mode():
        outputs = specialist.model.generate(**kwargs)

    generated_ids = outputs.sequences[0, input_ids.shape[1]:]
    raw_answer = specialist.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if raw_answer.endswith(stop_str):
        raw_answer = raw_answer[:-len(stop_str)].strip()

    confidence = specialist._compute_confidence(outputs.scores)
    tokens = specialist.tokenizer.encode(raw_answer, add_special_tokens=False)
    
    return {
        "image": image_path,
        "answer": raw_answer,
        "confidence": confidence,
        "tokens": len(tokens),
        "building_count": raw_answer.count("building"),
    }

def main():
    print("=" * 80)
    print("EVALUATING STEP 3(a): no_repeat_ngram_size=3 FIX")
    print("=" * 80)

    sp = GeoChatSpecialist()
    sp.load()
    query = "What objects and features are visible in this satellite image?"

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

    print("\n--- Running 10-Image Sweep with no_repeat_ngram_size=3 in exact sequence ---")
    results = []
    for i, p in enumerate(image_paths_10, 1):
        res = run_single(sp, p, query, no_repeat_ngram_size=3, repetition_penalty=1.05)
        print(f"[{i:02d}/10] {p}: tokens={res['tokens']}, buildings={res['building_count']}, conf={res['confidence']:.4f}")
        print(f"       Output: {repr(res['answer'])}")
        results.append(res)

    print("\n--- Testing 00018.png 3 times with no_repeat_ngram_size=3 ---")
    runs_18 = []
    for r in range(1, 4):
        res_18 = run_single(sp, "data/SECOND/im2/00018.png", query, no_repeat_ngram_size=3, repetition_penalty=1.05)
        print(f"Run {r} for 00018.png: tokens={res_18['tokens']}, buildings={res_18['building_count']}, conf={res_18['confidence']:.4f}")
        print(f"       Output: {repr(res_18['answer'])}")
        runs_18.append(res_18)

    sp.unload()

    with open("scratch/results_step3_ngram3.json", "w") as f:
        json.dump({"sweep_10": results, "runs_00018": runs_18}, f, indent=2)

if __name__ == "__main__":
    main()
