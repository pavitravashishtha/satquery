import os
import sys
import json
import torch
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

def run_sweep(sp, gen_kwargs_extra, name):
    print(f"\n{'='*70}\nRunning Sweep: {name} ({gen_kwargs_extra})\n{'='*70}")
    results = []
    for i, img_path in enumerate(image_paths_10, 1):
        pil_image = Image.open(img_path).convert("RGB")
        img_t = process_images_demo([pil_image], sp.image_processor).to(device=f"cuda:{sp.GPU_ID}", dtype=torch.float16)
        
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
        conv.messages[-1][1] = conv.messages[-1][1] + " " + query
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        ids = tokenizer_image_token(prompt, sp.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(sp.model.device)
        ids[ids < 0] = 0
        sc = KeywordsStoppingCriteria([conv.sep2], sp.tokenizer, ids)

        kwargs = {
            "input_ids": ids,
            "images": img_t,
            "past_key_values": None,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "max_new_tokens": 300,
            "use_cache": True,
            "stopping_criteria": [sc],
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        kwargs.update(gen_kwargs_extra)

        with torch.inference_mode():
            outputs = sp.model.generate(**kwargs)

        gen_ids = outputs.sequences[0, ids.shape[1]:]
        raw_answer = sp.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if raw_answer.endswith(conv.sep2):
            raw_answer = raw_answer[:-len(conv.sep2)].strip()

        conf = sp._compute_confidence(outputs.scores)
        tokens = sp.tokenizer.encode(raw_answer, add_special_tokens=False)
        bldg_cnt = raw_answer.count("building")

        print(f"[{i:02d}/10] {img_path} | tok={len(tokens):3d} | bldg={bldg_cnt:2d} | conf={conf:.4f}")
        print(f"    {repr(raw_answer)}")

        results.append({
            "image": img_path,
            "answer": raw_answer,
            "tokens": len(tokens),
            "building_count": bldg_cnt,
            "confidence": conf
        })
    return results

def main():
    sp = GeoChatSpecialist()
    sp.load()

    # Test 1: repetition_penalty=1.15
    res_rp115 = run_sweep(sp, {"repetition_penalty": 1.15}, "repetition_penalty=1.15")

    # Test 2: repetition_penalty=1.20
    res_rp120 = run_sweep(sp, {"repetition_penalty": 1.20}, "repetition_penalty=1.20")

    # Test 3: no_repeat_ngram_size=4
    res_ngram4 = run_sweep(sp, {"repetition_penalty": 1.05, "no_repeat_ngram_size": 4}, "no_repeat_ngram_size=4")

    # Re-run failing image 00018.png 3 times under repetition_penalty=1.15
    print("\n" + "="*70 + "\nRe-running 00018.png 3 times under repetition_penalty=1.15\n" + "="*70)
    for r in range(1, 4):
        p = "data/SECOND/im2/00018.png"
        pil_image = Image.open(p).convert("RGB")
        img_t = process_images_demo([pil_image], sp.image_processor).to(device=f"cuda:{sp.GPU_ID}", dtype=torch.float16)
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
        conv.messages[-1][1] = conv.messages[-1][1] + " " + query
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, sp.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(sp.model.device)
        ids[ids < 0] = 0
        sc = KeywordsStoppingCriteria([conv.sep2], sp.tokenizer, ids)
        with torch.inference_mode():
            outputs = sp.model.generate(
                input_ids=ids, images=img_t, past_key_values=None, do_sample=True,
                temperature=0.6, top_p=0.9, repetition_penalty=1.15, max_new_tokens=300,
                use_cache=True, stopping_criteria=[sc], output_scores=True, return_dict_in_generate=True
            )
        gen_ids = outputs.sequences[0, ids.shape[1]:]
        raw_answer = sp.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        conf = sp._compute_confidence(outputs.scores)
        print(f"Run {r}: tok={len(gen_ids)}, bldg={raw_answer.count('building')}, conf={conf:.4f}")
        print(f"    {repr(raw_answer)}")

    sp.unload()

    with open("scratch/sweep_comparisons.json", "w") as f:
        json.dump({
            "rp_115": res_rp115,
            "rp_120": res_rp120,
            "ngram4": res_ngram4,
        }, f, indent=2)

if __name__ == "__main__":
    main()
