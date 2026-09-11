import os
import sys
import json
import torch

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from scripts.geochat_specialist import GeoChatSpecialist

def main():
    query = "What objects and features are visible in this satellite image?"
    img_18 = "data/SECOND/im2/00018.png"
    new_images = [
        "data/SECOND/im2/00029.png",
        "data/SECOND/im2/00031.png",
        "data/SECOND/im2/00034.png",
        "data/SECOND/im2/00038.png",
        "data/SECOND/im2/00040.png",
    ]

    print("=" * 80)
    print("STEP 1: REPRODUCE AND CHARACTERIZE REPETITION LOOP")
    print("=" * 80)

    specialist = GeoChatSpecialist()
    specialist.load()

    print("\n--- Part 1: Re-running 00018.png 3 times ---")
    runs_00018 = []
    for i in range(1, 4):
        print(f"\n[Run {i}/3 for {img_18}]")
        res = specialist.run(image=img_18, query=query, task_type="vqa")
        answer = res["answer"]
        conf = res["confidence"]
        
        # Tokenize answer to count tokens
        tokens = specialist.tokenizer.encode(answer, add_special_tokens=False)
        tok_len = len(tokens)
        
        # Count occurrences of 'a building' or 'building'
        bldg_count = answer.count("a building")
        
        # Find where repetition starts: search for consecutive 'a building'
        idx_first_rep = -1
        first_rep_tok = -1
        phrase = "a building"
        first_pos = answer.find(phrase)
        if first_pos != -1:
            second_pos = answer.find(phrase, first_pos + len(phrase))
            # Find the position where consecutive repetition takes over
            # Check for ", a building, a building"
            rep_start = answer.find("a building, a building")
            if rep_start != -1:
                idx_first_rep = rep_start
                # Token count up to rep_start
                prefix = answer[:rep_start]
                prefix_tokens = specialist.tokenizer.encode(prefix, add_special_tokens=False)
                first_rep_tok = len(prefix_tokens)
                
        print(f"Run {i} Result:")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Token length: {tok_len}")
        print(f"  'a building' count: {bldg_count}")
        print(f"  Consecutive repetition begins around character index {idx_first_rep} (token ~{first_rep_tok})")
        print(f"  Decoded output: {repr(answer)}")

        runs_00018.append({
            "run": i,
            "token_length": tok_len,
            "bldg_count": bldg_count,
            "rep_start_char": idx_first_rep,
            "rep_start_tok": first_rep_tok,
            "answer": answer,
            "confidence": conf
        })

    print("\n--- Part 2: Testing 5 Additional Unused Images ---")
    new_results = []
    for i, img_path in enumerate(new_images, 1):
        print(f"\n[New Image {i}/5: {img_path}]")
        res = specialist.run(image=img_path, query=query, task_type="vqa")
        answer = res["answer"]
        conf = res["confidence"]
        tokens = specialist.tokenizer.encode(answer, add_special_tokens=False)
        tok_len = len(tokens)
        
        # Check for repetition loops: n-gram repetition
        words = answer.split()
        ngrams = [" ".join(words[j:j+3]) for j in range(len(words)-2)]
        max_ngram_rep = max([ngrams.count(ng) for ng in set(ngrams)]) if ngrams else 0
        has_loop = max_ngram_rep >= 5

        print(f"  Output: {repr(answer)}")
        print(f"  Confidence: {conf:.4f}, Tokens: {tok_len}, Max 3-gram rep: {max_ngram_rep}, Has loop: {has_loop}")

        new_results.append({
            "image": img_path,
            "answer": answer,
            "confidence": conf,
            "tokens": tok_len,
            "max_ngram_rep": max_ngram_rep,
            "has_loop": has_loop
        })

    specialist.unload()

    with open("scratch/step1_characterization.json", "w") as f:
        json.dump({"runs_00018": runs_00018, "new_images": new_results}, f, indent=2)

if __name__ == "__main__":
    main()
