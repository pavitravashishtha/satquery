import os
import sys
import json
import torch

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from scripts.geochat_specialist import GeoChatSpecialist

def check_consecutive_repetition(text: str, min_n=1, max_n=6, min_repeats=3):
    words = text.split()
    if len(words) < 6:
        return False, None, 0

    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n * 2):
            ngram = [w.strip(",.;:!?") for w in words[i:i+n]]
            if not any(ngram):
                continue
            repeats = 1
            j = i + n
            while j + n <= len(words):
                next_ngram = [w.strip(",.;:!?") for w in words[j:j+n]]
                if next_ngram == ngram:
                    repeats += 1
                    j += n
                else:
                    break
            if repeats >= min_repeats:
                return True, " ".join(words[i:i+n]), repeats
    return False, None, 0

def main():
    print("=" * 80)
    print("PART 1: 30-RUN STRESS TEST ON data/SECOND/im2/00018.png (repetition_penalty=1.15)")
    print("=" * 80)

    image_path = "data/SECOND/im2/00018.png"
    query = "What objects and features are visible in this satellite image?"

    specialist = GeoChatSpecialist()
    specialist.load()

    runs_data = []
    loop_count = 0

    for run_idx in range(1, 31):
        res = specialist.run(image=image_path, query=query, task_type="vqa")
        answer = res["answer"]
        conf = res["confidence"]
        is_deg = res.get("is_degenerate", False)

        tokens = specialist.tokenizer.encode(answer, add_special_tokens=False)
        token_count = len(tokens)

        has_loop, phrase, rep_count = check_consecutive_repetition(answer)
        if has_loop or is_deg:
            loop_count += 1

        print(f"[{run_idx:02d}/30] tok={token_count:3d} | conf={conf:.4f} | loop={has_loop} (rep={rep_count}, phrase={repr(phrase)})")
        print(f"       Text: {repr(answer)}")

        runs_data.append({
            "run": run_idx,
            "token_count": token_count,
            "has_loop": has_loop,
            "is_degenerate": is_deg,
            "repeating_phrase": phrase,
            "repeat_count": rep_count,
            "confidence": conf,
            "decoded_output": answer
        })

    specialist.unload()

    with open("scratch/results_stress_test_30.json", "w") as f:
        json.dump(runs_data, f, indent=2)

    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)
    print(f"Total Runs: 30")
    print(f"Total Loops: {loop_count}/30 ({loop_count / 30 * 100:.1f}%)")
    token_counts = [r["token_count"] for r in runs_data]
    print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}, mean={sum(token_counts)/len(token_counts):.1f}")
    if loop_count > 0:
        print("\nRuns that looped:")
        for r in runs_data:
            if r["has_loop"] or r["is_degenerate"]:
                print(f"  Run {r['run']} ({r['token_count']} tok, phrase={repr(r['repeating_phrase'])}):")
                print(f"    {repr(r['decoded_output'])}")
    else:
        print("\nALL 30 RUNS PASSED WITHOUT REPETITION LOOPS (0/30 loops).")

if __name__ == "__main__":
    main()
