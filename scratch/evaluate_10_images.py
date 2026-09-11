import os
import sys
import json
import torch

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from scripts.geochat_specialist import GeoChatSpecialist

image_paths = [
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

print("=" * 80)
print("RUNNING GEOCHAT ON 10 UNUSED IMAGES")
print("=" * 80)

specialist = GeoChatSpecialist()
specialist.load()

results = []

for i, img_path in enumerate(image_paths, 1):
    print(f"\n[{i}/10] Processing {img_path} ...")
    res = specialist.run(image=img_path, query=query, task_type="vqa")
    answer = res["answer"]
    confidence = res["confidence"]
    print(f"  Confidence: {confidence}")
    print(f"  Output: {repr(answer)}")
    results.append({
        "index": i,
        "image_path": img_path,
        "decoded_output": answer,
        "confidence": confidence,
    })

specialist.unload()

# Save JSON results
with open("scratch/results_10_images.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("SUMMARY TABLE OF DECODED OUTPUTS")
print("=" * 80)
print(f"{'#':<3} | {'Image Path':<26} | {'Decoded Output'}")
print("-" * 80)
for r in results:
    print(f"{r['index']:<3} | {r['image_path']:<26} | {r['decoded_output']}")
print("=" * 80)
