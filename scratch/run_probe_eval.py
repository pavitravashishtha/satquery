"""
scratch/run_probe_eval.py — Run all 3 specialists against Cartosat/RISAT proxy dataset.
Evaluates domain collapse (>60% identical output).
"""

import os
import json
import torch
from PIL import Image
import sys

sys.path.insert(0, ".")
from scripts.model_registry import build_registry, GENERAL_VLM, GEOCHAT, FUSION
from eval.schemas.eval_io import EvalResults, ModelPrediction

PROBE_DIR = "data/cartosat_risat_probe"
with open(os.path.join(PROBE_DIR, "cartosat_risat_proxy_references.json"), "r") as f:
    references = json.load(f)

manager = build_registry(device="cuda")

# -------------------------------------------------------------
# 1. Evaluate FusionSpecialist
# -------------------------------------------------------------
print("\n=======================================================")
print("Evaluating FusionSpecialist on Cartosat/RISAT Proxy Pairs")
print("=======================================================")
fusion = manager.load(FUSION)
from scripts.orchestrator import _to_optical_tensor, _to_sar_tensor

fusion_preds = []
for sample_id in sorted(references.keys()):
    sample_dir = os.path.join(PROBE_DIR, sample_id)
    opt_path = os.path.join(sample_dir, "optical.png")
    sar_path = os.path.join(sample_dir, "sar.png")

    opt_tensor = _to_optical_tensor(opt_path)
    sar_tensor = _to_sar_tensor(sar_path)

    out = fusion.run(sar_tensor=sar_tensor, optical_tensor=opt_tensor)
    ans = out.get("answer", "")
    conf = out.get("confidence", 0.5)
    print(f"[{sample_id}] Fusion Output: {ans[:100]}... (conf={conf})")

    fusion_preds.append(
        ModelPrediction(
            sample_id=sample_id,
            task="vqa",
            model_name="fusion_dual_cnn",
            answer=ans,
        )
    )

manager.unload(FUSION)
torch.cuda.empty_cache()

fusion_eval_results = EvalResults(
    model_name="fusion_dual_cnn",
    benchmark="cartosat_risat_proxy",
    predictions=fusion_preds,
)
os.makedirs("eval/predictions", exist_ok=True)
with open("eval/predictions/fusion_dual_cnn_cartosat_risat_proxy_vqa.json", "w") as f:
    json.dump(fusion_eval_results.model_dump(), f, indent=2)

# -------------------------------------------------------------
# 2. Evaluate GeneralVLMSpecialist (Qwen2.5-VL-3B)
# -------------------------------------------------------------
print("\n=======================================================")
print("Evaluating GeneralVLMSpecialist on Cartosat/RISAT Proxy Pairs")
print("=======================================================")
vlm = manager.load(GENERAL_VLM)
vlm_preds = []
for sample_id in sorted(references.keys()):
    sample_dir = os.path.join(PROBE_DIR, sample_id)
    opt_path = os.path.join(sample_dir, "optical.png")
    question = references[sample_id]["question"]

    out = vlm.run(image=opt_path, query=question, task_type="vqa")
    ans = out.get("answer", "")
    conf = out.get("confidence", 0.9)
    print(f"[{sample_id}] Qwen VLM Output: \"{ans}\" (conf={conf})")

    vlm_preds.append(
        ModelPrediction(
            sample_id=sample_id,
            task="vqa",
            model_name="qwen2.5-vl-3b-general",
            answer=ans,
        )
    )

manager.unload(GENERAL_VLM)
torch.cuda.empty_cache()

vlm_eval_results = EvalResults(
    model_name="qwen2.5-vl-3b-general",
    benchmark="cartosat_risat_proxy",
    predictions=vlm_preds,
)
with open("eval/predictions/qwen2.5-vl-3b-general_cartosat_risat_proxy_vqa.json", "w") as f:
    json.dump(vlm_eval_results.model_dump(), f, indent=2)

# -------------------------------------------------------------
# 3. Evaluate GeoChatSpecialist (GeoChat-7B)
# -------------------------------------------------------------
print("\n=======================================================")
print("Evaluating GeoChatSpecialist on Cartosat/RISAT Proxy Pairs")
print("=======================================================")
geochat = manager.load(GEOCHAT)
geochat_preds = []
for sample_id in sorted(references.keys()):
    sample_dir = os.path.join(PROBE_DIR, sample_id)
    opt_path = os.path.join(sample_dir, "optical.png")
    question = references[sample_id]["question"]

    out = geochat.run(image=opt_path, query=question, task_type="vqa")
    ans = out.get("answer", "")
    conf = out.get("confidence", 0.5)
    print(f"[{sample_id}] GeoChat-7B Output: \"{ans}\" (conf={conf})")

    geochat_preds.append(
        ModelPrediction(
            sample_id=sample_id,
            task="vqa",
            model_name="geochat-7b",
            answer=ans,
        )
    )

manager.unload(GEOCHAT)
torch.cuda.empty_cache()

geochat_eval_results = EvalResults(
    model_name="geochat-7b",
    benchmark="cartosat_risat_proxy",
    predictions=geochat_preds,
)
with open("eval/predictions/geochat-7b_cartosat_risat_proxy_vqa.json", "w") as f:
    json.dump(geochat_eval_results.model_dump(), f, indent=2)

print("\nAll 3 specialist runs completed successfully.")
