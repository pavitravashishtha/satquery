#!/usr/bin/env python3
"""
Comprehensive Confidence Calibration & Expected Calibration Error (ECE) Evaluation
for SatQuery AI Specialists (ChangeVQASpecialist and GeneralVLMSpecialist).

Computes:
- Uncalibrated ECE (10 bins)
- Reliability diagrams (bin accuracy vs. bin confidence)
- Post-hoc Temperature Scaling (token-level and sequence-level)
- Calibrated ECE and Brier Score
- AUROC (ranking / discriminative separation)
"""

import os
import sys
import json
import time
import math
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import torch

sys.path.insert(0, "/home/pavitra/satquery/scripts")
sys.path.insert(0, "/home/pavitra/satquery")

from cdvqa_dataset import load_cdvqa_split
from qwen_specialist import ChangeVQASpecialist, release_shared_qwen_vl
from general_vlm_specialist import GeneralVLMSpecialist


def compute_ece(confidences: List[float], correctness: List[bool], n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics.
    Bins are [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    """
    N = len(confidences)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    bins_data = []
    total_ece = 0.0
    max_ce = 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include upper bound in last bin
        if i == n_bins - 1:
            indices = [idx for idx, c in enumerate(confidences) if low <= c <= high]
        else:
            indices = [idx for idx, c in enumerate(confidences) if low <= c < high]

        count = len(indices)
        if count > 0:
            bin_acc = sum(correctness[idx] for idx in indices) / count
            bin_conf = sum(confidences[idx] for idx in indices) / count
            diff = abs(bin_acc - bin_conf)
            weight = count / N
            total_ece += weight * diff
            max_ce = max(max_ce, diff)
        else:
            bin_acc = None
            bin_conf = None
            diff = None

        bins_data.append({
            "bin_idx": i,
            "range": [round(float(low), 2), round(float(high), 2)],
            "count": count,
            "accuracy": round(float(bin_acc), 4) if bin_acc is not None else None,
            "avg_confidence": round(float(bin_conf), 4) if bin_conf is not None else None,
            "calibration_gap": round(float(diff), 4) if diff is not None else None,
        })

    brier = sum((c - (1.0 if corr else 0.0)) ** 2 for c, corr in zip(confidences, correctness)) / N

    return {
        "ece": round(float(total_ece), 4),
        "mce": round(float(max_ce), 4),
        "brier_score": round(float(brier), 4),
        "bins": bins_data,
        "sample_count": N,
        "overall_accuracy": round(float(sum(correctness) / N), 4),
        "overall_avg_confidence": round(float(sum(confidences) / N), 4),
    }


def compute_auroc(confidences: List[float], correctness: List[bool]) -> float:
    """Computes AUROC distinguishing correct (1) from incorrect (0) based on confidence."""
    y = np.array([1 if c else 0 for c in correctness])
    scores = np.array(confidences)
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Mann-Whitney U test formula
    ranks = np.argsort(np.argsort(scores)) + 1
    u = np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0
    auroc = u / (n_pos * n_neg)
    return round(float(auroc), 4)


def fit_temperature_scaling(
    raw_token_logits_list: List[List[torch.Tensor]], 
    generated_token_ids_list: List[List[int]], 
    correctness: List[bool]
) -> Tuple[float, List[float]]:
    """
    Fits optimal temperature T on token-level logits to minimize Negative Log Likelihood (NLL)
    of predicting the sequence correctness, and returns optimal T and scaled confidences.
    """
    y = np.array([1.0 if c else 0.0 for c in correctness])

    def get_confs_for_T(T_val: float) -> np.ndarray:
        confs = []
        for scores, gen_ids in zip(raw_token_logits_list, generated_token_ids_list):
            if not scores or not gen_ids:
                confs.append(0.5)
                continue
            probs = []
            for score, tok_id in zip(scores, gen_ids):
                # Scale logits by T
                scaled_score = score / T_val
                prob = torch.softmax(scaled_score, dim=-1)[tok_id].item()
                probs.append(prob)
            confs.append(sum(probs) / len(probs) if probs else 0.5)
        return np.array(confs)

    # Grid search over T in [0.5, 20.0] to minimize MSE / Brier score
    best_t = 1.0
    best_loss = 1e9

    for T_cand in np.linspace(1.0, 20.0, 191):
        c_cand = get_confs_for_T(T_cand)
        # Brier loss
        loss = np.mean((c_cand - y) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_t = float(T_cand)

    scaled_confs = get_confs_for_T(best_t).tolist()
    return round(best_t, 2), scaled_confs


def run_evaluation():
    val_samples = load_cdvqa_split("Val")
    N = 200
    samples = val_samples[:N]
    print(f"Loaded {len(samples)} CDVQA validation samples.")

    # 1. Evaluate ChangeVQASpecialist
    print("\n" + "=" * 70)
    print("RUNNING ChangeVQASpecialist on 200 CDVQA samples...")
    print("=" * 70)
    
    change_specialist = ChangeVQASpecialist()
    change_specialist.load()

    change_results = []
    change_raw_scores = []
    change_gen_ids = []

    t0 = time.time()
    for i, s in enumerate(samples):
        # Open images
        im1 = Image.open(s["im1_path"]).convert("RGB")
        im2 = Image.open(s["im2_path"]).convert("RGB")
        q = s["question"]
        gt = s["answer"].strip().lower()

        # Custom run to capture raw token logits
        im1_r = im1.resize((128, 128), Image.BILINEAR)
        im2_r = im2.resize((128, 128), Image.BILINEAR)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": im1_r},
                {"type": "image", "image": im2_r},
                {"type": "text", "text": q},
            ],
        }]
        inputs = change_specialist.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(change_specialist.model.device)

        with torch.no_grad():
            output = change_specialist.model.generate(
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_ids = output.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        pred_text = change_specialist.processor.decode(gen_ids, skip_special_tokens=True).strip().lower()

        # Uncalibrated confidence: mean token probability
        token_scores_cpu = [s[0].detach().cpu() for s in output.scores] if output.scores else []
        probs = []
        for idx_t, score in enumerate(token_scores_cpu):
            tok_id = gen_ids[idx_t] if idx_t < len(gen_ids) else None
            if tok_id is not None:
                prob = torch.softmax(score, dim=-1)[tok_id].item()
                probs.append(prob)
        conf = sum(probs) / len(probs) if probs else 0.0

        # Substring / relaxed match
        is_corr = (gt in pred_text) or (pred_text in gt) or (gt == pred_text)

        change_results.append({
            "idx": i,
            "question": q,
            "ground_truth": gt,
            "prediction": pred_text,
            "confidence": conf,
            "is_correct": is_corr,
        })
        change_raw_scores.append(token_scores_cpu)
        change_gen_ids.append(gen_ids)

        if (i + 1) % 50 == 0:
            print(f"  [ChangeVQA] Evaluated {i + 1}/{N} samples... elapsed {time.time() - t0:.1f}s")

    change_time = time.time() - t0
    print(f"ChangeVQASpecialist finished 200 samples in {change_time:.1f}s.")

    # 2. Evaluate GeneralVLMSpecialist
    print("\n" + "=" * 70)
    print("RUNNING GeneralVLMSpecialist on 200 CDVQA samples (single-image VQA on im2)...")
    print("=" * 70)
    
    general_specialist = GeneralVLMSpecialist()
    general_specialist.load()

    general_results = []
    general_raw_scores = []
    general_gen_ids = []

    t0 = time.time()
    for i, s in enumerate(samples):
        im2 = Image.open(s["im2_path"]).convert("RGB")
        q = s["question"]
        gt = s["answer"].strip().lower()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": im2},
                {"type": "text", "text": q},
            ],
        }]
        inputs = general_specialist.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(general_specialist.model.device)

        with torch.no_grad():
            with general_specialist.model.disable_adapter():
                output = general_specialist.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

        gen_ids = output.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        pred_text = general_specialist.processor.decode(gen_ids, skip_special_tokens=True).strip().lower()

        token_scores_cpu = [s[0].detach().cpu() for s in output.scores] if output.scores else []
        probs = []
        for idx_t, score in enumerate(token_scores_cpu):
            tok_id = gen_ids[idx_t] if idx_t < len(gen_ids) else None
            if tok_id is not None:
                prob = torch.softmax(score, dim=-1)[tok_id].item()
                probs.append(prob)
        conf = sum(probs) / len(probs) if probs else 0.0

        is_corr = (gt in pred_text) or (pred_text in gt) or (gt == pred_text)

        general_results.append({
            "idx": i,
            "question": q,
            "ground_truth": gt,
            "prediction": pred_text,
            "confidence": conf,
            "is_correct": is_corr,
        })
        general_raw_scores.append(token_scores_cpu)
        general_gen_ids.append(gen_ids)

        if (i + 1) % 50 == 0:
            print(f"  [GeneralVLM] Evaluated {i + 1}/{N} samples... elapsed {time.time() - t0:.1f}s")

    general_time = time.time() - t0
    print(f"GeneralVLMSpecialist finished 200 samples in {general_time:.1f}s.")

    # Unload GPU memory
    release_shared_qwen_vl()

    # 3. Compute ECE and Calibration for ChangeVQASpecialist
    print("\n" + "=" * 70)
    print("COMPUTING CALIBRATION METRICS FOR ChangeVQASpecialist...")
    print("=" * 70)
    change_confs_raw = [r["confidence"] for r in change_results]
    change_corr = [r["is_correct"] for r in change_results]
    change_ece_raw = compute_ece(change_confs_raw, change_corr)
    change_auroc_raw = compute_auroc(change_confs_raw, change_corr)

    # Fit temperature scaling
    opt_t_change, change_confs_scaled = fit_temperature_scaling(
        change_raw_scores, change_gen_ids, change_corr
    )
    change_ece_scaled = compute_ece(change_confs_scaled, change_corr)
    change_auroc_scaled = compute_auroc(change_confs_scaled, change_corr)

    print(f"ChangeVQASpecialist Accuracy: {change_ece_raw['overall_accuracy']*100:.1f}%")
    print(f"Uncalibrated Average Confidence: {change_ece_raw['overall_avg_confidence']*100:.2f}%")
    print(f"Uncalibrated ECE: {change_ece_raw['ece']:.4f} ({change_ece_raw['ece']*100:.2f}%)")
    print(f"Uncalibrated Brier Score: {change_ece_raw['brier_score']:.4f}")
    print(f"Uncalibrated AUROC: {change_auroc_raw:.4f}")
    print(f"Optimal Temperature (T): {opt_t_change}")
    print(f"Temperature-Scaled Average Confidence: {change_ece_scaled['overall_avg_confidence']*100:.2f}%")
    print(f"Temperature-Scaled ECE: {change_ece_scaled['ece']:.4f} ({change_ece_scaled['ece']*100:.2f}%)")
    print(f"Temperature-Scaled Brier Score: {change_ece_scaled['brier_score']:.4f}")
    print(f"Temperature-Scaled AUROC: {change_auroc_scaled:.4f}")

    # 4. Compute ECE and Calibration for GeneralVLMSpecialist
    print("\n" + "=" * 70)
    print("COMPUTING CALIBRATION METRICS FOR GeneralVLMSpecialist...")
    print("=" * 70)
    general_confs_raw = [r["confidence"] for r in general_results]
    general_corr = [r["is_correct"] for r in general_results]
    general_ece_raw = compute_ece(general_confs_raw, general_corr)
    general_auroc_raw = compute_auroc(general_confs_raw, general_corr)

    opt_t_general, general_confs_scaled = fit_temperature_scaling(
        general_raw_scores, general_gen_ids, general_corr
    )
    general_ece_scaled = compute_ece(general_confs_scaled, general_corr)
    general_auroc_scaled = compute_auroc(general_confs_scaled, general_corr)

    print(f"GeneralVLMSpecialist Accuracy: {general_ece_raw['overall_accuracy']*100:.1f}%")
    print(f"Uncalibrated Average Confidence: {general_ece_raw['overall_avg_confidence']*100:.2f}%")
    print(f"Uncalibrated ECE: {general_ece_raw['ece']:.4f} ({general_ece_raw['ece']*100:.2f}%)")
    print(f"Uncalibrated Brier Score: {general_ece_raw['brier_score']:.4f}")
    print(f"Uncalibrated AUROC: {general_auroc_raw:.4f}")
    print(f"Optimal Temperature (T): {opt_t_general}")
    print(f"Temperature-Scaled Average Confidence: {general_ece_scaled['overall_avg_confidence']*100:.2f}%")
    print(f"Temperature-Scaled ECE: {general_ece_scaled['ece']:.4f} ({general_ece_scaled['ece']*100:.2f}%)")
    print(f"Temperature-Scaled Brier Score: {general_ece_scaled['brier_score']:.4f}")
    print(f"Temperature-Scaled AUROC: {general_auroc_scaled:.4f}")

    # 5. Save structured report
    report_data = {
        "benchmark": "cdvqa",
        "sample_count": N,
        "change_vqa_specialist": {
            "uncalibrated": change_ece_raw,
            "auroc_uncalibrated": change_auroc_raw,
            "optimal_temperature": opt_t_change,
            "calibrated": change_ece_scaled,
            "auroc_calibrated": change_auroc_scaled,
        },
        "general_vlm_specialist": {
            "uncalibrated": general_ece_raw,
            "auroc_uncalibrated": general_auroc_raw,
            "optimal_temperature": opt_t_general,
            "calibrated": general_ece_scaled,
            "auroc_calibrated": general_auroc_scaled,
        }
    }

    os.makedirs("/home/pavitra/satquery/eval/reports", exist_ok=True)
    with open("/home/pavitra/satquery/eval/reports/confidence_calibration_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("\nSaved full calibration data to eval/reports/confidence_calibration_report.json")

    return report_data


if __name__ == "__main__":
    run_evaluation()
