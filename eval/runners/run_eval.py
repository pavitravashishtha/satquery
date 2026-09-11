import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.schemas.eval_io import EvalResults, ModelPrediction
from eval.metrics.vqa import vqa_accuracy
from eval.metrics.caption import caption_score_bleu
from eval.metrics.grounding import grounding_score
from eval.metrics.change import change_vqa_accuracy

def load_references(benchmark):
    """
    Loads ground truth references for supported benchmarks.
    """
    bench = benchmark.lower().strip()
    if bench == "cdvqa":
        answers_path = "/home/pavitra/satquery/data/CDVQA/Val_answers.json"
        if not os.path.exists(answers_path):
            raise FileNotFoundError(f"CDVQA Val answers file not found at {answers_path}")
        with open(answers_path, "r", encoding="utf-8") as f:
            raw_answers = json.load(f)["answers"]
        # Map str(question_id) -> ground truth answer string
        ref_dict = {str(a["question_id"]): a["answer"] for a in raw_answers}
        print(f"Loaded {len(ref_dict)} reference answers for benchmark '{benchmark}' from {answers_path}")
        return ref_dict
    elif bench in ("cartosat_risat_proxy", "probe"):
        # Proxy benchmark standing in for ISRO Cartosat-2S / RISAT-1 scenes
        ref_path = "/home/pavitra/satquery/data/cartosat_risat_probe/cartosat_risat_proxy_references.json"
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Cartosat/RISAT proxy references file not found at {ref_path}")
        with open(ref_path, "r", encoding="utf-8") as f:
            raw_refs = json.load(f)
        ref_dict = {str(k): v.get("answer", "") if isinstance(v, dict) else str(v) for k, v in raw_refs.items()}
        print(f"Loaded {len(ref_dict)} reference answers for proxy benchmark 'cartosat_risat_proxy' from {ref_path}")
        return ref_dict
    elif bench in ("ben_ge_8k", "ben-ge-8k"):
        # Real multi-modal Optical-SAR land cover ground truth from BigEarthNet ben-ge-8k
        from scripts.fusion_dataset import BenGeDataset, map_raw_labels_to_19
        data_dir = "/home/pavitra/satquery/data/ben-ge-8k"
        meta_csv = os.path.join(data_dir, "ben-ge-8k_meta.csv")
        if not os.path.exists(meta_csv):
            raise FileNotFoundError(f"ben-ge-8k metadata not found at {meta_csv}")
        
        ref_dict = {}
        ds = BenGeDataset(data_dir=data_dir)
        for item in ds.samples:
            pid = item["patch_id"]
            with open(item["meta_path"]) as f:
                raw_labels = json.load(f).get("labels", [])
            mapped = map_raw_labels_to_19(raw_labels)
            ref_dict[str(pid)] = ", ".join(sorted(mapped)) if mapped else "no significant land-cover features detected"
        print(f"Loaded {len(ref_dict)} reference ground truths for benchmark '{benchmark}' from {data_dir}")
        return ref_dict
    else:
        raise NotImplementedError(
            f"Benchmark '{benchmark}' reference loader is not implemented yet. "
            f"Currently supported benchmarks with verified ground truth: ['cdvqa', 'cartosat_risat_proxy', 'ben_ge_8k']."
        )


def run_evaluation(predictions_file, benchmark, task_type):
    with open(predictions_file, 'r') as f:
        data = json.load(f)
        
    try:
        # Validate input schema
        eval_results = EvalResults(**data)
    except Exception as e:
        print(f"Error validating input JSON against schema: {e}")
        sys.exit(1)
        
    references = load_references(benchmark)
    
    # Match predictions against ground truth references; skip unmatched samples
    matched_predictions = []
    matched_refs = []
    unmatched_samples = []

    for p in eval_results.predictions:
        sid = str(p.sample_id)
        if sid in references:
            matched_predictions.append(p)
            matched_refs.append(references[sid])
        else:
            unmatched_samples.append(sid)

    if unmatched_samples:
        print(f"Warning: {len(unmatched_samples)} sample(s) had no ground truth in '{benchmark}' and were skipped.")

    if not matched_predictions:
        print(f"Error: 0 out of {len(eval_results.predictions)} samples matched references in '{benchmark}'! Aborting.")
        sys.exit(1)
    
    score = 0.0
    metric_name = ""
    extra_metrics = {}
    
    if task_type in ["vqa", "change_vqa"]:
        from collections import Counter
        if task_type == "vqa":
            score = vqa_accuracy(matched_predictions, matched_refs)
        else:
            score = change_vqa_accuracy(matched_predictions, matched_refs)
        metric_name = "accuracy (relaxed)"

        ref_counts = Counter([str(r).strip().lower() for r in matched_refs])
        majority_class, majority_count = ref_counts.most_common(1)[0]
        maj_baseline = majority_count / len(matched_refs)
        extra_metrics = {
            "majority_baseline": round(maj_baseline, 4),
            "majority_class": majority_class,
            "baseline_gap": round(score - maj_baseline, 4),
        }
    elif task_type == "caption":
        score = caption_score_bleu(matched_predictions, matched_refs)
        metric_name = "BLEU"
    elif task_type == "grounding":
        score = grounding_score(matched_predictions, matched_refs)
        metric_name = "IoU >= 0.5"
    else:
        print(f"Unknown task type: {task_type}")
        sys.exit(1)
        
    # Domain collapse detection: flag if a model outputs the exact same answer for > 60% of samples
    pred_texts = [str(getattr(p, "answer", None) or getattr(p, "prediction", "")).strip().lower() for p in matched_predictions]
    if pred_texts:
        from collections import Counter
        top_pred, top_count = Counter(pred_texts).most_common(1)[0]
        collapse_ratio = top_count / len(pred_texts)
        domain_collapse = collapse_ratio > 0.60
        extra_metrics["domain_collapse_detected"] = domain_collapse
        extra_metrics["dominant_prediction"] = top_pred
        extra_metrics["dominant_prediction_ratio"] = round(collapse_ratio, 4)
        if domain_collapse:
            print(f"[WARNING: DOMAIN COLLAPSE DETECTED] Model predicted identical output for {collapse_ratio * 100:.1f}% of samples: '{top_pred}'")

    report = {
        "model_name": eval_results.model_name,
        "benchmark": benchmark,
        "task": task_type,
        "metric": metric_name,
        "score": score,
        "n_samples": len(matched_predictions),
        **extra_metrics,
    }
    
    os.makedirs(os.path.join("eval", "reports"), exist_ok=True)
    report_file = os.path.join("eval", "reports", f"{eval_results.model_name}_{benchmark}_{task_type}_report.json")
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Report saved to {report_file}")
    print(f"Score ({metric_name}): {score:.4f}")
    if "majority_baseline" in extra_metrics:
        print(f"Majority Baseline:          {extra_metrics['majority_baseline']:.4f} (class '{extra_metrics['majority_class']}')")
        print(f"Gap vs Majority Baseline:   {extra_metrics['baseline_gap']:+.4f} ({extra_metrics['baseline_gap'] * 100:+.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", required=True, help="Path to JSON file conforming to EvalResults schema")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g. cdvqa, vrsbench)")
    parser.add_argument("--task", required=True, help="Task type (e.g. change_vqa, vqa)")
    args = parser.parse_args()
    
    run_evaluation(args.model_outputs, args.benchmark, args.task)
