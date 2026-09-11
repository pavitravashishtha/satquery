import os
import json
import glob

def generate_markdown_report(report_files, output_file):
    if not report_files:
        print("No report JSONs found to generate markdown.")
        return

    # Group by model
    model_reports = {}
    for f_path in report_files:
        with open(f_path, 'r') as f:
            data = json.load(f)
            model = data["model_name"]
            if model not in model_reports:
                model_reports[model] = []
            model_reports[model].append(data)

    with open(output_file, 'w') as out:
        out.write("# Evaluation Scorecards\n\n")
        
        for model, results in model_reports.items():
            out.write(f"## Model Scorecard: {model}\n\n")
            out.write("| Benchmark | Task | Metric | Score | MajBase | Gap | n |\n")
            out.write("|-----------|------|--------|-------|---------|-----|---|\n")
            for r in results:
                maj_base_str = f"{r['majority_baseline']:.4f}" if "majority_baseline" in r else "N/A"
                gap_str = f"{r['baseline_gap']:+.4f}" if "baseline_gap" in r else "N/A"
                out.write(f"| {r['benchmark']} | {r['task']} | {r['metric']} | {r['score']:.4f} | {maj_base_str} | {gap_str} | {r['n_samples']} |\n")
            out.write("\n")

if __name__ == "__main__":
    reports_dir = os.path.join("eval", "reports")
    if os.path.exists(reports_dir):
        json_files = glob.glob(os.path.join(reports_dir, "*.json"))
        out_md = os.path.join(reports_dir, "scorecard.md")
        generate_markdown_report(json_files, out_md)
        print(f"Scorecard generated at {out_md}")
    else:
        print(f"Directory {reports_dir} does not exist. Run evaluations first.")
