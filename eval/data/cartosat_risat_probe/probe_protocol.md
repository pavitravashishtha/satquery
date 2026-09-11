# Cartosat/RISAT Probe Protocol

This document outlines the evaluation protocol for domain-gap mitigation on the hidden ISRO-SAC dataset (Cartosat-2S + RISAT).

## 1. Preparation
- Download 3-5 sample pairs (optical from Cartosat-2S, SAR from RISAT) over relevant Indian domains (e.g. flood site, urban site).
- Place them in `data/cartosat_risat_probe/`.

## 2. Preprocessing Standard
- ALL models must run their inputs through `shared/preprocess.py` -> `normalize_pair(optical, sar)` before inference.

## 3. Running the Models
- Pass the preprocessed samples to the model.
- Instruct the model to output predictions using the standard JSON schema defined in `eval/schemas/eval_io.py`.

## 4. Identifying Domain Collapse
- If a model outputs the exact same text/answer for >60% of the probe samples, flag it as **domain collapse**.
- Compare qualitative performance (visual sanity of outputs) on these samples versus standard Sentinel samples.

## 5. Reporting
- Fill out a `DOMAIN_GAP_REPORT.md` for each model track, detailing:
  - Probe accuracy vs. Benchmark accuracy
  - Side-by-side sample outputs (Sentinel vs. ISRO-SAC)
  - Clear qualitative statement on domain shift.
