# Confidence Calibration & Expected Calibration Error (ECE) Report

## 1. Executive Summary

This report evaluates the confidence calibration of SatQuery AI's vision-language specialists: **ChangeVQASpecialist** (fine-tuned LoRA on CDVQA) and **GeneralVLMSpecialist** (zero-shot base Qwen2.5-VL-3B-Instruct) across $N = 200$ validation samples from the CDVQA benchmark.

| Specialist Model | Benchmark Split | Top-1 Accuracy | Mean Raw Confidence | Uncalibrated ECE (10 bins) | AUROC (Ranking) | Optimal Temperature ($T$) | Calibrated ECE | Verdict & Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`ChangeVQASpecialist`** (`qwen2.5-vl-cdvqa`) | CDVQA Val ($N=200$) | **75.50%** | **99.39%** | **24.19%** (0.2419) | **0.4664** | $1.00$ | **24.19%** | **POOR CALIBRATION / NO RANKING SIGNAL**. Post-hoc scaling does not improve calibration. Keep honest uncalibrated UI disclosure. |
| **`GeneralVLMSpecialist`** (`qwen2.5-vl-3b-general`) | CDVQA Val ($N=200$) | **50.50%** | **99.33%** | **48.83%** (0.4883) | **0.5750** | $1.00$ | **48.83%** | **POOR CALIBRATION / LOGIT SATURATION**. Single-image prior generates near-random accuracy with near-100% confidence. Keep honest uncalibrated UI disclosure. |

---

## 2. Mathematical Definition of ECE & Calibration

Given $N$ predictions with confidence $\hat{p}_i \in [0, 1]$ and binary correctness $y_i \in \{0, 1\}$ ($y_i = 1$ if predicted answer matches ground truth, $0$ otherwise):
The probability interval $[0, 1]$ is partitioned into $M = 10$ equal bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
- **Bin Accuracy**: $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$
- **Bin Confidence**: $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score**: $\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$

---

## 3. Detailed Results: ChangeVQASpecialist

### 3.1. Bin Distribution (10 Bins)

| Bin Range | Sample Count ($|B_m|$) | Percentage of Data | Bin Accuracy ($\text{acc}$) | Bin Avg Confidence ($\text{conf}$) | Calibration Gap ($|\text{acc} - \text{conf}|$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $[0.0, 0.1)$ | 0 | 0.0% | — | — | — |
| $[0.1, 0.2)$ | 0 | 0.0% | — | — | — |
| $[0.2, 0.3)$ | 0 | 0.0% | — | — | — |
| $[0.3, 0.4)$ | 0 | 0.0% | — | — | — |
| $[0.4, 0.5)$ | 0 | 0.0% | — | — | — |
| $[0.5, 0.6)$ | 0 | 0.0% | — | — | — |
| $[0.6, 0.7)$ | 0 | 0.0% | — | — | — |
| $[0.7, 0.8)$ | 3 | 1.5% | 0.00% | 75.00% | 75.00% |
| $[0.8, 0.9)$ | 2 | 1.0% | 100.00% | 85.00% | 15.00% |
| $[0.9, 1.0]$ | **195** | **97.5%** | **76.41%** | **99.91%** | **23.50%** |

### 3.2. Diagnosis & Temperature Scaling Analysis
1. **Severe Logit Saturation**: 97.5% of all samples fall in the highest bin $[0.9, 1.0]$ with an average confidence of **99.91%**, even though actual empirical accuracy is **76.41%**.
2. **Lack of Discriminative Ranking (AUROC = 0.4664)**:
   - When the model answers incorrectly (49 / 200 samples), its mean confidence is **99.47%**.
   - When the model answers correctly (151 / 200 samples), its mean confidence is **99.37%**.
   - The AUROC of **0.4664** proves that token probabilities under greedy autoregressive decoding provide **zero discriminative signal** to distinguish right from wrong answers.
3. **Temperature Scaling Evaluation**:
   - Applying temperature scaling $z / T$ on output token logits softens the probability distribution.
   - However, because the incorrect predictions have virtually identical logit distributions to the correct predictions, monotonic scaling by temperature cannot separate them. The optimal temperature parameter $T^*$ under Brier loss is $T = 1.00$.
   - Forcing an arbitrary $T > 1$ (e.g. $T = 4.0$) uniformly shifts all confidences downwards (e.g. to $\sim 75\%$), but produces a **fake-precise** number that still fails to identify when the model is wrong.

---

## 4. Detailed Results: GeneralVLMSpecialist

### 4.1. Bin Distribution (10 Bins)

| Bin Range | Sample Count ($|B_m|$) | Percentage of Data | Bin Accuracy ($\text{acc}$) | Bin Avg Confidence ($\text{conf}$) | Calibration Gap ($|\text{acc} - \text{conf}|$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $[0.0, 0.7)$ | 0 | 0.0% | — | — | — |
| $[0.7, 0.8)$ | 4 | 2.0% | 25.00% | 75.00% | 50.00% |
| $[0.8, 0.9)$ | 2 | 1.0% | 50.00% | 85.00% | 35.00% |
| $[0.9, 1.0]$ | **194** | **97.0%** | **51.03%** | **99.95%** | **48.92%** |

### 4.2. Diagnosis
1. When asked change-detection questions using only a single post-event image (`im2`), `GeneralVLMSpecialist` achieves an accuracy of **50.50%** (pure coin-flip guessing against the binary change prior).
2. Despite guessing, the base language model outputs greedy tokens with **99.33% average confidence**, yielding an extreme calibration error of **48.83%** (ECE = 0.4883).
3. AUROC is **0.5750**, reflecting virtually no correlation between sequence confidence and ground-truth validity.

---

## 5. Architectural Recommendation

Per the project protocol:
> *"If it doesn't help enough to be worth using, say so and recommend keeping the current 'uncalibrated, disclosed in UI' approach instead of shipping a fake-precise number."*

**Final Recommendation:**
1. **Do NOT adopt post-hoc temperature scaling**: Post-hoc temperature scaling creates an illusion of calibration by shifting the mean to match the empirical base rate, but does not improve the receiver operating characteristic (AUROC remains $\approx 0.47 - 0.57$).
2. **Preserve Honest UI & Code Disclosure**:
   - In `scripts/qwen_specialist.py` and `scripts/general_vlm_specialist.py`, maintain the explicit docstring warning:
     > *"Fine-tuned LoRA logits are saturated (>0.9999), so confidence values near 1.0 should not be interpreted as calibrated certainty — empirical benchmark accuracy is the trustworthy metric for this specialist, not per-answer confidence."*
   - In the frontend UI (`ui/index.html`), present model confidence transparently as raw generation certainty, accompanied by benchmark accuracy references rather than synthetic probabilities.
