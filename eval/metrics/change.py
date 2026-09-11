import re

def change_vqa_accuracy(predictions, references, relaxed=True):
    """
    Computes Exact Match or Relaxed Match accuracy for Change-VQA on paired samples.
    """
    if not predictions or not references:
        return 0.0

    correct = 0
    for p, r in zip(predictions, references):
        p_ans = p.answer.strip().lower() if p.answer else ""
        r_ans = r.strip().lower() if isinstance(r, str) else str(r).strip().lower()

        if relaxed:
            p_ans = re.sub(r'[^\w\s]', '', p_ans)
            r_ans = re.sub(r'[^\w\s]', '', r_ans)

        if p_ans == r_ans:
            correct += 1

    return correct / max(len(references), 1)
