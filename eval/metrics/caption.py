import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure we have wordnet and punkt for basic tokenization if needed, though we can use split() for simplicity
def caption_score_bleu(predictions, references):
    """
    Computes BLEU score for captions.
    `references` is expected to be a list of lists of reference strings, 
    since BLEU supports multiple references per prediction.
    For simplicity, if references are just lists of strings, we'll wrap them.
    """
    if not predictions or not references:
        return 0.0
        
    smoothie = SmoothingFunction().method4
    total_score = 0.0
    
    for p, r_list in zip(predictions, references):
        p_ans = p.answer.strip().lower() if p.answer else ""
        # If there's only one reference, wrap it in a list
        if isinstance(r_list, str):
            refs = [r_list.strip().lower().split()]
        else:
            refs = [ref.strip().lower().split() for ref in r_list]
            
        cand = p_ans.split()
        score = sentence_bleu(refs, cand, smoothing_function=smoothie)
        total_score += score
        
    return total_score / max(len(references), 1)
