def iou(box_a, box_b):
    """
    Computes Intersection over Union for two bounding boxes.
    Boxes should be in the format [x_min, y_min, x_max, y_max]
    """
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    
    inter = max(0, xb - xa) * max(0, yb - ya)
    
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    return inter / (a_area + b_area - inter + 1e-6)

def grounding_score(predictions, references, iou_threshold=0.5):
    """
    Computes the grounding score based on IoU threshold.
    """
    if not predictions or not references:
        return 0.0

    hits = 0
    for p, r in zip(predictions, references):
        if p.box is not None and r is not None:
            if iou(p.box, r) >= iou_threshold:
                hits += 1
                
    return hits / max(len(references), 1)
