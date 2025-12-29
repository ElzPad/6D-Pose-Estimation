import torch
import torchvision
import numpy as np


def _parse_single_result(result):
    """
    Parses YOLO result and filters to keep ONLY the best (highest confidence) 
    bounding box for each distinct class ID found in the image.
    """
    if result.boxes.shape[0] == 0:
        return []

    # 1. Extract data to CPU numpy
    boxes = result.boxes.xywh.cpu().numpy()  # (N, 4)
    classes = result.boxes.cls.cpu().numpy() # (N,)
    confs = result.boxes.conf.cpu().numpy()  # (N,) # BEST w.r.t this value

    # 2. Filter: Keep only the best box per class
    unique_classes = np.unique(classes)
    keep_indices = []

    for cls in unique_classes:
        # Find all indices where the class matches
        cls_indices = np.where(classes == cls)[0]
        
        # Find the index (within the subset) that has the maximum confidence
        best_idx_subset = np.argmax(confs[cls_indices])
        
        # Map back to the global index
        best_idx_global = cls_indices[best_idx_subset]
        keep_indices.append(best_idx_global)

    # 3. Apply filter
    boxes = boxes[keep_indices]
    classes = classes[keep_indices]

    # 4. Stack and return [xc, yc, w, h, class_id]
    # We strictly respect the order expected by your pipeline
    detections = np.column_stack((boxes, classes))
    
    return detections.tolist()