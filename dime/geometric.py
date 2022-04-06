#!/usr/bin/env python3

"""
Metrics to assess performance on geometric evaluation.
"""
import numpy as np

__ALL__ = [
    "iou",
    "docbank_overlap"
]

def iou(y_true, y_pred):
    """Computes Intersection over Union given 2 lists of numpy arrays of bounding boxes
        Expects Bounding Boxes in the form: [xmin, ymin, xmax, ymax]

    Args:
        y_true (list or numpy array): Ground truth bounding box array
        y_pred (list or numpy array): Predicted bounding box array

    Raises:
        ValueError: Invalid IoU being calculated is greater than 1
        TypeError: Invalid bounding box array(s). Each bounding box array should have size 4

    Returns:
        iou (float): The calculated (intersection area / union area)
    """
    if len(y_true) != 4 or len(y_pred) != 4:
        raise TypeError("Invalid bounding box array(s). Each bounding box array should have size 4")
    # determine the (x, y)-coordinates of the intersection rectangle
    
    xA = max(y_true[0], y_pred[0])
    yA = max(y_true[1], y_pred[1])
    xB = min(y_true[2], y_pred[2])
    yB = min(y_true[3], y_pred[3])
    if xB <= xA or yB <= yA:  # IMPORTANT: Returns IoU > 1 without this check
        return 0


    # compute the area of intersection rectangle
    inter_area = (xB - xA) * (yB - yA)

    y_true_area = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])
    y_pred_area = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    iou = inter_area / float(y_true_area + y_pred_area - inter_area)

    if iou > 1:
        raise ValueError("Invalid IoU being calculated is greater than 1")
    return max(iou, 0)


def docbank_overlap(y_true, y_pred, return_values=False):    
    """Computes the docbank overlap (token wise overlap) between 2 numpy arrays of bounding boxes

    Args:
        y_true (numpy array)          : Ground Truth Bounding box array
        y_pred (numpy array)          : Prediction Bounding box array
        return_values (bool) [optional]: If true, returns details of precision, recall and f1-score

    Returns:
        total_overlap : Total overlap area of ground truth boxes and prediction boxes
        (or)
        Detailed dictionary (dict): Dictionary containing precision, recall and f1-score along with total areas and overlap area
    """
    total_overlap = 0
    for gt_box in y_true:
        total_overlap += calc_overlap_all(gt_box, y_pred)
    if not return_values:
        return total_overlap

    gt_area = get_total_area(y_true)
    pred_area = get_total_area(y_pred)
    precision = total_overlap / pred_area
    recall = total_overlap / gt_area
    f1_score = 2*precision*recall / (precision + recall)

    return {'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'gt_area': gt_area,
            'pred_area': pred_area,
            'overlap_area': total_overlap
            }




def get_total_area(bboxes):
    """Util function for calculating total area of a list of bounding boxes

    Args:
        bboxes (numpy array): List or Array of bounding boxes

    Returns:
        total_area : Total area of all bounding boxes
    """
    return np.sum((bboxes[:,3] - bboxes[:,1])* (bboxes[:,2] - bboxes[:,0]))


def calc_overlap_all(bbox1, bbox_all):
    """Util function for calculating maximum overlap of multiple boxes in bbox_all with bbox1

    Args:
        bbox1 (numpy array): Single bounding box array
        bbox_all (numpy array): Multiple bounding boxes array

    Returns:
        max_overlap : Maximum overlap of multiple boxes in bbox_all with bbox1
    """
    x_overlap = np.maximum(0, np.minimum(bbox_all[:,2], bbox1[2]) - np.maximum(bbox_all[:,0], bbox1[0]))
    y_overlap = np.maximum(0, np.minimum(bbox_all[:,3], bbox1[3]) - np.maximum(bbox_all[:,1], bbox1[1]))
    return np.max(x_overlap * y_overlap)

