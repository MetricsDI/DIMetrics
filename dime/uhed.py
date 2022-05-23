from collections import defaultdict

import numpy as np

from dime.hed import cumulative_lcs, _add_dictionaries
from scipy.optimize import linear_sum_assignment

# noinspection PyUnresolvedReferences
@cython.compile
@cython.locals(gt_dict=cython.list, pred_dict=cython.list)
def get_hm_matched_pairs(gt_dict, pred_dict, list_name='Items'):
    """
    Hungarian Matching by maximizing TP of LCS for line items
    Args:
        gt_dict (dict): ground_truth/inference dict
        pred_dict (dict): predicted/reference dict
        list_name (str) [optional]: Name of key containing list of items
    Returns:
        pairs (list): [ [{gt_lineitem}, {pred_lineitem}], ... ]
        gt_headers (dict) : Dictionary of GT containing not lineitems
        pred_headers (dict) : Dictionary of Pred containing not lineitems
    """
    gt_lineitems = gt_dict[list_name]
    pred_lineitems = pred_dict[list_name]
    lcs_tp_mat = np.zeros((len(gt_lineitems), len(pred_lineitems)))
    for gt_idx in range(len(gt_lineitems)):
        for pred_idx in range(len(pred_lineitems)):
            ans, _ = cumulative_lcs(gt_lineitems[gt_idx], pred_lineitems[pred_idx])
            lcs_tp_mat[gt_idx][pred_idx] = ans[0]

    row_ind, col_ind = linear_sum_assignment(-lcs_tp_mat)
    pairs = [[gt_lineitems[gt_ind], pred_lineitems[pred_ind]] for gt_ind, pred_ind in zip(row_ind, col_ind)]
    gt_headers = {key: value for key, value in gt_dict.items() if key != list_name}
    pred_headers = {key: value for key, value in pred_dict.items() if key != list_name}
    return pairs, gt_headers, pred_headers

# noinspection PyUnresolvedReferences
@cython.compile
def uhed(gt_dict, pred_dict, list_name='Items'):
    """Unordered Hierarchical Edit Distance for single file (Dynamic Programming)
    Args:
        gt_dict (dict): ground_truth/inference dict
        pred_dict (dict): predicted/reference dict
        list_name (str) [optional]: Name of key containing list of items
    Returns:
        answer [tp, fp, fn] (numpy array): Number of True Positives, False Positives, False Negatives
        answer_dict (dict): Dictionary containing field-wise LCS metrics (TP, FP, FN)
    """
    lineitem_pairs, gt_headers, pred_headers = get_hm_matched_pairs(gt_dict, pred_dict, list_name)
    header, header_dict = cumulative_lcs(gt_headers, pred_headers)
    # [tp, fp, fn]
    line = np.zeros(3)
    line_edit_dict = defaultdict(lambda: np.zeros(3))
    for gt_dict, pred_dict in lineitem_pairs:
        lcs_arr, lcs_dict = cumulative_lcs(gt_dict, pred_dict)
        line += lcs_arr
        line_edit_dict = _add_dictionaries(line_edit_dict, lcs_dict)
    answer = header + line
    answer_dict = _add_dictionaries(header_dict, line_edit_dict)
    return answer, answer_dict
