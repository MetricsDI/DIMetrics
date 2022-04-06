#!/usr/bin/env python3

"""
Functions to assess performance using HED metric.
"""

import numpy as np
from collections import defaultdict
from .edit_distance import lc_subsequence

__ALL__ = [
    "cumulative_lcs",
    "line_item_edit_distance",
    "hed",
]


def cumulative_lcs(gt_item: dict, pred_item: dict):
    """Computes TP, FP, FN using longest common subsequence between two dictionaries of data

    Args:
        gt_item (dict): ground_truth dictionary
        pred_item (dict): predicted dictionary

    Returns:
        answer [tp, fp, fn] (numpy array): Numpy array with total number of True Postives, False Positives, False Negatives
        lcs_dict (dict): Dictionary containing fieldwise LCS metrics (TP, FP, FN)
    """
    total_keys = list(gt_item.keys())
    total_keys.extend(pred_item.keys())
    total_keys = set(total_keys)
    lcs_dict = defaultdict(lambda: np.zeros(3))
    answer = np.zeros(3)
    for key in total_keys:
        gt_str = gt_item.get(key, "")
        pred_str = pred_item.get(key, "")
        ans = lc_subsequence(gt_str, pred_str)
        lcs_dict[key] += ans
        answer += ans
    return answer, lcs_dict


def _add_dictionaries(a: dict, b: dict):
    """Util function to add 2 dictionaries containing fieldwise LCS metrics (TP, FP, FN)

    Args:
        a (dict): Dictionary 1
        b (dict): Dictionary 2

    Returns:
        c (dict): Addition of Dictionaries a and b
    """
    c = defaultdict(lambda: np.zeros(3))
    for k in a:
        c[k] += a[k]
    for k in b:
        c[k] += b[k]
    return c

def line_item_edit_distance(gt_list: list, pred_list: list):
    """Line Item Edit Distance (Dynamic Programming)

    Args:
        gt_list (List of dicts): GroundTruth List
        pred_list (List of dicts): Predicted List

    Returns:
        distances[m][n] (tuple): Returns a tuple containing the following 2 variables

        [tp, fp, fn] (numpy array): Numpy array with total number of True Postives, False Positives, False Negatives
        lcs_dict (dict): Dictionary containing fieldwise LCS metrics (TP, FP, FN)
    """

    m = len(gt_list)
    n = len(pred_list)

    distances = [[(np.zeros(3), defaultdict(lambda: np.zeros(3))) for i in range(n + 1)] for j in range(m + 1)]

    gt_lengths = np.zeros(m)
    gt_dicts = [None for i in range(m)]
    total_fn_dict = defaultdict(lambda: np.zeros(3))
    total_fn = 0
    for t1 in range(1, m+1):
        gt_lengths[t1-1] = sum([len(val) for key, val in gt_list[t1-1].items()])
        gt_dicts[t1-1] = {key:np.array([0, 0, len(val)]) for key, val in gt_list[t1-1].items()}
        total_fn += gt_lengths[t1-1]
        total_fn_dict = _add_dictionaries(total_fn_dict, gt_dicts[t1-1])
        distances[t1][0] = (np.array([0, 0, total_fn]), total_fn_dict)

    pred_lengths = np.zeros(n)
    pred_dicts = [None for i in range(n)]
    total_fp = 0
    total_fp_dict = defaultdict(lambda: np.zeros(3))
    for t2 in range(1, n+1):
        pred_lengths[t2-1] = sum([len(val) for key, val in pred_list[t2-1].items()])
        pred_dicts[t2-1] = {key:np.array([0, 0, len(val)]) for key, val in pred_list[t2-1].items()}
        total_fp += pred_lengths[t2-1]
        total_fp_dict = _add_dictionaries(total_fp_dict, pred_dicts[t2-1])
        distances[0][t2] = (np.array([0, total_fp, 0]), total_fp_dict)
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, m + 1):
        for t2 in range(1, n + 1):
            total, lcs_dict = cumulative_lcs(gt_list[t1-1], pred_list[t2-1]) 
            a = total + distances[t1 - 1][t2 - 1][0]
            adict = _add_dictionaries(lcs_dict, distances[t1 - 1][t2 - 1][1])

            b = np.array([0, 0, gt_lengths[t1-1]]) + distances[t1 - 1][t2][0]
            bdict = _add_dictionaries(gt_dicts[t1-1], distances[t1 - 1][t2][1])

            c = np.array([0, pred_lengths[t2-1], 0]) + distances[t1][t2 - 1][0]
            cdict = _add_dictionaries(pred_dicts[t2-1], distances[t1][t2 - 1][1])

            total = np.stack([a, b, c])
            total_dicts = [adict, bdict, cdict]
            LCSDist = total[:,1] + total[:,2]
            minimum_ind = np.argmin(LCSDist)
            distances[t1][t2] = (total[minimum_ind], total_dicts[minimum_ind])

    return distances[m][n]


def hed(gt_dict: dict, pred_dict: dict, list_name='Items'):
    """Hierarchical Edit Distance (Dynamic Programming)

    Args:
        gt_dict (dict): ground_truth/inference dict
        pred_dict (dict): predicted/reference dict
        list_name (str) [optional]: Name of key containing list of items
    
    Returns:
        answer [tp, fp, fn] (numpy array): Number of True Postives, False Positives, False Negatives
        answer_dict (dict): Dictionary containing fieldwise LCS metrics (TP, FP, FN)
    """
    gt_headers = {key: value for key, value in gt_dict.items() if key != list_name}
    pred_headers = {key: value for key, value in pred_dict.items() if key != list_name}
    header, header_dict = cumulative_lcs(gt_headers, pred_headers)
    line_edit, line_edit_dict = line_item_edit_distance(gt_dict[list_name], pred_dict[list_name])
    answer = header + line_edit
    answer_dict = _add_dictionaries(header_dict, line_edit_dict)
    return answer, answer_dict
