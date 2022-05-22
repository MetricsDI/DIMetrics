#!/usr/bin/env python3

"""
Functions to assess performance using HED metric.
"""

from collections import defaultdict

import cython
import numpy as np

from .textual import lc_subsequence

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
        answer [tp, fp, fn] (numpy array): Numpy array with total number of True Positives, False Positives,
        False Negatives
        lcs_dict (dict): Dictionary containing field-wise LCS metrics (TP, FP, FN)
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
    """Util function to add 2 dictionaries containing field-wise LCS metrics (TP, FP, FN)

    Args:
        a (dict): Dictionary 1
        b (dict): Dictionary 2

    Returns:
        c (dict): Addition of Dictionaries a and b
    """
    c = defaultdict(lambda: np.zeros(3))
    for k in a:
        c[k] += a[k]
    # for k in set(b.keys()).difference(a.keys()):
    for k in b:  # FIXME: this will add 2nd time for the keys from a duplicated in b (suggestion above)
        c[k] += b[k]
    return c


# def line_item_edit_distance(gt_list: list, pred_list: list):
# noinspection PyUnresolvedReferences
@cython.compile
@cython.locals(gt_list=cython.list, pred_list=cython.list, m=cython.int, n=cython.int, i=cython.int,
               t1=cython.int, t2=cython.int, k=cython.int)
def line_item_edit_distance(gt_list, pred_list):
    """Line Item Edit Distance (Dynamic Programming)

    Args:
        gt_list (List of dicts): Ground Truth List
        pred_list (List of dicts): Predicted List

    Returns:
        distances[m][n] (tuple): Returns a tuple containing the following 2 variables

        [tp, fp, fn] (numpy array): Numpy array with total number of True Positives, False Positives, False Negatives
        lcs_dict (dict): Dictionary containing field-wise LCS metrics (TP, FP, FN)
    """

    m = len(gt_list)
    n = len(pred_list)

    distances = [[(np.zeros(3), defaultdict(lambda: np.zeros(3))) for _ in range(n + 1)] for _ in range(m + 1)]

    def _collect(elem_list, update_fun):
        k = len(elem_list)
        lengths = np.zeros(k)
        dicts = []
        total_dict = defaultdict(lambda: np.zeros(3))
        total_cnt = 0
        for i in range(k):
            i_len = sum([len(val) for val in gt_list[i].values()])
            lengths[i] = i_len
            total_cnt += i_len
            
            i_dict = {key: np.array([0, 0, len(val)]) for key, val in gt_list[i].items()}
            dicts.append(i_dict)
            total_dict = _add_dictionaries(total_dict, i_dict)
            
            update_fun(i, total_cnt, total_dict)
            
        return lengths, dicts

    def gt_update_fun(i, total_cnt, total_dict):  # FN
        distances[i + 1][0] = (np.array([0, 0, total_cnt]), total_dict)

    gt_lengths, gt_dicts = _collect(gt_list, gt_update_fun)

    def pred_update_fun(i, total_cnt, total_dict):  # FP
        distances[0][i + 1] = (np.array([0, total_cnt, 0]), total_dict)

    pred_lengths, pred_dicts = _collect(pred_list, pred_update_fun)
    
    for t1 in range(0, m):
        for t2 in range(0, n):
            total, lcs_dict = cumulative_lcs(gt_list[t1], pred_list[t2])

            a = total + distances[t1][t2][0]
            adict = _add_dictionaries(lcs_dict, distances[t1][t2][1])

            b = np.array([0, 0, gt_lengths[t1]]) + distances[t1][t2 + 1][0]
            bdict = _add_dictionaries(gt_dicts[t1], distances[t1][t2 + 1][1])

            c = np.array([0, pred_lengths[t2], 0]) + distances[t1 + 1][t2][0]
            cdict = _add_dictionaries(pred_dicts[t2], distances[t1 + 1][t2][1])

            total = np.stack([a, b, c])
            total_dicts = [adict, bdict, cdict]
            lcs_dist = total[:, 1] + total[:, 2]
            minimum_ind = np.argmin(lcs_dist)
            distances[t1][t2] = (total[minimum_ind], total_dicts[minimum_ind])

    return distances[m][n]


def hed(gt_dict: dict, pred_dict: dict, list_name='Items'):
    """Hierarchical Edit Distance (Dynamic Programming)

    Args:
        gt_dict (dict): ground_truth/inference dict
        pred_dict (dict): predicted/reference dict
        list_name (str) [optional]: Name of key containing list of items
    
    Returns:
        answer [tp, fp, fn] (numpy array): Number of True Positives, False Positives, False Negatives
        answer_dict (dict): Dictionary containing field-wise LCS metrics (TP, FP, FN)
    """
    gt_headers = {key: value for key, value in gt_dict.items() if key != list_name}
    pred_headers = {key: value for key, value in pred_dict.items() if key != list_name}
    header, header_dict = cumulative_lcs(gt_headers, pred_headers)
    line_edit, line_edit_dict = line_item_edit_distance(gt_dict[list_name], pred_dict[list_name])
    answer = header + line_edit
    answer_dict = _add_dictionaries(header_dict, line_edit_dict)
    return answer, answer_dict
