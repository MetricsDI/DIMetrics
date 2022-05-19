#!/usr/bin/env python3

"""
Metrics to assess performance on textual evaluation.
"""

from unicodedata import normalize as normalize_str

import cython
import numpy as np
try:
    import torch
    TORCH = True
except ImportError:
    torch = None
    TORCH = False


__ALL__ = [
    "levenshtein_distance",
    "lc_subsequence",
    "lc_subsequence_torch",
    "str_exact_match",
]


# def levenshtein_distance(y_true: str, y_pred: str, normalize: bool = False):
# noinspection PyUnresolvedReferences
@cython.compile
@cython.locals(i1=cython.int, i2=cython.int, eps=cython.float)
def levenshtein_distance(y_true, y_pred, normalize=False):
    """
    Computes the Levenshtein Edit Distance between two strings.
    :param str y_true: ground_truth/inference string
    :param str y_pred: predicted/reference string
    :param bool normalize: Boolean for normalizing the distance by length of longer string
    :returns:
        distance (float): The computed Levenshtein Edit Distance between y_true and y_pred
                        (normalized distance between 0.0 and 1.0 will be returned if normalize=True)
    """
    # print("COMPILED" if cython.compiled else "INTERPRETED")

    s1 = y_true
    s2 = y_pred
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s2) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    if normalize:
        eps = 1e-10
        return 1 - distances[-1] / (len(s2) + eps)
    return distances[-1]


# def lc_subsequence(y_true: str, y_pred: str, ret_dp_table: bool = False):
# noinspection PyUnresolvedReferences
@cython.compile
@cython.locals(m=cython.int, n=cython.int, prev_i=cython.int, prev_j=cython.int, i=cython.int, j=cython.int,
               eps=cython.float)
def lc_subsequence(y_true, y_pred, get_dp_table=False):
    """
    Computes TP (True Positives), FP (False Positives), FN (False Negatives)
    using the Longest Common Subsequence algorithm.
   :param str y_true: ground_truth/inference string
   :param str y_pred: predicted/reference string
   :param bool get_dp_table: a flag to get data points array (default: no)
    :returns:
        [tp, fp, fn] (numpy array): Number of True Positives, False Positives, False Negatives
    """
    m = len(y_true)
    n = len(y_pred)
    dp_table = np.zeros((m + 1, n + 1))
    # iterate bottom up
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prev_i = i - 1
            prev_j = j - 1
            if y_true[prev_i] == y_pred[prev_j]:
                dp_table[prev_i, prev_j] = 1 + dp_table[prev_i - 1, prev_j - 1]
            else:
                dp_table[prev_i, prev_j] = max(dp_table[prev_i, prev_j - 1],
                                               dp_table[prev_i - 1, prev_j])

    tp = dp_table[m - 1, n - 1]
    fp = n - tp
    fn = m - tp
    if get_dp_table:
        # TODO: do we need to allow for the table to get returned???
        # TODO: why np.array and not a 3-tuple???
        return np.array([tp, fp, fn]), dp_table
    else:
        return np.array([tp, fp, fn])


def str_exact_match(y_true: str, y_pred: str, unicode_normalize: bool = False):
    """
    Computes exact match between two strings.
    :param str y_true: ground_truth/inference string
    :param str y_pred: predicted/reference string
    :param bool unicode_normalize: if True before comparison NFKD (a.k.a. NFD)
        (compatibility decomposition) Unicode normalization is applied.
    :return: True if matching; False if not matching
    """
    if unicode_normalize:
        return normalize_str('NFKD', y_true) == normalize_str('NFKD', y_pred)
    else:
        return y_true == y_pred
