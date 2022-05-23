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
    TORCH = False


__ALL__ = [
    "levenshtein_distance",
    "levenshtein_distance_cython",
    "levenshtein_distance_cython_numpy",
    "lc_subsequence",
    "lc_subsequence_torch",
    "str_exact_match",
]


def levenshtein_distance(y_true: str, y_pred: str, normalize: bool = False):
    """
    Computes the Levenshtein Edit Distance between two strings.
    :param str y_true: ground_truth/inference string
    :param str y_pred: predicted/reference string
    :param bool normalize: Boolean for normalizing the distance by length of longer string
    :returns:
        distance (float): The computed Levenshtein Edit Distance between y_true and y_pred
                        (normalized distance between 0.0 and 1.0 will be returned if normalize=True)
    """
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


def lc_subsequence(y_true: str, y_pred: str, ret_dp_table=False):
    """
    Computes TP, FP, FN using longest common subsequence between two strings.
   :param str y_true: ground_truth/inference string
   :param str y_pred: predicted/reference string
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
    if ret_dp_table:
        return np.array([tp, fp, fn]), dp_table
    else:
        return np.array([tp, fp, fn])


def lc_subsequence_torch(y_true: str, y_pred: str, ret_dp_table: bool=False, device: str='cpu'):
    """
    Computes TP, FP, FN using longest common subsequence between two strings using PyTorch
   :param str y_true: ground_truth/inference string
   :param str y_pred: predicted/reference string
    :returns:
        [tp, fp, fn] (numpy array): Number of True Positives, False Positives, False Negatives
    """
    if TORCH:
        m = len(y_true)
        n = len(y_pred)
        dp_table = torch.zeros((m + 1, n + 1),device=device)
        # iterate bottom up
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prev_i = i - 1
                prev_j = j - 1
                if y_true[prev_i] == y_pred[prev_j]:
                    dp_table[prev_i, prev_j] = 1 + dp_table[prev_i - 1, prev_j - 1]
                else:
                    dp_table[prev_i, prev_j] = torch.max(dp_table[prev_i, prev_j - 1],
                                                   dp_table[prev_i - 1, prev_j])
        if device !='cpu':
            dp_table = dp_table.detach().cpu()

        tp = dp_table[m - 1, n - 1]
        tp = float(tp.numpy())
        fp = n - tp
        fn = m - tp
        if ret_dp_table:
            return (tp, fp, fn), dp_table
        else:
            return (tp, fp, fn)
    else:
        return lc_subsequence(y_true, y_pred, ret_dp_table)


# def str_exact_match(y_true: str, y_pred: str, unicode_normalize: bool = False):
# noinspection PyUnresolvedReferences
@cython.compile
def str_exact_match(y_true, y_pred, unicode_normalize=False):
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


# noinspection PyUnresolvedReferences
@cython.compile
@cython.locals(i=cython.int, j=cython.int, eps=cython.float)
def levenshtein_distance_cython(str1, str2):
    edits = [[x for x in range(len(str1) + 1)] for y in range(len(str2) + 1)]
    for i in range(1, len(str2) + 1):
        edits[i][0] = edits[i - 1][0] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str2[i - 1] == str1[j - 1]:
                edits[i][j] = edits[i - 1][j - 1]
            else:
                edits[i][j] = 1 + min(edits[i - 1][j - 1], edits[i][j - 1], edits[i - 1][j])
        return edits[-1][-1]


# noinspection PyUnresolvedReferences
@cython.compile
@cython.locals(i=cython.int, j=cython.int, len_str1=cython.int,  len_str2=cython.int, eps=cython.float)
def levenshtein_distance_cython_numpy(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # suggestion from: https://stackoverflow.com/a/46231169/6573902
    edits = np.empty((len_str2, len_str1), dtype=np.int32)
    line = np.arange(0, len_str1)
    edits[:] = line[None, :]
    first_col = np.arange(0, len_str2)
    edits[:, 0] = first_col
    for i in range(1, len_str2):
        i_less = i - 1
        for j in range(1, len_str1):
            j_less = j - 1
            if str2[i_less] == str1[j_less]:
                edits[i][j] = edits[i_less][j_less]
            else:
                edits[i][j] = 1 + min(edits[i_less][j_less], edits[i][j_less], edits[i_less][j])

    return edits[-1][-1]
