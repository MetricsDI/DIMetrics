import dime.hm.LineItemMetrics as LIM
import numpy as np
import pandas as pd

from dime.geometric import iou
from dime.hed import hed
from dime.textual import levenshtein_distance, lc_subsequence, lc_subsequence_torch
from dime.token_classification import proc_token_classification
from dime.uhed import uhed


# to do add pytest

def test_token_class():
    pref = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    ref = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    (p, r, f1) = proc_token_classification(pref, ref)
    assert p == 0.5


def test_lcs():
    pred = "abc"
    ref = "aaabc"
    tp, fp, fn = lc_subsequence(pred, ref)
    assert tp == 3.0


def test_lcs_torch():
    pred = "abc"
    ref = "aaabc"
    tp, fp, fn = lc_subsequence_torch(pred, ref)
    assert tp == 3.0


def test_lev():
    pred = "abc"
    ref = "aaabc"
    dist = levenshtein_distance(pred, ref)
    assert dist == 2


def test_iou():
    pred = [637, 773, 693, 782]
    ref = [639, 778, 698, 788]
    ol = np.round(iou(pred, ref), 2)
    assert ol == 0.25


def test_hed():
    # correct
    pred1 = {'Items': [{'ItemName': 'BASO TAHU',
                        'ItemPrice': '43,181',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '43,181'},
                       {'ItemName': 'ES JERUK',
                        'ItemPrice': '13,000',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '13,000'}],
             'Tax': '5,618',
             'Total': '61,799'}

    # missing first line item entirely
    pred2 = {'Items': [{'ItemName': 'ES JERUK',
                        'ItemPrice': '13,000',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '13,000'}],
             'Tax': '5,618',
             'Total': '61,799'}

    # predict one extra line item
    pred3 = {'Items': [{'ItemName': 'BASO TAHU',
                        'ItemPrice': '43,181',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '43,181'},
                       {'ItemName': 'BASO TAHU',
                        'ItemPrice': '43,181',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '43,181'},
                       {'ItemName': 'ES JERUK',
                        'ItemPrice': '13,000',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '13,000'}],
             'Tax': '5,618',
             'Total': '61,799'}

    # ground truth
    gt = {'Items': [{'ItemName': 'BASO TAHU',
                     'ItemPrice': '43,181',
                     'ItemQuantity': '1',
                     'ItemTotalPrice': '43,181'},
                    {'ItemName': 'ES JERUK',
                     'ItemPrice': '13,000',
                     'ItemQuantity': '1',
                     'ItemTotalPrice': '13,000'}],
          'Tax': '5,618',
          'Total': '61,799'}

    doc_right, fields_right = (hed(pred1, gt))
    doc_missing, fields_missing = (hed(pred2, gt))
    doc_extra, fields_extra = (hed(pred3, gt))

    assert doc_missing[1] == doc_extra[2]


def test_uhed():
    # correct
    pred1 = {'Items': [{'ItemName': 'ES JERUK',
                        'ItemPrice': '13,000',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '13,000'},
                       {'ItemName': 'BASO TAHU',
                        'ItemPrice': '43,181',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '43,181'}
                       ],
             'Tax': '5,618',
             'Total': '61,799'}

    # missing first line item entirely
    pred2 = {'Items': [{'ItemName': 'ES JERUK',
                        'ItemPrice': '13,000',
                        'ItemQuantity': '1',
                        'ItemTotalPrice': '13,000'}],
             'Tax': '5,618',
             'Total': '61,799'}

    # predict one extra line item
    pred3 = {'Items': [
        {'ItemName': 'ES JERUK',
         'ItemPrice': '13,000',
         'ItemQuantity': '1',
         'ItemTotalPrice': '13,000'},
        {'ItemName': 'BASO TAHU',
         'ItemPrice': '43,181',
         'ItemQuantity': '1',
         'ItemTotalPrice': '43,181'},
        {'ItemName': 'BASO TAHU',
         'ItemPrice': '43,181',
         'ItemQuantity': '1',
         'ItemTotalPrice': '43,181'}
    ],
        'Tax': '5,618',
        'Total': '61,799'}

    # ground truth
    gt = {'Items': [{'ItemName': 'BASO TAHU',
                     'ItemPrice': '43,181',
                     'ItemQuantity': '1',
                     'ItemTotalPrice': '43,181'},
                    {'ItemName': 'ES JERUK',
                     'ItemPrice': '13,000',
                     'ItemQuantity': '1',
                     'ItemTotalPrice': '13,000'}],
          'Tax': '5,618',
          'Total': '61,799'}

    doc_right, fields_right = (uhed(pred1, gt))
    doc_missing, fields_missing = (uhed(pred2, gt))
    doc_extra, fields_extra = (uhed(pred3, gt))

    assert doc_missing[1] == doc_extra[2]


def test_hm():
    input_file = 'tests/sample_file.csv'
    df = pd.read_csv(input_file)
    df.rename(columns={'Original_Map_Label': 'Map_Name_Original', 'Grouping': 'lineitem_original'}, inplace=True)

    df['idx'] = df.index
    df_groups = df.groupby(['fileId', 'PageNumber'])
    num_invoices = len(df_groups)
    lim1 = LIM.LineItemMetric(df_groups=df_groups)
    partialMatch_report = lim1._get_metrics()


if __name__ == "__main__":
    test_token_class()
    test_lcs()
    test_lcs_torch()
    test_lev()
    test_iou()
    test_hed()
    test_uhed()
    test_hm()

    print("PASS")
