import os
import timeit

import numpy as np

from dime.geometric import iou
from dime.hed import hed
from dime.textual import levenshtein_distance, lc_subsequence, lc_subsequence_torch
from dime.uhed import uhed

import dime.hm.LineItemMetrics as LIM


# timeit parameters for performance measurement
PERF_MEASUREMENT_REPETITION = 5
PERF_MEASUREMENT_LOOPS = 10
PERF_TEST_DATA_FILE = 'str_distance_test_data.csv'


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


# def test_hm():
#     input_file = 'tests/sample_file.csv'
#     df = pd.read_csv(input_file)
#     df.rename(columns={'Original_Map_Label': 'Map_Name_Original', 'Grouping': 'lineitem_original'}, inplace=True)
#
#     df['idx'] = df.index
#     df_groups = df.groupby(['fileId', 'PageNumber'])
#     num_invoices = len(df_groups)
#     lim1 = LIM.LineItemMetric(df_groups=df_groups)
#     partialMatch_report = lim1._get_metrics()


def performance_print(stmt, setup='', number=PERF_MEASUREMENT_REPETITION, repeat=PERF_MEASUREMENT_LOOPS, verbose=False):
    """
    Adapted from timeit module.
    :param stmt: Statement measured.
    :param setup: Imports and other setup of the test.
    :param number: number of loops of the code in one test.
    :param repeat: number of repetitions of the test.
    :param verbose: as in timeit module
    :return: None
    """
    tmr = timeit.Timer(setup=setup, stmt=stmt)
    raw_timings = tmr.repeat(number, repeat)

    units = {"nsec": 1e-9, "usec": 1e-6, "msec": 1e-3, "sec": 1.0}
    precision = 3

    def format_time(dt):
        scales = [(scale, unit) for unit, scale in units.items()]
        scale, unit = scales[-1]
        for scale, unit in sorted(scales, reverse=True):
            if dt >= scale:
                break

        return "%.*g %s" % (precision, dt / scale, unit)

    if verbose:
        print("raw times: %s\n" % ", ".join(map(format_time, raw_timings)))

    timings = [dt / number for dt in raw_timings]

    best = min(timings)
    print("%d loop%s, best of %d: %s per loop"
          % (number, 's' if number != 1 else '',
             repeat, format_time(best)))

    best = min(timings)
    worst = max(timings)
    if worst >= best * 4:
        import warnings
        warnings.warn_explicit("The test results are likely unreliable. "
                               "The worst time (%s) was more than four times "
                               "slower than the best time (%s)."
                               % (format_time(worst), format_time(best)),
                               UserWarning, '', 0)


def _perf_test_fun(fun_name, module_name, optimization_type):
    print(f"\n{optimization_type} version of function '{module_name} > {fun_name}'")
    performance_print(stmt=f'[{fun_name}(*ln.strip().split("\t")) '
                           f'for ln in open("{PERF_TEST_DATA_FILE}", "rt").readlines()]',
                      setup=f"from {module_name} import {fun_name}")


def test_perf_levenshtein_distance():
    _perf_test_fun('levenshtein_distance', 'dime.textual', 'Cython-compiled')


def test_perf_levenshtein_distance2():
    _perf_test_fun('levenshtein_distance', 'tests.textual2', 'Standard python')


def test_perf_levenshtein_distance3():
    _perf_test_fun('levenshtein_distance', 'tests.textual3', 'Numba')


def test_perf_lc_subsequence():
    _perf_test_fun('lc_subsequence', 'dime.textual', 'Cython-compiled')


def test_perf_lc_subsequence2():
    _perf_test_fun('lc_subsequence', 'tests.textual2', 'Standard python')


def test_perf_lc_subsequence3():
    _perf_test_fun('lc_subsequence', 'tests.textual3', 'Numba')


def test_perf_str_exact_match2():
    _perf_test_fun('str_exact_match', 'dime.textual', 'Standard python')


def test_perf_str_exact_match():
    _perf_test_fun('str_exact_match', 'tests.textual2', 'Cython-compiled')


def test_perf_str_exact_match3():
    _perf_test_fun('str_exact_match', 'tests.textual3', 'Numba')


if __name__ == "__main__":
    print(f"Run tests using the following commands:\n\tcd {os.curdir}\n\tpytest")
