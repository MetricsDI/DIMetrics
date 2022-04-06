import re

import numpy as np
import pandas as pd

from .geometric import iou
from .hed import hed
from .textual import levenshtein_distance, lc_subsequence
from .uhed import uhed


def get_iou_metrics(df):
    if not df['gt_bbox']:
        df['FP'] = 1
        ious = 0
    elif not df['pred_bbox']:
        df['FN'] = 1
        ious = 0
    else:
        ious = iou(df['gt_bbox'], df['pred_bbox'])
        df['TP'] = 1
    df['iou'] = ious
    return df


def get_precision(df):
    deno = df['TP'] + df['FP']
    if deno == 0:
        return 0
    precision = df['TP'] / deno
    return precision


def get_recall(df):
    deno = df['TP'] + df['FN']
    if deno == 0:
        return 0
    recall = df['TP'] / deno
    return recall


def get_f1(df, prec_col, recall_col):
    deno = df[prec_col] + df[recall_col]
    if deno == 0:
        return 0
    f1 = 2 * df[prec_col] * df[recall_col] / deno
    return f1


def get_iou_label(df):
    d = {}
    TP = sum(df["TP"])
    FN = sum(df["FN"])
    FP = sum(df["FP"])
    count = len(df)
    precision = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0 if TP + FN == 0 else TP / (TP + FN)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    d['iou_precision'] = round(precision, 2)
    d['iou_recall'] = round(recall, 2)
    d['iou_f1'] = round(f1, 2)
    d['support'] = int(count)
    return pd.Series(d, index=['iou_f1', 'iou_precision', 'iou_recall', 'support'])


def get_weighted_mean(iou_metrics_df, prefix, volum_col):
    total = iou_metrics_df[volum_col].sum() + 1e-10
    prec_mean = sum(iou_metrics_df[prefix + '_precision'] * iou_metrics_df[volum_col]) / total
    rec_mean = sum(iou_metrics_df[prefix + '_recall'] * iou_metrics_df[volum_col]) / total
    f1_mean = sum(iou_metrics_df[prefix + '_f1'] * iou_metrics_df[volum_col]) / total
    return pd.DataFrame(
        [{'f1_mean': round(f1_mean, 2), 'prec_mean': round(prec_mean, 2), 'recall_mean': round(rec_mean, 2)}])


def get_iou_metrics_df(df):
    df = df.loc[:, ['label', 'gt_bbox', 'pred_bbox']]
    df.loc[:, 'FN'] = 0
    df.loc[:, 'FP'] = 0
    df.loc[:, 'TP'] = 0
    metrics_df = df.apply(get_iou_metrics, axis=1).fillna(0)
    iou_metrics_df = metrics_df[['FN', 'FP', 'TP', 'label']].groupby('label').apply(get_iou_label)
    mean_df = get_weighted_mean(iou_metrics_df, 'iou', 'support')
    return iou_metrics_df, mean_df


def get_text_metrics(df, pattern=re.compile('[\W_]+')):
    df = df.loc[:, ['pred_text', 'gt_text', 'label']]
    df['direct'] = (df['gt_text'] == df['pred_text']).astype(float)
    df["gt_text_alpha"] = df["gt_text"].apply(lambda x: pattern.sub('', x))  # Remove non-alphanumeric characters
    df["pred_text_alpha"] = df["pred_text"].apply(lambda x: pattern.sub('', x))  # Remove non-alphanumeric characters
    df['direct_alpha'] = (df['gt_text_alpha'] == df['pred_text_alpha']).astype(float)
    df['levenshtein'] = df.apply(lambda x: levenshtein_distance(x['gt_text'], x['pred_text'], normalize=True), axis=1)
    df['levenshtein_alpha'] = df.apply(
        lambda x: levenshtein_distance(x['gt_text_alpha'], x['pred_text_alpha'], normalize=True), axis=1)
    df['lcsubsequence'] = df.apply(lambda x: lc_subsequence(x['gt_text'], x['pred_text']), axis=1)
    df['TP'] = df['lcsubsequence'].apply(lambda x: x[0])
    df['FP'] = df['lcsubsequence'].apply(lambda x: x[1])
    df['FN'] = df['lcsubsequence'].apply(lambda x: x[2])
    df['lcs_precision'] = df.apply(get_precision, axis=1)
    df['lcs_recall'] = df.apply(get_recall, axis=1)
    df['lcs_f1'] = df.apply(lambda x: get_f1(x, 'lcs_precision', 'lcs_recall'), axis=1)
    counts = pd.Series(df.groupby(['label']).count()['gt_text'], name='count')
    df.drop(columns=['pred_text', 'gt_text', 'TP', 'FP', 'FN', 'gt_text_alpha', 'pred_text_alpha'], inplace=True)
    metrics_df = pd.merge(df.groupby(['label']).mean(), counts, on=['label'])
    metrics_df['direct'] = metrics_df['direct'].apply(lambda x: round(x, 2))
    metrics_df['direct_alpha'] = metrics_df['direct_alpha'].apply(lambda x: round(x, 2))
    metrics_df['levenshtein'] = metrics_df['levenshtein'].apply(lambda x: round(x, 2))
    metrics_df['levenshtein_alpha'] = metrics_df['levenshtein_alpha'].apply(lambda x: round(x, 2))
    metrics_df['lcs_f1'] = metrics_df['lcs_f1'].apply(lambda x: round(x, 2))
    metrics_df['lcs_precision'] = metrics_df['lcs_precision'].apply(lambda x: round(x, 2))
    metrics_df['lcs_recall'] = metrics_df['lcs_recall'].apply(lambda x: round(x, 2))
    return metrics_df


def get_hed_document(gt_item_list, pred_item_list, unorder=False):
    if unorder:
        file_results = [uhed(gt_item, pred_item)[0] for gt_item, pred_item in zip(gt_item_list, pred_item_list)]
    else:
        file_results = [hed(gt_item, pred_item)[0] for gt_item, pred_item in zip(gt_item_list, pred_item_list)]
    file_hed_df = pd.DataFrame(np.stack(file_results), columns=['TP', 'FP', 'FN'])
    file_hed_df['precision'] = file_hed_df.apply(get_precision, axis=1)
    file_hed_df['recall'] = file_hed_df.apply(get_recall, axis=1)
    file_hed_df['f1-score'] = file_hed_df.apply(lambda x: get_f1(x, "precision", "recall"), axis=1)
    file_hed_df['precision'] = file_hed_df['precision'].apply(lambda x: round(x, 2))
    file_hed_df['recall'] = file_hed_df['recall'].apply(lambda x: round(x, 2))
    file_hed_df['f1-score'] = file_hed_df['f1-score'].apply(lambda x: round(x, 2))
    return file_hed_df.drop(columns=['TP', 'FP', 'FN']).mean()


def get_hed_label(gt_item_list, pred_item_list, unorder=False):
    if unorder:
        results = [uhed(gt_item, pred_item)[1] for gt_item, pred_item in zip(gt_item_list, pred_item_list)]
    else:
        results = [hed(gt_item, pred_item)[1] for gt_item, pred_item in zip(gt_item_list, pred_item_list)]
    hed_df = pd.concat([pd.DataFrame(result).T.reset_index() for i, result in enumerate(results)])
    hed_df = hed_df.rename(columns={0: 'TP', 1: 'FP', 2: 'FN'})
    hed_df['precision'] = hed_df.apply(get_precision, axis=1)
    hed_df['recall'] = hed_df.apply(get_recall, axis=1)
    hed_df['f1-score'] = hed_df.apply(lambda x: get_f1(x, "precision", "recall"), axis=1)
    label_df = hed_df.drop(columns=['TP', 'FP', 'FN']).groupby('index').mean()
    mean_df = hed_df.drop(columns=['TP', 'FP', 'FN']).mean()
    return label_df, mean_df
