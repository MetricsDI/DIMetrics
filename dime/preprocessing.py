import json
import os

import numpy as np
import pandas as pd
from lxml import objectify


def combine_text(gt_df):
    gt_df['X'] = gt_df['bbox'].apply(lambda x: x[0])
    new_gt_df = pd.DataFrame()
    for i, group in gt_df.groupby(['group_label']):
        for label in group['label'].unique():
            g = group[group['label'] == label]
            if g.shape[0] > 1:
                group_label = g.iloc[0]["group_label"]
                g = g.sort_values(by='X')
                combined_bbox = [np.inf, np.inf, -np.inf, -np.inf]
                combined_text = ""

                for j, row in g.iterrows():
                    combined_bbox[0] = min(row['bbox'][0], combined_bbox[0])
                    combined_bbox[1] = min(row['bbox'][1], combined_bbox[1])
                    combined_bbox[2] = max(row['bbox'][2], combined_bbox[2])
                    combined_bbox[3] = max(row['bbox'][3], combined_bbox[3])

                    combined_text += row['text'] + ' '
                new_gt_df = new_gt_df.append(pd.DataFrame([[combined_bbox, combined_text.strip(), label, group_label]],
                                                          columns=['bbox', 'text', 'label', 'group_label']),
                                             ignore_index=True, sort=True)
            else:
                new_gt_df = new_gt_df.append(g.drop(columns=['X']), ignore_index=True, sort=True)
    return new_gt_df


def assign_group(df, lineitem_prefix, line_threshold=25):
    df["group_label"] = 0
    if isinstance(lineitem_prefix, str):
        temp_df = df.loc[df["label"].apply(lambda x: lineitem_prefix in x), :]
    elif isinstance(lineitem_prefix, list):
        temp_df = df.loc[df["label"].apply(lambda x: x in lineitem_prefix), :]
    if temp_df.shape[0] == 0:
        return df
    if isinstance(lineitem_prefix, str):
        notline_df = df.loc[df["label"].apply(lambda x: lineitem_prefix not in x), :]
    elif isinstance(lineitem_prefix, list):
        notline_df = df.loc[df["label"].apply(lambda x: x not in lineitem_prefix), :]
    temp_df["Y"] = temp_df["bbox"].apply(lambda x: x[1])
    temp_df.sort_values(by="Y", inplace=True)
    prev = temp_df.iloc[0]["Y"]
    group_label = 1
    new_temp_df = pd.DataFrame()
    for i, row in temp_df.iterrows():
        if row["Y"] - prev <= line_threshold:
            row["group_label"] = group_label
        else:
            group_label += 1
            row["group_label"] = group_label
            prev = row["Y"]
        new_temp_df = new_temp_df.append(row, ignore_index=True, sort=True)

    return new_temp_df.append(notline_df, ignore_index=True, sort=True)


def process_cdip_gt(gt_file, pdf_filename=None):
    with open(gt_file) as f:
        gt_dict = json.load(f)

    gt_df = pd.DataFrame(gt_dict['1'])
    gt_df['bbox'] = gt_df['bounding_box'].apply(lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]])
    gt_df['text'] = gt_df['bounding_box'].apply(lambda x: x[-1])
    gt_df = gt_df.rename(columns={'classification_label': 'label'})
    gt_df = gt_df.drop(columns=['activity_label', 'bounding_box', 'class_probability'])
    if pdf_filename:
        gt_df['file'] = pdf_filename
    return gt_df


def process_layoutlm_pred_file(gt_file, lineitem_prefix, line_threshold, pdf_filename=None, cord=False,
                               remove_label=False):
    with open(gt_file) as f:
        gt_dict = json.load(f)

    gt_df = pd.DataFrame(gt_dict["predictions"])
    gt_df['bbox'] = gt_df['bounding_box'].apply(lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]])
    gt_df['text'] = gt_df['bounding_box'].apply(lambda x: x[-1])

    gt_df = gt_df.rename(columns={'classification_label': 'label'})
    gt_df['label'] = gt_df['label'].apply(lambda x: x.replace("S-", ""))
    gt_df = gt_df[gt_df['label'] != "No class"]
    if gt_df.shape[0] == 0:
        return pd.DataFrame()
    gt_df = gt_df[gt_df['label'].notna()]
    gt_df = gt_df.drop(columns=['bounding_box', 'class_probability', 'PSL_classification_label', 'box_id', 'scores'],
                       errors='ignore')
    if remove_label:
        gt_df = gt_df.drop(columns=['PSL_lineitem_number'], axis=1)
    if 'PSL_lineitem_number' in gt_df.columns:
        gt_df = gt_df.rename(columns={'PSL_lineitem_number': 'group_label'})
        gt_df.loc[gt_df["group_label"] == "", "group_label"] = 0
    else:
        gt_df = assign_group(gt_df, lineitem_prefix, line_threshold)

    if not cord:
        new_gt_df = combine_text(gt_df)
    else:
        new_gt_df = gt_df
    if pdf_filename:
        new_gt_df['file'] = pdf_filename
    return new_gt_df


def process_google_ai_preds(pred_file, field_mapping):
    pred_df = pd.read_csv(pred_file)
    pred_df.fillna("", inplace=True)
    new_df = []
    for i, row in pred_df.iterrows():
        for label in ["line_item/description", "line_item/amount", "line_item/quantity", "line_item/unit_price"]:
            if row[label] != "":
                new_df.append({'label': field_mapping[label], 'file': row['File_Name'], 'text': row[label],
                               'bbox': row[label + "_bb"], 'group_label': row['ItemNo']})
    new_df = pd.DataFrame(new_df)
    new_df['bbox'] = new_df['bbox'].astype(str).apply(lambda x: x[1:-1].split(","))
    new_df['bbox'] = new_df['bbox'].apply(lambda x: [float(x[0]), float(x[1]), float(x[4]), float(x[5])])
    return new_df


def process_deepcpcfg_cord_xml_preds(file, save_dir=None):
    tree = objectify.parse(file)
    root = tree.getroot()
    res = []
    group = 1
    for line in root["Receipt"]["LineItem"].iterchildren():
        for lineitem_label in ["MenuCnt", "MenuNm", "MenuPrice", "MenuUnitprice"]:
            if not hasattr(line, lineitem_label):
                continue
            for i in line[lineitem_label].getchildren():
                row = {'text': i.text,
                       'bbox': [float(i.attrib.get('l')), float(i.attrib.get('t')), float(i.attrib.get('r')),
                                float(i.attrib.get('b'))],
                       'label': lineitem_label, 'group_label': group}
                res.append(row)
        group += 1
    if hasattr(root["Receipt"], 'TotalPrice'):
        price = root["Receipt"]['TotalPrice'].getchildren()[0]
        row = {'text': price.text,
               'bbox': [float(price.attrib.get('l')), float(price.attrib.get('t')), float(price.attrib.get('r')),
                        float(price.attrib.get('b'))],
               'label': 'TotalPrice', 'group_label': 0}
        res.append(row)
    df = pd.DataFrame(res)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, str(os.path.basename(file)).replace(".xml", ".csv")), index=False)
    return df


def process_deepcpcfg_cdip_xml_preds(file, save_dir=None):
    tree = objectify.parse(file)
    root = tree.getroot()
    res = []
    group = 1
    for line in root["Invoice"]["LineItem"].iterchildren():
        for lineitem_label in ["ItemDescription", "ItemTotal", "ItemUnitCount", "ItemUnitValue"]:
            if hasattr(line, lineitem_label):
                for i in line[lineitem_label].getchildren():
                    row = {'text': i.text,
                           'bbox': [float(i.attrib.get('l')), float(i.attrib.get('t')), float(i.attrib.get('r')),
                                    float(i.attrib.get('b'))],
                           'label': lineitem_label, 'group_label': group}
                    res.append(row)
        group += 1
    for label in ["InvoiceNumber", "TotalAmount", "IssuedDate"]:
        if hasattr(root["Invoice"], label):
            price = root["Invoice"][label].getchildren()[0]
            row = {'text': price.text,
                   'bbox': [float(price.attrib.get('l')), float(price.attrib.get('t')), float(price.attrib.get('r')),
                            float(price.attrib.get('b'))],
                   'label': label, 'group_label': 0}
            res.append(row)
    df = pd.DataFrame(res)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, str(os.path.basename(file)).replace(".xml", ".csv")), index=False)
    return df
