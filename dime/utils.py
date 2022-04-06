import os
import re

import pandas as pd

from .geometric import iou
from .preprocessing import *
from .textual import lc_subsequence, levenshtein_distance


def process_gt_lineitems_json(file, field_mapping=None):
    with open(file) as f:
        data = json.load(f)
    new_data = dict()
    new_data["Items"] = []
    for line in data['LineItem']:
        new_line = {}
        for key, value in line.items():
            if value == "":
                continue
            if field_mapping and key in field_mapping.keys():
                new_line[field_mapping[key]] = line[key]
            else:
                new_line[key] = line[key]
        new_data["Items"].append(new_line)
    for key, value in data.items():
        if key != 'LineItem':
            if value == "":
                continue
            if field_mapping and key in field_mapping.keys():
                new_data[field_mapping[key]] = data[key]
            else:
                new_data[key] = data[key]

    return new_data


def box_inside(y_true, y_pred):
    xA = max(y_true[0], y_pred[0])
    yA = max(y_true[1], y_pred[1])
    xB = min(y_true[2], y_pred[2])
    yB = min(y_true[3], y_pred[3])
    inset = [xA, yA, xB, yB]
    if all([inset[i] == y_true[i] for i in range(4)]) or all([inset[i] == y_pred[i] for i in range(4)]):
        return True
    else:
        return False


def get_matched_gt_pred(gt_df, pred_df, iou_threshold=0.51):
    gt_df.drop(columns=['file'], axis=1, inplace=True, errors='ignore')
    pred_df.drop(columns=['file'], axis=1, inplace=True, errors='ignore')
    gt_df["X"] = gt_df['bbox'].apply(lambda x: x[0])
    gt_df["Y"] = gt_df['bbox'].apply(lambda x: x[1])
    gt_df['gt_id'] = gt_df.index
    gt_df.rename(columns={'group_label': 'gt_group_label', 'bbox': 'gt_bbox'}, inplace=True)
    pred_df["X"] = pred_df['bbox'].apply(lambda x: x[0])
    pred_df["Y"] = pred_df['bbox'].apply(lambda x: x[1])
    pred_df['pred_id'] = pred_df.index
    pred_df.rename(columns={'group_label': 'pred_group_label', 'bbox': 'pred_bbox'}, inplace=True)
    # Get same (Label, X,Y,Text) data
    common = gt_df.merge(pred_df, on=['label', 'text', 'X', 'Y'], how='inner')
    common.drop(columns=["X", "Y"], axis=1, inplace=True)
    common.rename(columns={'text': 'gt_text'}, inplace=True)
    common.loc[:, 'pred_text'] = common.loc[:, 'gt_text']
    gt_rest = gt_df[gt_df.gt_id.apply(lambda x: x not in common.gt_id.tolist())]
    gt_rest.drop(columns=["X", "Y"], axis=1, inplace=True, errors='ignore')
    pred_rest = pred_df[pred_df.pred_id.apply(lambda x: x not in common.pred_id.tolist())]
    pred_rest.drop(columns=["X", "Y"], axis=1, inplace=True, errors='ignore')

    if gt_rest.shape[0] == 0:
        gt_df.drop(columns=['gt_id'], axis=1, inplace=True)
        pred_rest.drop(columns=['pred_id'], axis=1, inplace=True)
        combined_df = gt_df.append(pred_rest, sort=True, ignore_index=True)
        return combined_df
    else:
        used_gt_ids = []
        used_pred_ids = []
        mix_df = pd.DataFrame(
            columns=['label', 'gt_group_label', 'gt_text', 'gt_bbox', 'pred_text', 'pred_bbox', 'pred_group_label'])
        add_idx = 0
        # If there are gt data which is not in pred, check if there is similar bbox which should be the identical one.
        for j, gt_row in gt_rest.iterrows():
            gt_label = gt_row['label']
            for i, pred_row in pred_rest.iterrows():
                if pred_row['label'] == gt_label:
                    ious = iou(gt_row['gt_bbox'], pred_row['pred_bbox'])
                    if (ious >= iou_threshold) or (box_inside(gt_row['gt_bbox'], pred_row['pred_bbox'])):
                        used_pred_ids.append(pred_row['pred_id'])
                        used_gt_ids.append(gt_row['gt_id'])
                        mix_df.loc[add_idx] = (gt_label, gt_row['gt_group_label'], gt_row['text'], gt_row['gt_bbox'],
                                               pred_row['text'], pred_row['pred_bbox'], pred_row['pred_group_label'])

                        add_idx += 1
                    else:
                        continue
                else:
                    continue
        # Append all pred data which has no matching gt data
        pred_nomatch = pred_rest[pred_rest.pred_id.apply(lambda x: x not in used_pred_ids)]
        pred_nomatch.rename(columns={'text': 'pred_text'}, inplace=True)
        gt_nomatch = gt_rest[gt_rest.gt_id.apply(lambda x: x not in used_gt_ids)]
        gt_nomatch.rename(columns={'text': 'gt_text'}, inplace=True)
        combined_df = common.append([mix_df, pred_nomatch, gt_nomatch], sort=True, ignore_index=True)
        combined_df.drop(columns=['pred_id', 'gt_id'], axis=1, inplace=True)
        return combined_df


def sort_line_items(item_df, by='group_label'):
    # by='mid_y' if you want to sort line items by Y coordinate.
    if len(item_df) == 0:
        return []
    ans = []
    for i, row in item_df.sort_values(by=[by]).iterrows():
        d = {}
        for j, col in row.iteritems():
            if type(col) == dict:
                d[j] = col['text']
        ans.append(d)
    return ans


def get_line_item_dic(df, lineitem_prefix, cord=False):
    dic = {}
    if cord:
        df = combine_text(df)
    item_df, other_df = pd.DataFrame(), pd.DataFrame()
    if isinstance(lineitem_prefix, str):
        item_df = df[df['label'].apply(lambda x: lineitem_prefix in x)]
        other_df = df[df['label'].apply(lambda x: lineitem_prefix not in x)]
    elif isinstance(lineitem_prefix, list):
        item_df = df[df['label'].apply(lambda x: x in lineitem_prefix)]
        other_df = df[df['label'].apply(lambda x: x not in lineitem_prefix)]
    for i, row in other_df.iterrows():
        dic[row['label']] = row['text']
    if len(item_df) > 0:
        item_df = get_item_df(item_df)
    dic['Items'] = sort_line_items(item_df)
    return dic


def get_merged_df_layoutlm(PRED_DIR, GT_DIR, GT_Lineitem_DIR, lineitem_prefix, address_labels, gt_lineitems2pred, line_threshold=25, cord=False,
                           remove_label=False):
    final_df = pd.DataFrame()
    gt_hed_lists = []
    pred_hed_lists = []
    file_num = 0
    for pred_file in os.listdir(PRED_DIR):
        if pred_file == '.DS_Store':
            continue
        if pred_file[:-5][-12:] != "pred_grouped":
            continue
        file_prefix = pred_file[:-5][:-len("_abbyy_pred_grouped")]
        pred_file = os.path.join(PRED_DIR, pred_file)
        if cord:
            gt_file = os.path.join(GT_DIR, file_prefix + '.json')
        else:
            gt_file = os.path.join(GT_DIR, file_prefix + '.pdf_annotations.json')
        if not os.path.exists(gt_file):
            continue

        pred_df = process_layoutlm_pred_file(pred_file, lineitem_prefix, line_threshold, os.path.basename(pred_file),
                                             cord, remove_label)
        if pred_df.shape[0] == 0:
            continue
        file_num += 1
        if cord:
            gt_df = process_cord_file(gt_file, pdf_filename=os.path.basename(gt_file))
        else:
            gt_df = process_cdip_gt(gt_file, pdf_filename=os.path.basename(gt_file))

        pred_li = get_line_item_dic(pred_df.loc[pred_df['label'].apply(lambda x: x not in address_labels), :],
                                    lineitem_prefix, cord)
        # If you want to sort GT line items by Y coordinate.
        # gt_li = get_line_item_dic(gt_df.loc[gt_df['label'].apply(lambda x: x not in address_labels), :],
        #                           lineitem_prefix, cord)
        if cord:
            file_id = file_prefix.replace("CORD_001_test_receipt_", "")
            gt_lineitem_file = "test_receipt_" + file_id + ".json"
        else:
            file_id = file_prefix.replace("ARIA_CDIP_Test_", "")
            gt_lineitem_file = file_id + ".json"
        gt_lineitem_path = os.path.join(GT_Lineitem_DIR, gt_lineitem_file)

        if os.path.exists(gt_lineitem_path):
            gt_li = process_gt_lineitems_json(gt_lineitem_path, gt_lineitems2pred)
            pred_hed_lists.append(pred_li)
            gt_hed_lists.append(gt_li)

        df = get_matched_gt_pred(gt_df, pred_df)
        df.loc[:, 'file'] = os.path.basename(gt_file)
        final_df = final_df.append(df, sort=True, ignore_index=True)
    final_df = final_df.fillna(0)
    final_df['pred_text'] = final_df['pred_text'].replace(0, '')
    final_df['gt_text'] = final_df['gt_text'].replace(0, '')
    return final_df, file_num, gt_hed_lists, pred_hed_lists


def get_merged_df_FR(PRED_DIR, GT_DIR, GT_Lineitem_DIR, cord, field_mapping, lineitem_prefix, multiplier, gt_lineitems2pred):
    final_df = pd.DataFrame()
    file_num = 0
    gt_hed_lists = []
    pred_hed_lists = []
    for pred in os.listdir(PRED_DIR):
        if pred == '.DS_Store':
            continue
        if cord:
            gt_file = os.path.join(GT_DIR, pred)
        else:
            gt_file = os.path.join(GT_DIR, pred[:-5] + '.pdf_annotations.json')
        pred_file = os.path.join(PRED_DIR, pred)
        if not os.path.exists(gt_file):
            continue
        pred_df, item_df, _, _ = process_pred_file(pred_file, pdf_filename=pred, multiplier=multiplier)
        pred_df = pred_df[pred_df['label'].notna()]
        if len(item_df) > 0:
            item_df['label'] = item_df['label'].map(field_mapping)
            item_df = item_df[item_df['label'].notna()]
            pred_df = pred_df.append(item_df, sort=True, ignore_index=True)
        if pred_df.shape[0] == 0:
            continue
        file_num += 1
        if cord:
            gt_df = process_cord_file(gt_file, os.path.basename(gt_file))
        else:
            gt_df = process_cdip_gt(gt_file, os.path.basename(gt_file))
        gt_df['label'] = gt_df[gt_df['label'].apply(lambda x: x in field_mapping.keys())]
        gt_df['label'] = gt_df['label'].map(field_mapping)
        gt_df = gt_df[gt_df['label'].notna()]
        pred_df = pred_df[pred_df['label'].apply(lambda x: x in field_mapping.values())]
        pred_df = pred_df[pred_df['label'].notna()]

        pred_li = get_line_item_dic(pred_df, lineitem_prefix, cord)
        # If you want to sort GT line items by Y coordinate.
        # gt_li = get_line_item_dic(gt_df, lineitem_prefix, cord)
        if cord:
            file_id = pred[:-5].replace("CORD_001_test_receipt_", "")
            gt_lineitem_file = "test_receipt_"+file_id+".json"
        else:
            file_id = pred[:-5].replace("ARIA_CDIP_Test_", "")
            gt_lineitem_file = file_id + ".json"
        gt_lineitem_path = os.path.join(GT_Lineitem_DIR, gt_lineitem_file)
        if os.path.exists(gt_lineitem_path):
            gt_li = process_gt_lineitems_json(gt_lineitem_path, gt_lineitems2pred)
            pred_hed_lists.append(pred_li)
            gt_hed_lists.append(gt_li)

        df = get_matched_gt_pred(gt_df, pred_df)
        df.loc[:, 'file'] = os.path.basename(pred)
        final_df = final_df.append(df, sort=True, ignore_index=True)

    final_df = final_df[final_df['label'].notna()]
    final_df = final_df.fillna(0)
    final_df['pred_text'] = final_df['pred_text'].replace(0, '')
    final_df['gt_text'] = final_df['gt_text'].replace(0, '')
    return final_df, file_num, gt_hed_lists, pred_hed_lists


def get_merged_df_DeepCPCFG(PRED_DIR, GT_DIR, cord, field_mapping, save_dir=None):
    final_df = pd.DataFrame()
    file_num = 0
    for pred in os.listdir(PRED_DIR):
        if pred == '.DS_Store':
            continue
        if cord:
            gt_file = os.path.join(GT_DIR, "CORD_001_" + pred.replace(".xml", "") + ".json")
        else:
            gt_file = os.path.join(GT_DIR, "ARIA_CDIP_Test_" + pred.replace(".xml", "") + '.pdf_annotations.json')
        pred_file = os.path.join(PRED_DIR, pred)
        if not os.path.exists(gt_file):
            continue
        if cord:
            pred_df = process_deepcpcfg_cord_xml_preds(pred_file, save_dir)
        else:
            pred_df = process_deepcpcfg_cdip_xml_preds(pred_file, save_dir)
        pred_df = pred_df[pred_df['label'].notna()]
        if pred_df.shape[0] == 0:
            continue
        file_num += 1
        if cord:
            gt_df = process_cord_file(gt_file, os.path.basename(gt_file))
        else:
            gt_df = process_cdip_gt(gt_file, os.path.basename(gt_file))
        gt_df['label'] = gt_df[gt_df['label'].apply(lambda x: x in field_mapping.keys())]
        gt_df['label'] = gt_df['label'].map(field_mapping)
        gt_df = gt_df[gt_df['label'].notna()]
        df = get_matched_gt_pred(gt_df, pred_df)
        df.loc[:, 'file'] = os.path.basename(pred)
        final_df = final_df.append(df, sort=True, ignore_index=True)

    final_df = final_df[final_df['label'].notna()]
    final_df = final_df.fillna(0)
    final_df['pred_text'] = final_df['pred_text'].replace(0, '')
    final_df['gt_text'] = final_df['gt_text'].replace(0, '')
    return final_df, file_num


def get_marged_df_googleAI(pred_file, GT_DIR, GT_Lineitem_DIR, LineItemPrefix, field_mapping, gt_lineitems2pred):
    file_df = process_google_ai_preds(pred_file, field_mapping)
    final_df = pd.DataFrame()
    gt_hed_lists = []
    pred_hed_lists = []
    file_num = 0
    for file in file_df['file'].unique():
        pred_df = file_df.loc[file_df['file'] == file, :]
        gt_file = os.path.join(GT_DIR, file + '_annotations.json')
        if not os.path.exists(gt_file):
            continue
        file_num += 1
        gt_df = process_cdip_gt(gt_file, pdf_filename=os.path.basename(gt_file))
        gt_df = gt_df.loc[gt_df['label'].apply(lambda x: x in field_mapping.values())]
        pred_li = get_line_item_dic(pred_df, LineItemPrefix, False)
        # If you want to sort GT line items by Y coordinate.
        # gt_li = get_line_item_dic(gt_df, LineItemPrefix, False)

        file_id = file[:-4].replace("ARIA_CDIP_Test_", "")
        gt_lineitem_file = file_id+".json"
        gt_lineitem_path = os.path.join(GT_Lineitem_DIR, gt_lineitem_file)
        if os.path.exists(gt_lineitem_path):
            gt_li = process_gt_lineitems_json(gt_lineitem_path, gt_lineitems2pred)
            pred_hed_lists.append(pred_li)
            gt_hed_lists.append(gt_li)

        df = get_matched_gt_pred(gt_df, pred_df)
        df.loc[:, 'file'] = os.path.basename(gt_file)
        final_df = final_df.append(df, sort=True, ignore_index=True)
    final_df = final_df.fillna(0)
    final_df['pred_text'] = final_df['pred_text'].replace(0, '')
    final_df['gt_text'] = final_df['gt_text'].replace(0, '')
    return final_df, file_num, gt_hed_lists, pred_hed_lists


def compare_dic(gt_dic, pred_dic):
    text_list = []
    for label, text in gt_dic.items():
        if label != "Items":
            if label in pred_dic.keys():
                text_list.append({'gt_text': text, 'pred_text': pred_dic[label], 'label': label})
                del pred_dic[label]
            else:
                text_list.append({'gt_text': text, 'pred_text': '', 'label': label})
    if len(pred_dic) > 0:
        for label, text in pred_dic.items():
            if label != "Items":
                text_list.append({'gt_text': '', 'pred_text': text, 'label': label})
    return text_list


def hedLists2textDf(gt_hed_lists, pred_hed_lists):
    text_df = []
    for i in range(len(gt_hed_lists)):
        gt_dic = gt_hed_lists[i]
        pred_dic = pred_hed_lists[i]
        min_len = min(len(gt_dic["Items"]), len(pred_dic["Items"]))
        for j in range(min_len):
            text_list = compare_dic(gt_dic["Items"][j], pred_dic["Items"][j])
            text_df.extend(text_list)
        for k in range(min_len, len(gt_dic["Items"])):
            for label, text in gt_dic["Items"][k].items():
                text_df.append({'gt_text': text, 'pred_text': '', 'label': label})
        for k in range(min_len, len(pred_dic["Items"])):
            for label, text in pred_dic["Items"][k].items():
                text_df.append({'gt_text': '', 'pred_text': text, 'label': label})

        text_df.extend(compare_dic(gt_dic, pred_dic))
    return pd.DataFrame(text_df)


def process_pred_row(label, value, filename=None, group_label=None, multiplier=1):
    if 'boundingBox' in value:
        bbox = value['boundingBox']
        text = value['text']
        min_x = bbox[0] * multiplier
        min_y = bbox[1] * multiplier
        max_x = bbox[4] * multiplier
        max_y = bbox[5] * multiplier
        d = {'label': label, 'text': text, 'bbox': [min_x, min_y, max_x, max_y], 'file': filename,
             'group_label': group_label}
        return d
    else:
        # No bounding boxes -> Blank prediction
        return {'label': label, 'text': "", 'bbox': [0, 0, 0, 0], 'file': filename, 'group_label': group_label}


def process_pred_file(pred_file, pdf_filename=None, multiplier=1):
    with open(pred_file) as f:
        pred_dict = json.load(f)

    preds = pred_dict['analyzeResult']['documentResults'][0]['fields']
    height = int(pred_dict['analyzeResult']['readResults'][0]['height'] * multiplier)
    width = int(pred_dict['analyzeResult']['readResults'][0]['width'] * multiplier)

    pred_list = [process_pred_row(key, value, pdf_filename, multiplier=multiplier) for (key, value) in preds.items() if
                 key != 'Items']

    item_list = []
    if 'Items' in preds:
        item_dict = preds['Items']
        for i, item in enumerate(item_dict['valueArray']):
            if 'valueObject' in item:
                item_list.extend(
                    [process_pred_row(key, value, filename=pdf_filename, group_label=i + 1, multiplier=multiplier) for
                     key, value in item['valueObject'].items()])

    return pd.DataFrame(pred_list), pd.DataFrame(item_list), height, width


# CORD
def process_cord_file(gt_file, pdf_filename=None):
    """Processes cord files

    Args:
        gt_file ([type]): [description]
        pdf_filename ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    with open(gt_file) as f:
        gt_dict = json.load(f)

    ## Adding most of the main fields to dataframe
    gt_df = pd.DataFrame(gt_dict['valid_line'])

    def combineTextandBox(row_words):
        combined_text = ""
        combined_bbox = [np.inf, np.inf, -np.inf, -np.inf]

        for word in row_words:
            # print(word)
            if not word['is_key']:
                combined_text += (word['text'] + " ")
                combined_bbox[0] = min(word['quad']['x1'], combined_bbox[0])
                combined_bbox[1] = min(word['quad']['y1'], combined_bbox[1])
                combined_bbox[2] = max(word['quad']['x3'], combined_bbox[2])
                combined_bbox[3] = max(word['quad']['y3'], combined_bbox[3])
            word['bbox'] = [0, 0, 0, 0]
            word['bbox'][0] = word['quad']['x1']
            word['bbox'][1] = word['quad']['y1']
            word['bbox'][2] = word['quad']['x3']
            word['bbox'][3] = word['quad']['y3']
            del word['quad']

        return combined_text[:-1], combined_bbox

    gt_df['text'], gt_df['bbox'] = zip(*gt_df['words'].apply(combineTextandBox))
    gt_df = gt_df[gt_df['text'] != ""]
    # print(gt_df['words'])
    gt_df = gt_df.rename(columns={'group_id': 'group_label', 'category': 'label'})
    gt_df = gt_df.drop(columns=['words'])  # Temporary. Using only Field Level Data

    if pdf_filename:
        gt_df['file'] = pdf_filename
    return gt_df


def get_dicts(df, ListPrefix="Item"):
    """Summary

    Args:
        df ([type]): [description]
        ListPrefix (str, optional): [description]. Defaults to "Item".

    Returns:
        [type]: [description]
    """
    if len(df) == 0:
        return {}
    ans = []
    for i, row in df.iterrows():
        d = {}
        for j, col in row.iteritems():
            if ListPrefix in j:
                if type(col) == dict:
                    d[j] = col['text']
        ans.append(d)
    return ans


def get_item_df(item_rows):
    """Summary

    Args:
        item_rows ([type]): [description]

    Returns:
        [type]: [description]
    """
    items = []
    for i, group in item_rows.groupby(['group_label']):
        filename = group['file'].values[0]
        g = group.set_index('label').drop(columns=['file'])

        combined_bbox = [np.inf, np.inf, -np.inf, -np.inf]
        combined_text = ""
        g = g.sort_index()

        for j, row in g.iterrows():
            combined_bbox[0] = min(row['bbox'][0], combined_bbox[0])
            combined_bbox[1] = min(row['bbox'][1], combined_bbox[1])
            combined_bbox[2] = max(row['bbox'][2], combined_bbox[2])
            combined_bbox[3] = max(row['bbox'][3], combined_bbox[3])

            combined_text += row['text'] + ' '

        # combined_text = pattern.sub('',combined_text)   # Removes Non-Alpha Characters
        item = g.to_dict('index')
        item['combined_text'] = combined_text
        item['bbox'] = combined_bbox
        item['mid_y'] = (combined_bbox[3] + combined_bbox[1]) / 2
        item['group_label'] = int(i)
        item['file'] = filename
        items.append(item)

    item_df = pd.DataFrame(items)

    # item_df = item_df.set_index('group_label')
    return item_df


class ImageTransformation:
    def __init__(self):
        pass

    def inch_pix(self, list_a):
        """Function to convert inch to pixel"""
        new_list = []
        for val in list_a:
            pix = val * 300
            new_list.append(pix)
        return new_list

    def boxA(self, X, Y, Width, height):
        """Toma puntos del GT y los convierte en punto inicial (arriba-izquierda) y final abajo-derecha"""
        left_X = X
        left_Y = Y
        right_X = X + Width
        right_Y = Y + height
        return left_X, left_Y, right_X, right_Y

    def boxB(self, lista):
        """Toma puntos del MS-form recognazer en punto inicial (arriba-izquierda) y final abajo-derecha"""
        left_X = lista[0] * 300
        left_Y = lista[1] * 300
        right_X = lista[2] * 300
        right_Y = lista[5] * 300
        return left_X, left_Y, right_X, right_Y

    def coord(self, list_a):
        """Function to convert bb from Azure to points with width and height and from inch to pixel"""
        if list_a == "":
            value_X = 0
            value_Y = 0
            width = 0
            height = 0
        else:
            value_X = list_a[0]
            value_Y = list_a[1]
            width = list_a[2] - list_a[0]
            height = list_a[5] - list_a[1]
        return value_X, value_Y, width, height

    def coordbox(self, tabla):
        """Me esta dando problemas con los empty, pasa el min value pero en el max queda el - y crayea."""
        lista = [
            "Value_Canvas_Region_X",
            "Value_Canvas_Region_Y",
            "Value_Canvas_Region_Width",
            "Value_Canvas_Region_Height",
        ]
        for i in lista:
            if tabla[i].empty == True:  # or (tabla[i] == ''):
                value_X = 0
                value_Y = 0
                value_width = 0
                value_height = 0
            else:
                value_X = min(tabla["Value_Canvas_Region_X"], default="EMPTY")
                value_Y = min(tabla["Value_Canvas_Region_Y"], default="EMPTY")
                value_width = (
                        max(
                            tabla["Value_Canvas_Region_X"]
                            + tabla["Value_Canvas_Region_Width"],
                            default="EMPTY",
                        )
                        - value_X
                )
                value_height = (
                        max(
                            tabla["Value_Canvas_Region_Y"]
                            + tabla["Value_Canvas_Region_Height"],
                            default="EMPTY",
                        )
                        - value_Y
                )
        return value_X, value_Y, value_width, value_height


class Metrics:
    def __init__(self):
        self.ImT = ImageTransformation()

    # IoU Function
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = round(interArea / float(boxAArea + boxBArea - interArea), 3)
        # return the intersection over union value
        return iou

    def find_iou_metric(self, name_json, df, gip_form, label_true, label_predicted):
        iou1 = 0
        iou2 = 0
        if name_json.endswith(".json"):
            dl = df[(df["File_Name"] == name_json[:-5] + ".pdf")]
            dl = dl[dl["Map_Name"].str.contains(label_true)]
            dl = pd.DataFrame(dl)
            dr = gip_form[(gip_form["File_Name"] == name_json[:-5] + ".pdf")]
            dr = pd.DataFrame(dr).reset_index()
            bb_cordl = self.ImT.coordbox(dl)
            BoxA = (
                bb_cordl[0],
                bb_cordl[1],
                bb_cordl[0] + bb_cordl[2],
                bb_cordl[1] + bb_cordl[3],
            )
            # GT_bb.append(BoxA)
            if dr.empty:
                BoxB = 0, 0, 0, 0
                textB = ""
            else:
                bb_cordr = self.ImT.coord(dr[label_predicted][0])
                BoxB = (
                    bb_cordr[0],
                    bb_cordr[1],
                    bb_cordr[0] + bb_cordr[2],
                    bb_cordr[1] + bb_cordr[3],
                )
                textB = dr[label_predicted[:-3]][0]
                # MSFR_bb.append(BoxB)
            iou1 = self.bb_intersection_over_union(BoxA, BoxB)

            IoUis, textA = [], []
            for i in dl.index:
                boxAi = (
                    dl["Value_Canvas_Region_X"][i],
                    dl["Value_Canvas_Region_Y"][i],
                    dl["Value_Canvas_Region_X"][i] + dl["Value_Canvas_Region_Width"][i],
                    dl["Value_Canvas_Region_Y"][i]
                    + dl["Value_Canvas_Region_Height"][i],
                )
                IoUi = self.bb_intersection_over_union(boxAi, BoxB)
                IoUis.append(IoUi)
                textA.append(dl['Map_Result_Text'][i])
            iou2 = round(sum(IoUis), 3)
            textA = ' '.join(textA)
            if dl.empty:
                iou1 = np.nan
                iou2 = np.nan
        return iou1, iou2, BoxA, BoxB, textA, textB

    def get_iou_table(self, json_dir, df, GIP_form):
        annotated_labels = ['vendor_address',
                            'vendor_name',
                            'issued_date',
                            'invoice_number',

                            # 'item_descrip',
                            # 'item_t',
                            'total_amount',
                            # 'Item_unit_c',
                            # 'item_unit_v',
                            'net_amount',
                            'tax_amount',
                            'buyer_name',
                            'buyer_address'
                            ]
        gip_labels = ['supplier_address_bb',
                      'supplier_name_bb',
                      'invoice_date_bb',
                      'invoice_id_bb',
                      'total_amount_bb',
                      'net_amount_bb',
                      'total_tax_amount_bb',
                      'receiver_name_bb',
                      'receiver_address_bb',
                      # 'CustomerAddressRecipient',
                      # 'Items', 'ServiceEndDate',
                      # 'ServiceStartDate',  'CustomerId',
                      # 'VendorAddressRecipient', 'AmountDue', 'PreviousUnpaidBalance',
                      # 'PurchaseOrder', 'BillingAddress', 'BillingAddressRecipient', 'DueDate',
                      # 'RemittanceAddress',
                      # 'RemittanceAddressRecipient', 'ShippingAddress',
                      # 'ShippingAddressRecipient', 'ServiceAddress',
                      # 'ServiceAddressRecipient'
                      ]
        # result_labels = ['File_name', 'MerchantName_bb', 'MerchantAddress_bb', 'Total_bb', 'TransactionDate_bb', 'Tax_bb', 'Subtotal_bb']
        json_files = os.listdir(json_dir)
        if ".DS_Store" in json_files:
            json_files.remove(".DS_Store")
        # print(json_files)
        IoU = pd.DataFrame({"File_name": json_files})
        IoU['File_name'] = IoU['File_name'].str.replace('.json', '.pdf')
        # print("aca",IoU)
        # print(msfr_labels,annotated_labels)
        # for json_file in json_files:
        for gip_label, true_label in zip(gip_labels, annotated_labels):
            IoU1 = []
            IoU2 = []
            Box_GT_bb = []
            Box_GIP_bb = []
            textsA, textsB = [], []
            # for ms_label, true_label in zip(msfr_labels, annotated_labels):
            for json_file in json_files:
                if json_file == ".DS_Store":
                    continue
                iou_1, iou_2, BoxA, BoxB, textA, textB = self.find_iou_metric(json_file, df, GIP_form,
                                                                              true_label, gip_label)
                # print("-->",iou_1,iou_2,BoxA,BoxB)
                # input('stop')
                IoU1.append(iou_1)
                IoU2.append(iou_2)
                Box_GT_bb.append(BoxA)
                Box_GIP_bb.append(BoxB)
                textsA.append(textA)
                textsB.append(textB)
            col1 = true_label + '_iou1'
            col2 = true_label + '_iou2'
            col3 = true_label + '_GT_bb'
            col4 = true_label + '_GIP_bb'
            col5 = true_label + '_GT_text'
            col6 = true_label + '_GIP_text'
            IoU[col1] = IoU1
            IoU[col2] = IoU2
            IoU[col3] = Box_GT_bb
            IoU[col4] = Box_GIP_bb
            IoU[col5] = textsA
            IoU[col6] = textsB
            # IoU[true_label + "_GT_bb"] = [BoxA]
            # IoU[true_label + "_MS_bb"] = [BoxB]
            # IoU[true_label + "_iou1"] = [iou_1]
            # IoU[true_label + "_iou2"] = [iou_2]
            print("---->eso:", IoU)

            print(gip_label, true_label)
        return IoU

    # Chequear estas dos funciones para ver si funcionan!

    def get_classical_metrics(self, table, col_IoU1, col_GIP_Form, col_GT):
        # True positives
        tp = len(table[table[col_IoU1] >= 0.51])

        # Total predicted
        mask = (table[col_GIP_Form].notna()) & (table[col_GIP_Form] != (0, 0, 0, 0))
        tpre = len(table[mask])

        # Total positives in GT
        mask = (table[col_GT].notna()) & (table[col_GT] != (0, 0, 0, 0))
        tpgt = len(table[mask])

        # Precision
        precision = tp / (tpre)

        # Recall
        recall = tp / (tpgt)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1, tpgt, tp

    def get_text_metrics(self, table, col_GT_text, col_GIP_text):
        def lcs(y_true: str, y_pred: str):
            tp, fp, fn = lc_subsequence(y_true, y_pred)
            eps = 1e-10
            prec, rec = tp / (tp + fp + eps), tp / (tp + fn + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            tpgt = tp + fn
            return prec, rec, f1, tpgt

        pattern = re.compile('[\W_]+')
        df = pd.DataFrame([])
        table[f"{col_GT_text}_alpha"] = table[col_GT_text].apply(lambda x: pattern.sub('', x)).str.strip().str.lower()
        table[f"{col_GIP_text}_alpha"] = table[col_GIP_text].apply(lambda x: pattern.sub('', x)).str.strip().str.lower()
        # table = table[(table[f"{col_GT_text}_alpha"].str.len() > 0)] # & (table[f"{col_GIP_text}_alpha"].str.len() > 0)]

        # table[f"{col_GT_text}_alsort"] = table[f"{col_GT_text}_alpha"].map(lambda x: ' '.join(sorted(x.split(' '))))
        # table[f"{col_GIP_text}_alsort"] = table[f"{col_GIP_text}_alpha"].map(lambda x: ' '.join(sorted(x.split(' '))))

        # df['direct'] = (table[col_GT_text] == table[col_GIP_text]).astype(float)
        # df['direct_alpha'] = (table[f'{col_GT_text}_alpha'] == table[f'{col_GIP_text}_alpha']).astype(float)
        # df['direct_alsort'] = (table[f'{col_GT_text}_alsort'] == table[f'{col_GIP_text}_alsort']).astype(float)

        df['levenshtein'] = table.apply(lambda x: levenshtein_distance(x[col_GT_text], x[col_GIP_text], True), axis=1)
        df['levenshtein_alpha'] = table.apply(lambda x:
                                              levenshtein_distance(x[f'{col_GT_text}_alpha'],
                                                                   x[f'{col_GIP_text}_alpha'], True),
                                              axis=1)
        # df['levenshtein_alsort'] = table.apply(lambda x:
        #                                        levenshtein_distance(x[f'{col_GT_text}_alsort'], x[f'{col_GIP_text}_alsort'], True),
        #                                        axis=1)
        df[['lcs_prec', 'lcs_rec', 'lcs_f1', 'lcs_support']] = \
            table.apply(lambda x: lcs(x[f'{col_GT_text}_alpha'], x[f'{col_GIP_text}_alpha']),
                        axis=1, result_type="expand")
        support = df['lcs_support'].sum()
        df.drop(['lcs_support'], axis=1, inplace=True)
        return df.mean(axis=0), support

    def get_prediction_table(self, table):
        annotated_labels = [
            'vendor_address',
            'vendor_name',
            'issued_date',
            'invoice_number',

            # 'item_descrip',
            # 'item_t',
            'total_amount',
            # 'Item_unit_c',
            # 'item_unit_v',
            'net_amount',
            'tax_amount',
            'buyer_name',
            'buyer_address'
        ]
        prediction_table = pd.DataFrame({("", "Detected_labels"): annotated_labels})
        Precision = []
        Recall = []
        F1_Score = []
        Total_positives_GT = []
        True_positives = []
        # txt_direct, txt_direct_alpha = [], []
        txt_lvs, txt_lvs_alpha = [], []
        # txt_lvs_alsort = []
        txt_lcs_prec, txt_lcs_rec, txt_lcs_f1, txt_lcs_support = [], [], [], []
        for true_label in annotated_labels:
            col_IoU1 = true_label + "_iou1"
            col_GIP_Form = true_label + "_GIP_bb"
            col_GT = true_label + "_GT_bb"
            col_GT_text = true_label + "_GT_text"
            col_GIP_text = true_label + "_GIP_text"
            precision, recall, f1, tpgt, tp = self.get_classical_metrics(table, col_IoU1, col_GIP_Form, col_GT)
            text_scores, lcs_support = self.get_text_metrics(table, col_GT_text, col_GIP_text)

            # print("prec=",precision)
            Precision.append(precision)
            print("prec=", Precision)
            Recall.append(recall)
            print("recall=", Recall)
            F1_Score.append(f1)
            print("f1-s=", F1_Score)
            Total_positives_GT.append(tpgt)
            print("tot_pos=", Total_positives_GT)
            True_positives.append(tp)
            print("tru_pos=", True_positives)
            # txt_direct.append(text_scores['direct'][0])
            # txt_direct_alpha.append(text_scores['direct_alpha'][0])
            txt_lvs.append(text_scores['levenshtein'])
            txt_lvs_alpha.append(text_scores['levenshtein_alpha'])
            # txt_lvs_alsort.append(text_scores['levenshtein_alsort'])
            txt_lcs_prec.append(text_scores['lcs_prec'])
            txt_lcs_rec.append(text_scores['lcs_rec'])
            txt_lcs_f1.append(text_scores['lcs_f1'])
            txt_lcs_support.append(lcs_support)

        print(prediction_table)
        prediction_table[("IOU", "Precision")] = Precision
        prediction_table[("IOU", "Recall")] = Recall
        prediction_table[("IOU", "F1")] = F1_Score
        prediction_table[("IOU", "Support")] = Total_positives_GT
        # prediction_table[("IOU", "TP")] = True_positives

        # prediction_table["txt_direct"] = txt_direct
        # prediction_table["txt_direct_alpha"] = txt_direct_alpha
        prediction_table[("Levenshtein", "norm")] = txt_lvs
        prediction_table[("Levenshtein", "alpha_norm")] = txt_lvs_alpha
        # prediction_table["txt_lvs_alpha_sorted_norm"] = txt_lvs_alsort
        prediction_table[("LC Subsequence", "Precision")] = txt_lcs_prec
        prediction_table[("LC Subsequence", "Recall")] = txt_lcs_rec
        prediction_table[("LC Subsequence", "F1")] = txt_lcs_f1
        prediction_table[("LC Subsequence", "Support")] = txt_lcs_support
        print(prediction_table)
        return prediction_table
