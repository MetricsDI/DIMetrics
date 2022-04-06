import glob

import time

from dime.core import *
from dime.utils import *


def main(GT_JSON_DIR, GT_Line_DIR, PRED_XML_DIR, PRED_JSON_DIR, cord2deep, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    final_df, file_num = get_merged_df_DeepCPCFG(PRED_XML_DIR, GT_JSON_DIR, True, cord2deep,
                                                 save_dir=os.path.join(save_dir, "DeepCPCFG_cord_xml2csv"))
    print(f"Processing {file_num} files...")

    iou_metrics_df, iou_mean_df = get_iou_metrics_df(final_df)

    pred_hed_lists = []
    gt_hed_lists = []
    hed_file_count = 0
    for file in glob.glob(os.path.join(PRED_JSON_DIR, "*.json")):
        file_name = os.path.basename(file)
        if os.path.exists(os.path.join(GT_Line_DIR, file_name)):
            pred_data = process_gt_lineitems_json(file)
            pred_hed_lists.append(pred_data)

            gt_data = process_gt_lineitems_json(os.path.join(GT_Line_DIR, file_name))
            gt_hed_lists.append(gt_data)
            hed_file_count += 1
    print(f"There are {hed_file_count} files used to calculate HED/UHED metrics.")
    doc_hed_mean_df = get_hed_document(gt_hed_lists, pred_hed_lists, unorder=True)
    print(doc_hed_mean_df)
    label_df, mean_df = get_hed_label(gt_hed_lists, pred_hed_lists, unorder=True)
    text_df = hedLists2textDf(gt_hed_lists, pred_hed_lists)
    lcs_df = get_text_metrics(text_df)
    lcs_mean_df = get_weighted_mean(lcs_df, 'lcs', 'count')

    with pd.ExcelWriter(os.path.join(save_dir, 'DeepCPCFG_cord_metrics.xlsx')) as writer:
        iou_metrics_df.to_excel(writer, sheet_name='IoU_per_label')
        iou_mean_df.to_excel(writer, sheet_name='IoU_per_doc')
        lcs_df.to_excel(writer, sheet_name='LCS_per_label')
        lcs_mean_df.to_excel(writer, sheet_name='LCS_per_doc')
        doc_hed_mean_df.to_excel(writer, sheet_name='HED_per_doc')
        label_df.to_excel(writer, sheet_name='HED_per_label')
        mean_df.to_excel(writer, sheet_name='HED_mean_label')
        writer.save()

    print(f"Saving all metcis in {save_dir}...")


if __name__ == "__main__":
    GT_JSON_DIR = '../../sample_data/cord_data/'
    GT_Line_DIR = '../../sample_data/cord_lineitems/'
    PRED_XML_DIR = '../../sample_data/model_preds/deepcpcfg/cord/xml'
    PRED_JSON_DIR = '../../sample_data/model_preds/deepcpcfg/cord/json/'

    cord2deep = {
        'total.total_price': 'TotalPrice',
        'menu.nm': 'MenuNm',
        'menu.unitprice': 'MenuUnitprice',
        'menu.cnt': 'MenuCnt',
        'menu.price': 'MenuPrice'}
    save_dir = "../outputs"
    start = time.time()
    main(GT_JSON_DIR, GT_Line_DIR, PRED_XML_DIR, PRED_JSON_DIR, cord2deep, save_dir)
    print(f"Evaluation was done in {time.time() - start} seconds.")
