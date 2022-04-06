import time

from dime.core import *
from dime.utils import *


def main(GT_DIR, GT_Lineitem_DIR, PRED_DIR, multiplier, LineItemPrefix, field_mapping, gt_lineitems2pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    final_df, file_num, gt_hed_lists, pred_hed_lists = get_merged_df_FR(PRED_DIR, GT_DIR, GT_Lineitem_DIR, False,
                                                                        field_mapping,
                                                                        LineItemPrefix, multiplier, gt_lineitems2pred)
    print(f"Processing {file_num} files...")

    iou_metrics_df, iou_mean_df = get_iou_metrics_df(final_df)

    lcs_df = get_text_metrics(final_df)

    lcs_mean_df = get_weighted_mean(lcs_df, 'lcs', 'count')

    doc_hed_mean_df = get_hed_document(gt_hed_lists, pred_hed_lists, unorder=True)
    print(doc_hed_mean_df)
    label_df, mean_df = get_hed_label(gt_hed_lists, pred_hed_lists, unorder=True)

    with pd.ExcelWriter(os.path.join(save_dir, 'FormRecognizer_cdip_metrics.xlsx')) as writer:
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
    GT_DIR = '../datasets/cdip_data/'
    GT_Lineitem_DIR = '../datasets/cdip_lineitems'
    PRED_DIR = '../datasets/model_preds/FR/cdip/'
    multiplier = 300
    LineItemPrefix = 'Item'
    field_mapping = {
        'main|||vendor_name||primary_fields': 'VendorName',
        'main|||buyer_name||primary_fields': 'CustomerName',
        'main|||invoice_number||primary_fields': 'InvoiceId',
        'main|||total_amount||primary_fields': 'InvoiceTotal',
        'main|||issued_date||primary_fields': 'InvoiceDate',
        'main|||item_description||line_items': 'ItemDescription',
        'main|||item_total||line_items': 'ItemTotal',
        'main|||Item_unit_count||line_items': 'ItemCount',
        'Description': 'ItemDescription',
        'Amount': 'ItemTotal',
        'Quantity': 'ItemCount',
    }
    gt_lineitems2pred = {
        'IssuedDate': 'InvoiceDate',
        'TotalAmount': 'InvoiceTotal',
        'InvoiceNumber': 'InvoiceId',
        'ItemUnitCount': 'ItemCount'
    }
    save_dir = "../outputs"
    start = time.time()
    main(GT_DIR, GT_Lineitem_DIR, PRED_DIR, multiplier, LineItemPrefix, field_mapping, gt_lineitems2pred, save_dir)
    print(f"Evaluation was done in {time.time() - start} seconds.")
