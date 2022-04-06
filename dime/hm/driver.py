import numpy as np
import pandas as pd
import sys

import LineItemMetrics as LIM

def _check_data_columns(df):
    """\
    Checks if all the required columns are present in the data
    """
    required_columns = set(['Map_Name_Original', 'Map_Name_Updated', 'lineitem_number_Updated',\
                            'lineitem_original', 'Map_Result_Text','fileId', 'PageNumber', 'idx'])
    df_columns = set(df.columns)
    intersections = required_columns.intersection(df_columns)

    if intersections == required_columns:
        print("Required data check passed.")
        return True
    print("ERROR: Missing data: {}".format(required_columns - intersections))
    sys.exit()
    return None

def keepLineItemLabels(x):
    line_item_class_list = ['No class',
              'main|||item_description||line_items',
              'main|||item_total||line_items',
              'main|||Item_unit_count||line_items',
              'main|||item_unit_value||line_items']
    if x in line_item_class_list:
        return x
    else:
        return 'No class'

def main():
    #metric flag determines which metrics to calculate
    # 1 -> 7 field partial metric
    # 2 -> 4 field partial metric
    # 3 -> both partial metric
    METRIC_FLAG = 2

    input_file = 'Inputs/sample_file.csv'
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.pkl'):
        df = pd.read_pickle(input_file)
    else:
        print('Input not supported')
        return None
    
    df.rename(columns = {'Original_Map_Label':'Map_Name_Original','Grouping':'lineitem_original'}, inplace = True)

    df['idx'] = df.index
    _check_data_columns(df)
    df_groups = df.groupby(['fileId', 'PageNumber'])
    num_invoices = len(df_groups)
    print(f"\ntotal number of pages: {num_invoices}\n ")

    if(METRIC_FLAG == 1 or METRIC_FLAG == 3):
        lim1 = LIM.LineItemMetric(df_groups = df_groups)

        print('Calculating Line Item Metric Reports for 7 fields...')

        partialMatch_report = lim1._get_metrics()
        partialMatch_report.loc['precision (no description)'] = partialMatch_report.loc['true positives']/partialMatch_report.loc['t_u_l_i']
        partialMatch_report.loc['recall (no description)'] = partialMatch_report.loc['true positives']/partialMatch_report.loc['t_o_l_i']

        partialMatch_report.to_csv('./Results/continuous_metrics_7_fields.csv')

        print("Line Item Metric Reports for 7 fields published.")
        print('----------------------------------------------------\n')
    

    if(METRIC_FLAG == 2 or METRIC_FLAG == 3):
        #convert the df to 4 fields and regenerate the metlsrics
        df['Map_Name_Original'] = df['Map_Name_Original'].apply(lambda x: keepLineItemLabels(x))
        #df['Map_Name_Predicted'] = df['Map_Name_Predicted'].apply(lambda x: keepLineItemLabels(x))
        df['Map_Name_Updated'] = df['Map_Name_Updated'].apply(lambda x: keepLineItemLabels(x))

        df_groups = df.groupby(['fileId', 'PageNumber'])
        lim2 = LIM.LineItemMetric(df_groups = df_groups)
        
        print('Calculating Line Item Metric Reports for 4 fields...')
        partialMatch_report = lim2._get_metrics()
        partialMatch_report.loc['precision (no description)'] = partialMatch_report.loc['true positives']/partialMatch_report.loc['t_u_l_i']
        partialMatch_report.loc['recall (no description)'] = partialMatch_report.loc['true positives']/partialMatch_report.loc['t_o_l_i']

        partialMatch_report.to_csv('./Results/continuous_metrics_4_fields.csv')
        print("Line Item Metric Reports for 4 fields published.\n")
    print(partialMatch_report.head())

    return None

if __name__ == '__main__':
    main()
    print("END.")