"""
Created on Thu Aug 06 11:13:03 2020

@author: divya.prakash1
@edit jdegange 09/21
"""
import numpy as np
import pandas as pd
import warnings
import sys
from scipy.optimize import linear_sum_assignment

warnings.simplefilter(action='ignore')

class LineItemMetric:
    def __init__(self, df_groups):
        self.original_line_item_dict = {}
        self.updated_line_item_dict = {}
        self.true_positive_match = {}
        self.true_positives = {}
        self.false_negatives = {}
        self.recall_values = {}
        self.precision_values = {}
        self.df_groups = df_groups
        self.t_o_l_i = 0
        self.t_u_l_i = 0

        self.recall = {'fileId_pg':[]}
        self.precision = {'fileId_pg':[]}
        self.recall['line_item'] = []
        self.precision['line_item'] = []

        self.class_list = ['main|||item_description||line_items',
                            'main|||item_total||line_items',
                            'main|||Item_unit_count||line_items',
                            'main|||item_unit_value||line_items',
                            'main|||product_number||line_items',
                            'main|||item_unit_vat_rate||line_items',
                            'main|||line_item_number||line_items']
    
    def _get_original_lineitem_groups(self, groups):
        local_file_item_dict = {}  
        i = 1
        for name, group in groups:
            g = pd.DataFrame(group)
            n = list(g['fileId'])[0] + "_" + str(list(g['PageNumber'])[0])
            if(n in self.original_line_item_dict.keys()):
                self.original_line_item_dict[n].append(g)
            else:
                self.original_line_item_dict[n] = [g]

            local_file_item_dict[n + "_" + str(i)] = g
            i = i + 1
                
        return local_file_item_dict

    def _get_updated_lineitem_groups(self, groups):
        local_file_item_dict = {}    
        i = 1
        for name, group in groups:
            g = pd.DataFrame(group)
            n = list(g['fileId'])[0] + "_" + str(list(g['PageNumber'])[0])
            if(n in self.updated_line_item_dict.keys()):
                self.updated_line_item_dict[n].append(g)
            else:
                self.updated_line_item_dict[n] = [g]

            local_file_item_dict[n + "_" + str(i)] = g
            i = i + 1
        
        return local_file_item_dict

    def _get_partial_bbox_match(self, originalDF, updatedDF):
        category_list = ['main|||item_total||line_items',
                'main|||Item_unit_count||line_items',
                'main|||item_unit_value||line_items',
                'main|||product_number||line_items',
                'main|||item_unit_vat_rate||line_items',
                'main|||line_item_number||line_items']
        
        #if the original is null then recall and precision are not defined
        idx_original = originalDF['idx']
        idx_updated  = updatedDF['idx']
        idx_intersect = list(set(idx_original).intersection(set(idx_updated)))
        
        #if nothing intersets return recall and precision as zero
        if(len(idx_intersect) == 0):
            return [0.0], [0.0]
        
        intersectDF = originalDF.loc[idx_intersect, :]

        li_recall = []
        li_precision = []
        
        #for each of the category calculate the precision and recall values

        for category in category_list:
            #if the category in not present in ground truth the recall is not defined
            if(category not in list(originalDF['Map_Name_Original'])):
                li_recall.append(np.nan)
            
            # if the category is not present in either map name original or map name upated then there
            # can be no overlap for that category
            elif((category not in list(intersectDF['Map_Name_Original'])) or (category not in list(intersectDF['Map_Name_Updated']))):
                li_recall.append(0)
                
            #if there is overlap
            else:
                overlap_bbox = len(intersectDF[(intersectDF['Map_Name_Original'] == category) &
                                                        (intersectDF['Map_Name_Original'] == intersectDF['Map_Name_Updated'])])
                bbox_original = len(originalDF[originalDF['Map_Name_Original']==category])
                li_recall.append(overlap_bbox/bbox_original)
            
            #if the category is not present in predicted the preision is not defined
            if(category not in list(updatedDF['Map_Name_Updated'])):
                li_precision.append(np.nan)
            
            # if the category is not present in either map name original or map name upated then there
            # can be no overlap for that category
            elif((category not in list(intersectDF['Map_Name_Original'])) or (category not in list(intersectDF['Map_Name_Updated']))):
                li_precision.append(0)
                
            #if there is overlap
            else:
                overlap_bbox = len(intersectDF[(intersectDF['Map_Name_Original'] == category) &
                                                        (intersectDF['Map_Name_Original'] == intersectDF['Map_Name_Updated'])])
                bbox_original = len(updatedDF[updatedDF['Map_Name_Updated']==category])
                li_precision.append(overlap_bbox/bbox_original)
                
        return li_recall, li_precision

    
    def _get_partial_match_nchar_description(self, originalDF, updatedDF):
        '''
        Returns the partial match score based on no. of characters in intersection wrt original
        in item description fields.
        '''
        idx_original = originalDF['idx']
        idx_updated  = updatedDF['idx']
        idx_intersect = list(set(idx_original).intersection(set(idx_updated)))
        textRecall = 0
        textPrecision = 0
        if(len(idx_original) == 0):
            textRecall = np.nan
        if(len(idx_updated) == 0):
            textPrecision = np.nan
        
        if(len(idx_intersect) == 0):
            return textRecall, textPrecision
        else:
            mylen = np.vectorize(len)
            
            #below line to take care of none edge case
            originalDF['Map_Result_Text'] = originalDF['Map_Result_Text'].apply(lambda x: '' if x==None else str(x))
            updatedDF['Map_Result_Text'] = updatedDF['Map_Result_Text'].apply(lambda x: '' if x==None else str(x))
            
            intersect_text = np.sum(mylen(originalDF.loc[idx_intersect, 'Map_Result_Text'].values))
            original_text = np.sum(mylen(originalDF['Map_Result_Text'].values))
            updated_text  = np.sum(mylen(updatedDF['Map_Result_Text'].values))
        
            textRecall = intersect_text/original_text
            textPrecision = intersect_text/updated_text
        
        return textRecall, textPrecision
    
    def _get_partial_match_line_item(self, original_dict, updated_dict):
        global true_positive_match
        global true_positive_recall
        true_positive = {}
        false_negative = {}
        recall_val = {}
        precision_val = {}
        
        indexes = list(original_dict.keys())
        columns = list(updated_dict.keys())

        #handle the edge case when there are no predictions -> all originals are false negatives 
        if(len(columns) == 0):
            for k in indexes:
                false_negative[k] = original_dict[k]
                recall_val[k] = 0
            return true_positive, false_negative, recall_val, precision_val

        if(len(indexes) == 0):
            return true_positive, false_negative, recall_val, precision_val

        #create an empty metric dataframe with rows as original line item and
        #cols as predicted line items
        metricDF = pd.DataFrame(index=indexes, columns=columns, dtype='object')

        #for each original line item compare it with all the predicted line items
        for k, v in original_dict.items():
            
            originalDF = original_dict[k]
            #total no. of category in original
            totalCategory = len(set(originalDF['Map_Name_Original']))

            originalDF_with_item_description = originalDF[originalDF['Map_Name_Original'] =='main|||item_description||line_items']
            originalDF_without_item_description = originalDF[originalDF['Map_Name_Original'] != 'main|||item_description||line_items']
            
            #comparing with predicted line item
            for k1, v1 in updated_dict.items():
                updatedDF = updated_dict[k1]
                updatedDF_with_item_description = updatedDF[updatedDF['Map_Name_Updated'] =='main|||item_description||line_items']
                updatedDF_without_item_description = updatedDF[updatedDF['Map_Name_Updated'] != 'main|||item_description||line_items']
                
                #get discrete recall and precision for fields without item_description
                recall_list, precision_list = self._get_partial_bbox_match(originalDF_without_item_description, updatedDF_without_item_description) 
                cont_recall, cont_precision = self._get_partial_match_nchar_description(originalDF_with_item_description, updatedDF_with_item_description)
                recall_list.append(cont_recall)
                precision_list.append(cont_precision)
                
                pairRecall = np.nanmean(recall_list)
                pairPrecision = np.nanmean(precision_list)
                # print(k, k1, recall_list, precision_list, cont_recall, cont_precision)
                # print('-------------------------------------------------------------------------')
                metricDF.at[k, k1] = [pairRecall, pairPrecision, totalCategory]

        #create a costMat matrix on which to perform max hungarian matching
        costMat = np.zeros((len(indexes), len(columns)))
        for i in range(len(indexes)):
            for j in  range(len(columns)):
                mf = 10000  #multiplication factor needs to be more thought upon
                costMat[i,j] = mf * metricDF.iloc[i,j][0] + metricDF.iloc[i,j][-1]


        #get the matched row and col indices
        row_ind, col_ind = linear_sum_assignment(costMat)

#        row_ind, col_ind = linear_sum_assignment(costMat,maximize=True)

        #case of matched indices
        for i in range(len(row_ind)):
            original_li = indexes[row_ind[i]]
            predicted_li = columns[col_ind[i]]

            #case of true positve only when the recall is greater than zero
            if(metricDF.at[original_li, predicted_li][0] > 0 ):
                self.true_positive_match[original_li] = predicted_li
                true_positive[original_li] = original_dict[original_li]
                recall_val[original_li] = metricDF.at[original_li, predicted_li][0]
                precision_val[original_li] = metricDF.at[original_li, predicted_li][1]
            
            #otherwise false negative
            else:
                false_negative[original_li] = original_dict[original_li]
                recall_val[original_li] = 0
        
        #non matched indices are straighforwasd false negative
        nonMatched_ind = [i for i in range(len(indexes)) if i not in row_ind]
        for idx in nonMatched_ind:
            original_li = indexes[idx]
            false_negative[original_li] = original_dict[original_li]
            recall_val[original_li] = 0
        
        return true_positive, false_negative, recall_val, precision_val

    def _get_partial_match_metrics(self):
        '''
        gives the partial match metric for the line items for PSL Output
        as mentioned in readme file.
        '''
        for name, group in self.df_groups:
            #Append the file name to the macro metrics
            self.recall['fileId_pg'].append(name)
            self.precision['fileId_pg'].append(name)

            ## remove no_class rows and no_lineitem rows from the dataframe containing updated Map_Name and lineitem_number_updated and group them by lineitem_number_updated
            updated_group = group.loc[(group['Map_Name_Updated'].isin(self.class_list)) & (group['lineitem_number_Updated'] != '')]
            lineitem_groups_updated = updated_group.groupby('lineitem_number_Updated')
            
            ## remove no_class rows and no_lineitem rows from the dataframe cotaining actual Map_Name and lineitem_original and group them by lineitem_original
            original_group = group.loc[(group['Map_Name_Original'].isin(self.class_list)) & (group['lineitem_original'] != 0) & (group['lineitem_original'] != '')]
            lineitem_groups_original = original_group.groupby('lineitem_original')
            
            original_line_dict = self._get_original_lineitem_groups(lineitem_groups_original)
            self.t_o_l_i = self.t_o_l_i + len(original_line_dict)

            updated_line_dict= self._get_updated_lineitem_groups(lineitem_groups_updated)
            self.t_u_l_i = self.t_u_l_i + len(updated_line_dict)

            true_p, false_n, recall_val, precision_val = self._get_partial_match_line_item(original_line_dict, updated_line_dict)
            
            
            #calculate and append the macro metrics
            self.recall['line_item'].append(np.nanmean(list(recall_val.values())))

            try:
                curr_precision = np.sum(list(precision_val.values())) / len(updated_line_dict)
                self.precision['line_item'].append(curr_precision)
            except:
                self.precision['line_item'].append(np.nan)
            
            self.true_positives.update(true_p)
            self.false_negatives.update(false_n)
            self.recall_values.update(recall_val)
            self.precision_values.update(precision_val)


        macro_recall = np.nanmean(self.recall['line_item']) #calculates mean ignoring NaNs
        macro_precision = np.nanmean(self.precision['line_item'])

        li_metric_dict = {}
        li_metric_dict['true positives'] = len(self.true_positives)
        li_metric_dict['t_o_l_i'] = self.t_o_l_i
        li_metric_dict['t_u_l_i'] = self.t_u_l_i
        li_metric_dict['Micro_Precision'] = np.sum(list(self.precision_values.values()))/self.t_u_l_i
        li_metric_dict['Micro_Recall'] = np.sum(list(self.recall_values.values()))/self.t_o_l_i
        li_metric_dict['Macro_Precision'] = macro_precision
        li_metric_dict['Macro_Recall'] = macro_recall
        df_li_report = pd.DataFrame(li_metric_dict, index=['value']).transpose()
        
        return df_li_report

    def _get_metrics(self):
        return self._get_partial_match_metrics()