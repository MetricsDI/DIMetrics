The project implements the continuous recall and precision for all fields for line items on outputs of PSL pipeline. The assignment of the ground truth to the predicted line item is essentially one to one mapping achieved via [hungarian matching algorithm](https://brilliant.org/wiki/hungarian-matching/). The open source [scipy package implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) of the hungarian matching algorithm is used in the code.

The documentation of the 4 fields and 7 fields continuos recall and precision for all fields is on the confluence page [PSL Line item grouping and metric](https://globalinnovation.atlassian.net/wiki/spaces/AILAB/pages/1172340791/PSL+Line+item+grouping+and+metrics)

### Usage:

`python driver.py`

This file generated the metrics for line item on outputs of PSL pipeline.

**Input:-** sample_file.csv in the Inputs folder (sample output of PSL pipeline)

Mandatory columns in the file are
* Map_Name_Original -> Ground truth field label
* Map_Name_Updated  -> Predicted field label (by the classifier)
* lineitem_original -> Ground truth line item group ID
* lineitem_number_Updated -> Predicted line item group ID (by the grouping model, e.g. PSL)
* Map_Result_Text -> Content of the bounding box (each bounding box has a field label)
* fileId
* PageNumber

The `METRIC_FLAG` in the driver.py dictates whether to compute and publish 4 field, 7 fields or both metrics.
* 1 -> 7 field continous metric
* 2 -> 4 field continous metric
* 3 -> both continous metric

**Outputs:-** 2 csv files
* continuous_metrics_7_fields: contains the continuous metrics(recall and precision) for 7 fields
* continuous_metrics_4_fields: contains the continuous metrics(recall and precision) for 4 fields

**The following four fields are considered both for 4 and 7 field metrics evaluation**
* main|||item_description||line_items
* main|||item_total||line_items
* main|||Item_unit_count||line_items
* main|||item_unit_value||line_items

**The additional 3 fields in the 7 fields metric are**
* main|||product_number||line_items
* main|||item_unit_vat_rate||line_items
* main|||line_item_number||line_items

`LineItemMetrics.py`

This is a class file with all the functions required for generating the metrics on line items. The class needs to be created by calling the constructor on dataframe grouped by fileId and PageNumber columns in driver.py.

The `METRIC_FLAG` in the driver.py dictates whether to compute and publish 4 field, 7 fields or both metrics.

### Requirements:
* Python 3.x
* Pandas
* numpy
* scipy 1.4.1