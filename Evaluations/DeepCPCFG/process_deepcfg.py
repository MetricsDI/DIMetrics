import argparse
import os
from glob import glob

import ray
import xmltodict
from metrics.longest_common_subsequence import test_lcseq
from parse_json import parse_json


@ray.remote
def json2dict(input_json, verbose=False):
    dict_temp = parse_json(input_json)
    if verbose:
        for key in dict_temp: print(f'{key} --> {dict_temp[key]}')


@ray.remote
def xml2dict(input_xml, verbose=False):
    with open(input_xml) as xml_file:
        dict_temp = xmltodict.parse(xml_file.read())

    if verbose:
        for key in dict_temp: print(f'{key} --> {dict_temp[key]}')

    return dict_temp


if __name__ == '__main__':
    num_cpus = len(os.sched_getaffinity(0))
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_json_dir",
        type=str,
        default="/data/home/djonathan/ai-science-di-metrics/datasets/NJK1_91holdout_julia_2dCKY/data/test_nj/output/json/",
        help="Input directory containing list of json files",
    )

    parser.add_argument(
        "--preds_xml_dir",
        type=str,
        default="/data/home/djonathan/ai-science-di-metrics/datasets/NJK1_91holdout_julia_2dCKY/data/test_nj/output/xml/",
        help="Input directory containing list of json files",
    )

    args = parser.parse_args()

    gt_json_dir_files = glob(args.gt_json_dir + "*.json")

    preds_xml_dir_files = glob(args.preds_xml_dir + "*.xml")

    # get the ground truths with parallel json parse (Lark Earley Parser + Ray)
    gt_dicts = ray.get([json2dict.remote(f, verbose=False) for f in gt_json_dir_files])
    # get xml files and convert them to dicts
    preds_dicts = ray.get([xml2dict.remote(f, verbose=False) for f in preds_xml_dir_files])

    # print(xml_file_dict)
    xml_file_keys, xml_file_text_values = [], []

    # print(gt_dicts[0])
    # print(preds_dicts[0],"\n")

    preds_dict_i = preds_dicts[1]
    for key in preds_dict_i:
        xml_file_dict = preds_dict_i
        fields_dict = {}
        key_list = []
        for key in xml_file_dict:
            print(key, xml_file_dict[key], "\n")
            key_list.append(key)
    print(len(key_list))

    # compare GT json and pred XML

    test_lcseq()

    # preds_dicts

    # metric_1
    print("shutting down ray")

    ray.shutdown()
