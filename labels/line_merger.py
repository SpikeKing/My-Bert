#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 20.12.20
"""
import os
import sys
import json

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class LineMerger(object):
    def __init__(self):
        pass

    def box_list_2_one(self, box_list):
        """
        多个框融合成1个框
        """
        x_list, y_list = [], []
        for rec in box_list:
            for i_rec in rec:
                x, y = i_rec
                x_list.append(x)
                y_list.append(y)
        x_min, y_min = min(x_list), min(y_list)
        x_max, y_max = max(x_list), max(y_list)
        return [x_min, y_min, x_max, y_max]

    def process(self):
        data_dir = os.path.join(DATA_DIR, 'labeled_data_out_20201218223625')
        paths_list, names_list = traverse_dir_files(data_dir)
        print('[Info] 文件数: {}'.format(len(paths_list)))

        data_lines_all = []
        for path, name in zip(paths_list, names_list):
            data_lines = read_file(path)
            data_lines_all += data_lines

        print('[Info] 文本行数: {}'.format(len(data_lines_all)))
        """
        out_dict = {
            "url": url,
            "p_box_list": p_box_list,
            "item_list": item_list,
            "labeled_url": labeled_url
        }
        """

        line_str_list = []
        for idx, data_line in enumerate(data_lines_all):
            data_dict = json.loads(data_line)
            item_list = data_dict["item_list"]
            img_url = data_dict["url"]
            print('[Info] img_url: {}'.format(img_url))
            is_ok, img_bgr = download_url_img(img_url)
            for item in item_list:
                box_list = item["box_list"]
                word_list = item["word_list"]
                print(box_list)
                print(word_list)
                p_box = self.box_list_2_one(box_list)
                img_patch = get_patch(img_bgr, p_box)
                show_img_bgr(img_patch)

            if idx == 10:
                break

def main():
    lm = LineMerger()
    lm.process()


if __name__ == '__main__':
    main()
