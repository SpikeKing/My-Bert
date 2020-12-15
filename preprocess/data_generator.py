#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 8.12.20
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR


class DataGenerator(object):
    def __init__(self):
        pass

    def process(self):
        print('[Info] 开始处理!')
        random.seed(47)

        file_path = os.path.join(DATA_DIR, 'toutiao_cat_data.txt')

        out_dir = os.path.join(DATA_DIR, 'toutiao_dataset')
        mkdir_if_not_exist(out_dir)

        train_file = os.path.join(out_dir, 'train.tsv')     # 训练集
        test_file = os.path.join(out_dir, 'test.tsv')       # 测试集
        dev_file = os.path.join(out_dir, 'dev.tsv')         # 验证集
        labels_file = os.path.join(out_dir, 'labels.txt')         # 验证集

        create_file(train_file)
        create_file(test_file)
        create_file(dev_file)
        create_file(labels_file)

        data_lines = read_file(file_path)
        out_lines = []
        label_set = set()
        for idx, data_line in enumerate(data_lines):
            items = data_line.split('_!_')
            label = items[2]
            label_set.add(label)
            content = items[3]
            content = content.replace('\0', '')
            if not content:
                print('[Info] error: {} content: {}'.format(idx, content))
                continue
            # print('[Info] label: {}, content: {}'.format(label, content))
            out_line = "{}\t{}".format(label, content)
            out_lines.append(out_line)
            if idx % 10000 == 0:
                print('[Info] idx: {}'.format(idx))

        label_list = sorted(list(label_set))
        write_list_to_file(labels_file, label_list)  # 写入标签

        n_lines = len(out_lines)
        print('[Info] 样本数: {}'.format(n_lines))
        random.shuffle(out_lines)

        cut_10 = n_lines // 10
        train_lines = out_lines[:cut_10*8]
        test_lines = out_lines[cut_10*8:cut_10*9]
        dev_lines = out_lines[cut_10*9:]

        # train: 306248, test: 38281, dev: 38281
        print('[Info] train: {}, test: {}, dev: {}'.format(len(train_lines), len(test_lines), len(dev_lines)))

        # 写入文件
        write_list_to_file(train_file, train_lines)
        write_list_to_file(test_file, test_lines)
        write_list_to_file(dev_file, dev_lines)

        print('[Info] 处理完成: {}'.format(out_dir))


def main():
    dg = DataGenerator()
    dg.process()


if __name__ == '__main__':
    main()
