#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 8.12.20
"""
import os

from root_dir import DATA_DIR

import csv
import tensorflow.compat.v1 as tf


def _read_tsv(input_file, quotechar=None):
    print('[Info] _read_tsv - input_file: {}'.format(input_file))
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        try:
            x_line = None
            for line in reader:
                x_line = line
                lines.append(line)
        except Exception as e:
            print(e)
            print(x_line)
        return lines

def main():
    file_path = os.path.join(DATA_DIR, 'toutiao_dataset', 'train.tsv')
    _read_tsv(file_path)


if __name__ == "__main__":
    main()
