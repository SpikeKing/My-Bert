#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 8.12.20
"""

import os
import sys


import tokenization
from myutils.cv_utils import *
from myutils.project_utils import *
from run_classifier import DataProcessor, InputExample


class MyTaskProcessor(DataProcessor):
    """
    Processor for the News data set (GLUE version).
    """

    def __init__(self):
        self.labels = [
            "news_agriculture",
            "news_car",
            "news_culture",
            "news_edu",
            "news_entertainment",
            "news_finance",
            "news_game",
            "news_house",
            "news_military",
            "news_sports",
            "news_story",
            "news_tech",
            "news_travel",
            "news_world",
            "stock",
        ]

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


def main():
    pass


if __name__ == '__main__':
    main()
