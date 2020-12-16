#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 16.12.20
"""

import os
import cv2
import sys
import copy

from myutils.project_utils import *
from myutils.cv_utils import *
from x_utils.vpf_utils import get_ocr_vpf_service


class OcrResults(object):
    """
    处理OCR结果
    """
    def __init__(self):
        pass

    @staticmethod
    def parse_pos(pos_list):
        """
        处理点
        """
        point_list = []
        for pos in pos_list:
            x = pos['x']
            y = pos['y']
            point_list.append([x, y])
        return point_list

    @staticmethod
    def draw_box_sequence(img_bgr, box_list, is_show=True):
        """
        绘制矩形列表
        """
        n_box = len(box_list)
        color_list = generate_colors(n_box)  # 随机生成颜色
        ori_img = copy.copy(img_bgr)
        img_copy = copy.copy(img_bgr)

        # 绘制颜色块
        for idx, (box, color) in enumerate(zip(box_list, color_list)):
            rec_arr = np.array(box)
            ori_img = cv2.fillPoly(ori_img, [rec_arr], color_list[idx])
        ori_img = cv2.addWeighted(ori_img, 0.4, img_copy, 0.6, 0)

        # 绘制方向和序号
        pre_point, next_point = None, None
        for idx, box in enumerate(box_list):
            point = ((box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2)
            pre_point = point
            if pre_point and next_point:  # 绘制箭头
                cv2.arrowedLine(ori_img, next_point, pre_point, color_list[idx], thickness=5,
                                line_type=cv2.LINE_4, shift=0, tipLength=0.05)
            next_point = point
            draw_text(ori_img, str(idx), point)  # 绘制序号

        if is_show:
            show_img_bgr(ori_img)
        return ori_img

    def process_url(self, url):
        """
        处理URL
        """
        print('[Info] url: {}'.format(url))
        res_dict = get_ocr_vpf_service(url)
        # print('[Info] res_dict: {}'.format(res_dict))
        data_dict = res_dict['data']['data']
        # print('[Info] data_dict: {}'.format(data_dict))
        word_num = data_dict['wordNum']
        # print('[Info] word_num: {}'.format(word_num))
        words_info = data_dict['wordsInfo']

        is_ok, img_bgr = download_url_img(url)
        # show_img_bgr(img_bgr)

        # color_list = generate_colors(word_num)
        box_list = []
        word_list = []

        box_dict = dict()
        for words_data in words_info:
            # print('[Info] words_data: {}'.format(words_data))
            word = words_data["word"]
            # print('[Info] word: {}'.format(word))
            pos = words_data["pos"]
            # print('[Info] pos: {}'.format(pos))
            word_rec = self.parse_pos(pos)
            # print('[Info] point_list: {}'.format(word_rec))
            box_list.append(word_rec)
            word_list.append(word)
            # point_arr = np.array(point_list)
            # img_tmp = copy.copy(img_bgr)
            # img_tmp = cv2.fillPoly(img_tmp, [point_arr], color_list[0])
            # show_img_bgr(img_tmp)
            # img_out = cv2.addWeighted(img_bgr, 0.6, img_tmp, 0.4, 0)
            # show_img_bgr(img_out)
            # img_out = img_out.clip(0, 255)

        box_dict = {
            "word_list": word_list,
            "box_list": box_list
        }
        boxes_str = json.dumps(box_dict)
        print('[Info] boxes_str: {}'.format(boxes_str))

        self.draw_box_sequence(img_bgr, box_list)

    def process(self):
        url = "https://img.alicdn.com/imgextra/i2/6000000007802/O1CN01q0TxUq27VMiw1yriy_!!6000000007802-0-quark.jpg"
        # url = "https://gk.sm.cn/souti_jueying/2f42dd2579e681bfe87e7a2e55afca2c.png"
        self.process_url(url)


def main():
    ors = OcrResults()
    ors.process()


if __name__ == '__main__':
    main()