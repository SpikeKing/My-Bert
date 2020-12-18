#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 16.12.20
"""

import cv2
import copy

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR
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
        ori_img = cv2.addWeighted(ori_img, 0.6, img_copy, 0.2, 0)

        # 绘制方向和序号
        pre_point, next_point = None, None
        pre_color = None
        for idx, box in enumerate(box_list):
            point = ((box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2)
            pre_point = point
            if pre_point and next_point:  # 绘制箭头
                cv2.arrowedLine(ori_img, next_point, pre_point, pre_color, thickness=5,
                                line_type=cv2.LINE_4, shift=0, tipLength=0.05)
            next_point = point
            pre_color = color_list[idx]
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
        angle = data_dict['angle']

        is_ok, img_bgr = download_url_img(url)
        # show_img_bgr(img_bgr)

        # color_list = generate_colors(word_num)
        box_list = []
        word_list = []

        for words_data in words_info:
            # print('[Info] words_data: {}'.format(words_data))
            word = words_data["word"]
            # print('[Info] word: {}'.format(word))
            pos = words_data["pos"]
            # print('[Info] pos: {}'.format(pos))
            prob = words_data["prob"]
            if prob == 0:
                continue
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

        img_bgr = rotate_img_for_4angle(img_bgr, angle)

        ori_img = self.draw_box_sequence(img_bgr, box_list, is_show=True)
        return ori_img, boxes_str, angle

    def process_cases(self):
        cases_file = os.path.join(DATA_DIR, 'problem_seg_1216_1000.txt')

        out_dir = os.path.join(DATA_DIR, 'problem_seg_1216_1000_out_v2')
        mkdir_if_not_exist(out_dir)
        out_info_file = os.path.join(DATA_DIR, 'problem_seg_1216_1000.out.txt')

        data_lines = read_file(cases_file)
        print('[Info] 行数: {}'.format(len(data_lines)))

        for data_line in data_lines:
            print('-' * 50)
            try:
                p_id, url_raw = data_line.split(',')
                url = url_raw.split("?")[0]
                name = url.split('/')[-1]
                out_path = os.path.join(out_dir, name)
                ori_img, boxes_str, angle = self.process_url(url)
            except Exception as e:
                print('[Info] error: {}'.format(data_line))
                continue
            cv2.imwrite(out_path, ori_img)
            write_line(out_info_file, "{},{},{}".format(url, angle, boxes_str))
        print('[Info] 处理完成: {}'.format(out_info_file))

    def process_case(self):
        url = "https://img.alicdn.com/imgextra/i2/6000000001271/O1CN01bydE601LGA2N46fn2_!!6000000001271-0-quark.jpg?width=805&amp;height=1367"
        # url = "https://gk.sm.cn/souti_jueying/2f42dd2579e681bfe87e7a2e55afca2c.png"
        self.process_url(url)


def main():
    ors = OcrResults()
    # ors.process_cases()
    ors.process_case()


if __name__ == '__main__':
    main()