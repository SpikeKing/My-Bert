#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 17.12.20
"""
import os
import copy
import cv2
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from x_utils.oss_utils import save_img_2_oss
from x_utils.vpf_utils import get_ocr_vpf_service
from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class DataPrelabeled(object):
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

    def get_boxes_from_items(self, label_str):
        print('[Info] label_str: {}'.format(label_str))
        label_list = json.loads(label_str)
        coord_list = label_list[0]
        box_list =[]
        for coord in coord_list:
            box = coord["coord"]
            box_list.append(box)
            print('[Info] box: {}'.format(box))
        return box_list

    @staticmethod
    def process_url(url):
        """
        处理URL
        """
        print('[Info] process url: {}'.format(url))
        res_dict = get_ocr_vpf_service(url)
        # print('[Info] res_dict: {}'.format(res_dict))
        data_dict = res_dict['data']['data']
        # print('[Info] data_dict: {}'.format(data_dict))
        word_num = data_dict['wordNum']
        # print('[Info] word_num: {}'.format(word_num))
        words_info = data_dict['wordsInfo']
        angle = data_dict['angle']

        content = data_dict['content']

        # color_list = generate_colors(word_num)
        box_list, word_list = [], []

        for words_data in words_info:
            # print('[Info] words_data: {}'.format(words_data))
            word = words_data["word"]
            rec_classify = words_data["recClassify"]
            # print('[Info] word: {}'.format(word))
            pos = words_data["pos"]
            # print('[Info] pos: {}'.format(pos))
            prob = words_data["prob"]
            if rec_classify != 0:
                word = "$${}$$".format(word)
            if prob == 0:
                continue
            word_rec = DataPrelabeled.parse_pos(pos)
            # print('[Info] point_list: {}'.format(word_rec))
            box_list.append(word_rec)
            word_list.append(word)

        return box_list, word_list, angle, content

    def check_inside_box(self, p_box, box, oh=0, ow=0):
        x_min, y_min, x_max, y_max = p_box
        is_inside = True
        for pt in box:
            if x_min - ow < pt[0] < x_max + ow and y_min - oh < pt[1] < y_max + oh:
                continue
            else:
                is_inside = False
        return is_inside

    def filter_boxes(self, img_bgr, p_box, box_list, word_list, angle, content):
        draw_box(img_bgr, p_box, is_show=False, is_new=False)
        h, w = p_box[3] - p_box[1], p_box[2] - p_box[0]
        oh, ow = int(h * 0.05), int(w * 0.05)
        new_box_list, new_word_list = [], []
        for box, word in zip(box_list, word_list):
            word = unicode_str(word)
            is_inside = self.check_inside_box(p_box, box, oh, ow)
            if is_inside:
                # draw_4p_rec(img_bgr, box, is_show=True, is_new=False)
                new_box_list.append(box)
                new_word_list.append(word)
        # new_content = " ".join(new_word_list)
        # new_content = new_content.replace("$$", "")
        # print('[Info] new_content: {}'.format(new_content))
        # print('[Info] content: {}'.format(content))
        # is_contain = True if new_content in content else False
        return new_box_list, new_word_list

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

        ori_img = cv2.addWeighted(ori_img, 0.5, img_copy, 0.5, 0)
        ori_img = np.clip(ori_img, 0, 255)

        if is_show:
            show_img_bgr(ori_img)
        return ori_img

    def process_file(self, file_path, out_file):
        print('[Info] file_path: {}'.format(file_path))
        data_lines = read_file(file_path)
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            if idx == 20:
                break
            print('[Info] ' + "-" * 50)
            items = data_line.split(';')
            label_str = items[3]
            url = items[5]
            url = url.split("?")[0]
            img_name = url.split("/")[-1]
            print('[Info] label: {}'.format(label_str))
            print('[Info] url: {}'.format(url))
            try:
                p_box_list = self.get_boxes_from_items(label_str)
            except Exception as e:
                print('[Info] label error: {}, label_str: {}'.format(url, label_str))
                continue
            if not p_box_list:
                continue
            is_ok, img_bgr = download_url_img(url)
            # show_img_bgr(img_bgr)
            # draw_box(img_bgr, p_box, is_show=True)
            box_list, word_list, angle, content = self.process_url(url)
            if angle != 0:
                print('[Info] angle error: {}, angle: {}'.format(url, angle))
                continue
            item_list = []
            for p_box in p_box_list:
                new_box_list, new_word_list = \
                    self.filter_boxes(img_bgr, p_box, box_list, word_list, angle, content)
                img_bgr = DataPrelabeled.draw_box_sequence(img_bgr, new_box_list, is_show=False)
                item_dict = {
                    "box_list": box_list,
                    "word_list": word_list
                }
                item_list.append(item_dict)
            labeled_url = save_img_2_oss(img_bgr, img_name, "zhengsheng.wcl/problems_segmentation/datasets/prelabeled/")
            out_dict = {
                "url": url,
                "item_list": item_list,
                "labeled_url": labeled_url
            }
            out_info = unicode_str(json.dumps(out_dict))
            write_line(out_file, out_info)
            print('[Info] 写入完成: {}'.format(out_info))

    def process(self):
        data_dir = os.path.join(DATA_DIR, 'labeled_data')
        print('[Info] 数据文件夹: {}'.format(data_dir))
        paths_list, names_list = traverse_dir_files(data_dir)
        out_file_format = os.path.join(DATA_DIR, 'labeled_data_imgs_{}.txt')
        for path, name in zip(paths_list, names_list):
            name = name.split(".")[0]
            out_file = out_file_format.format(name)
            print('[Info] 输出文件: {}'.format(out_file))
            self.process_file(path, out_file)
            break


def main():
    dp = DataPrelabeled()
    dp.process()


if __name__ == '__main__':
    main()