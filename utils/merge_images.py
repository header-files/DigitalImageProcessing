import warnings

import cv2 as cv
import numpy as np


def merge_imgs(imgs_list, scale, order=None, border=8, border_color=0):
    """
    拼接多张灰度图像
    :param scale: float 原图缩放尺度
    :param imgs_list: list 待显示图像（灰度图，最高支持8位灰度）
    :param order: list or tuple 图像排列顺序 行 * 列
    :param border: int 分割线宽度
    :param border_color: tuple 分割线颜色
    :return: 返回拼接好的numpy数组
    """
    imgs_list = imgs_list.copy()

    if order is None:
        order = [1, len(imgs_list)]
    elif len(imgs_list) > order[0] * order[1]:
        warnings.warn('显示顺序不足以显示所有图片，自动调整为一行显示')
        order = [1, len(imgs_list)]
    elif len(imgs_list) < order[0] * order[1]:
        warnings.warn('图片数量过少，自动调整为一行显示')
        order = [1, len(imgs_list)]

    height = []
    weight = []
    img_idx = 0
    # 遍历得到每行的高度以及每张图片的宽度
    for i in range(order[0]):
        height.append([])
        weight.append([])

        for j in range(order[1]):
            imgs_list[img_idx] = cv.resize(imgs_list[img_idx], dsize=None, fx=scale, fy=scale)

            height[i].append(imgs_list[img_idx].shape[0])
            weight[i].append(imgs_list[img_idx].shape[1])

            img_idx += 1

    show_height = np.sum(np.max(height, axis=1)) + border * (order[0] - 1)  # 每行的高度最大值 + 分割线
    show_weight = np.max(np.sum(weight, axis=1)) + border * (order[1] - 1)  # 所有行宽度中的最大值 + 分割线

    imgs_show = border_color * np.ones((show_height, show_weight), dtype=np.uint8)

    img_idx = 0
    heighted = np.max(height, axis=1)
    # 拼接图片
    for i in range(order[0]):
        for j in range(order[1]):
            x_start = int(np.sum(heighted[0:i]) + border * i)
            x_end = int(x_start + imgs_list[img_idx].shape[0])

            y_start = int(np.sum(weight[i][0:j]) + border * j)
            y_end = int(y_start + imgs_list[img_idx].shape[1])

            imgs_show[x_start:x_end, y_start: y_end] = imgs_list[img_idx]
            img_idx += 1

    return imgs_show
