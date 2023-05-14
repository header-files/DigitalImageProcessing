import cv2 as cv
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'


def hist_specification(img, regulation):
    """
    直方图规定化灰度图像
    :param img: numpy数组 待校正图像
    :param regulation: list or tuple 规定的直方图（大小：256，元素为该灰度像素个数）
    :return: 返回直方图规定化后的numpy数组
    """
    img = img.copy()
    img = cv.equalizeHist(img)

    # 计算规定直方图的变换函数
    p = regulation / np.sum(regulation)
    lut = list(range(-256, 0))
    for i, p_i in enumerate(p):
        g = round(255 * np.sum(p[0:(i + 1)]))
        lut[g] = i

    # 防止某些灰度值在变换的过程中缺失
    if lut[0] < 0:
        lut[0] = 0

    for i, g_z in enumerate(lut):
        if g_z < 0:
            lut[i] = lut[i - 1]

    img = cv.LUT(img, np.array(lut))

    return img


img = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0316(1)(top_left).tif', 0)
img_regulation = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0316(2)(2nd_from_top).tif', 0)

regulation = np.bincount(img_regulation.ravel())
img_spe = hist_specification(img, regulation)

fig, ax = plt.subplots(2, 3)

ax[0, 0].set_title('原始图像')
ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].axis('off')
ax[1, 0].hist(img.ravel(), bins=256)

ax[0, 1].set_title('用于规定化的图像')
ax[0, 1].imshow(img_regulation, cmap='gray')
ax[0, 1].axis('off')
ax[1, 1].hist(img_regulation.ravel(), bins=256)

ax[0, 2].set_title('规定化后图像')
ax[0, 2].imshow(img_spe, cmap='gray')
ax[0, 2].axis('off')
ax[1, 2].hist(img_spe.ravel(), bins=256)

plt.show()
