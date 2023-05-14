import cv2 as cv
import numpy as np

from merge_images import merge_imgs


def gamma_trans(img, gamma, c):
    """
    伽马矫正
    :param img: numpy数组 待校正图像
    :param gamma: float
    :param c: float
    :return: numpy数组 校正后的图像
    """
    img = img.copy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = c * (img ** gamma) * 255

    return img


img = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0307(a)(intensity_ramp).tif', 0)
img1 = gamma_trans(img, 2.5, 1)
img2 = gamma_trans(img, 5.0, 1)
img3 = gamma_trans(img, 10.0, 1)
imgs1 = merge_imgs([img, img1, img2, img3], 0.5, [1, 4])

img = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0308(a)(fractured_spine).tif', 0)
img1 = gamma_trans(img, 0.6, 1)
img2 = gamma_trans(img, 0.4, 1)
img3 = gamma_trans(img, 0.3, 1)
imgs2 = merge_imgs([img, img1, img2, img3], 0.3, [1, 4])

img = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0309(a)(washed_out_aerial_image).tif', 0)
img1 = gamma_trans(img, 3.0, 1)
img2 = gamma_trans(img, 4.0, 1)
img3 = gamma_trans(img, 5.0, 1)
imgs3 = merge_imgs([img, img1, img2, img3], 0.3, [1, 4])

cv.imshow('图片'.encode("gb2312").decode(errors="ignore"), imgs1)
cv.imshow('脊椎骨折'.encode("gb2312").decode(errors="ignore"), imgs2)
cv.imshow('航拍图像'.encode("gbk").decode(errors="ignore"), imgs3)
cv.waitKey(0)
cv.destroyAllWindows()
