import cv2 as cv
import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'


def myLaplacian(src, kernel, c):
    """
    拉普拉斯进行锐化
    :param src: numpy数组 待锐化图像
    :param kernel: list or tuple 拉普拉斯算子
    :param c: int 常数，1或-1
    :return: 返回锐化后的图像
    """
    img = src.copy()
    img = cv.addWeighted(img, c, cv.filter2D(img, -1, kernel), 1, 0)
    return img


def myUnsharpMasking(src, k):
    """
    非锐化掩蔽和高提升滤波
    :param src: numpy数组 待锐化图像
    :param k: int 常数。k = 1,非锐化掩蔽；k > 1，高提升滤波
    :return: 返回处理后的图像
    """
    img = src.copy()
    img = cv.addWeighted(img, 1, img - cv.GaussianBlur(img, (5, 5), 3), k, 0)

    return img


img1 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0338(a)(blurry_moon).tif', 0)
kernel_1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel_2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
img1_1 = myLaplacian(img1, kernel_1, -1)
img1_2 = myLaplacian(img1, kernel_2, -1)
img1_3 = cv.Laplacian(img1, ddepth=-1)

img2 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0340(a)(dipxe_text).tif', 0)
img2_1 = cv.GaussianBlur(img2, (5, 5), 3)
img2_2 = myUnsharpMasking(img2, 1)
img2_3 = myUnsharpMasking(img2, 4.5)

img3 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0342(a)(contact_lens_original).tif', 0)
img3_x = cv.Sobel(img3, -1, 1, 0)
img3_y = cv.Sobel(img3, -1, 0, 1)
img3_x = cv.convertScaleAbs(img3_x)
img3_y = cv.convertScaleAbs(img3_y)
img3_1 = cv.addWeighted(img3_x, 0.5, img3_y, 0.5, 0)

fig, ax = plt.subplots(3, 4)
ax[0, 0].set_title('原始图像')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].axis('off')

ax[0, 1].set_title('模版1')
ax[0, 1].imshow(img1_1, cmap='gray')
ax[0, 1].axis('off')

ax[0, 2].set_title('模版2')
ax[0, 2].imshow(img1_2, cmap='gray')
ax[0, 2].axis('off')

ax[0, 3].set_title('OpenCV')
ax[0, 3].imshow(img1_3, cmap='gray')
ax[0, 3].axis('off')

ax[1, 0].set_title('原始图像')
ax[1, 0].imshow(img2, cmap='gray')
ax[1, 0].axis('off')

ax[1, 1].set_title('高斯模糊')
ax[1, 1].imshow(img2_1, cmap='gray')
ax[1, 1].axis('off')

ax[1, 2].set_title('非锐化掩蔽')
ax[1, 2].imshow(img2_2, cmap='gray')
ax[1, 2].axis('off')

ax[1, 3].set_title('高提升滤波')
ax[1, 3].imshow(img2_3, cmap='gray')
ax[1, 3].axis('off')

ax[2, 0].set_title('原始图像')
ax[2, 0].imshow(img3, cmap='gray')
ax[2, 0].axis('off')

ax[2, 1].set_title('Sobel_x')
ax[2, 1].imshow(img3_x, cmap='gray')
ax[2, 1].axis('off')

ax[2, 2].set_title('Sobel_y')
ax[2, 2].imshow(img3_y, cmap='gray')
ax[2, 2].axis('off')

ax[2, 3].set_title('Sobel')
ax[2, 3].imshow(img3_1, cmap='gray')
ax[2, 3].axis('off')

plt.show()
