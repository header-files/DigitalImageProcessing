import cv2 as cv
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'
from src.CH04.ch04_09_3 import GaussianHighPassFilter


def HighFrequencyEmphasisFilter(src, k1, k2):
    """
    高频强调滤波
    :param src: numpy数组 待锐化图像
    :param k1: float 非负数，控制距原点的偏移量
    :param k2: float 非负数，控制高频的贡献
    :return: 返回处理后的图像
    """
    img = src.copy()

    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    high_filter = GaussianHighPassFilter(dft_shift.shape, (dft_shift.shape[0] / 2, dft_shift.shape[1] / 2), 40)
    fshift = (k1 + k2 * high_filter) * dft_shift
    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


img = cv.imread('../../img/DIP3E_Original_Images_CH04/Fig0459(a)(orig_chest_xray).tif', 0)
img_back = HighFrequencyEmphasisFilter(img, 0.5, 0.75)
img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
img_equal = cv.equalizeHist(img_back)

fig, ax = plt.subplots(1, 3)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('高频强调')
ax[1].imshow(img_back, cmap='gray')
ax[1].axis('off')

ax[2].set_title('高频强调+直方图均衡')
ax[2].imshow(img_equal, cmap='gray')
ax[2].axis('off')

plt.show()
