import cv2 as cv
import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'


def GaussianLowPassFilter(size, center, radius):
    """
    高斯低通滤波器
    :param size: tuple 图片尺寸（行， 列）
    :param center: tuple 滤波器中心坐标
    :param radius: int 半径
    :return: 返回高斯滤波器
    """
    kernel = []

    for row in range(size[0]):
        kernel.append([])
        for col in range(size[1]):
            d = ((row - center[0]) ** 2 + (col - center[1]) ** 2) ** 0.5
            h = np.exp(-1 * (d ** 2) / (2 * (radius ** 2)))
            kernel[row].append([h, h])

    return np.array(kernel)


img = cv.imread('../../img/DIP3E_Original_Images_CH04/Fig0448(a)(characters_test_pattern).tif', 0)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

mask = GaussianLowPassFilter(img.shape, (img.shape[0] / 2, img.shape[1] / 2), 160)

fshift = dft_shift * mask
fshift_magnitude_spectrum = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)

img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

fig, ax = plt.subplots(1, 4)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('原始图像频谱图')
ax[1].imshow(magnitude_spectrum, cmap='gray')
ax[1].axis('off')

ax[2].set_title('高斯低通滤波后频谱图')
ax[2].imshow(fshift_magnitude_spectrum, cmap='gray')
ax[2].axis('off')

ax[3].set_title('高斯低通滤波处理后')
ax[3].imshow(img_back, cmap='gray')
ax[3].axis('off')

plt.show()
