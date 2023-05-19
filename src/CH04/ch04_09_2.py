import cv2 as cv
import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'


def BHPF(size, center, radius, n):
    """
    布特沃斯高通滤波器
    :param size: tuple 图片尺寸（行， 列）
    :param center: tuple 滤波器中心坐标
    :param radius: int 半径
    :param n: int 阶数
    :return: 返回滤波器
    """
    kernel = []

    for row in range(size[0]):
        kernel.append([])
        for col in range(size[1]):
            d = ((row - center[0]) ** 2 + (col - center[1]) ** 2) ** 0.5
            if d == 0:
                h = 0
            else:
                h = 1 / (1 + (radius / d) ** (2 * n))
            kernel[row].append([h, h])

    return np.array(kernel)


img = cv.imread('../../img/DIP3E_Original_Images_CH04/Fig0457(a)(thumb_print).tif', 0)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

mask = BHPF(img.shape, (img.shape[0] / 2, img.shape[1] / 2), 50, 4)

fshift = dft_shift * mask
fshift_magnitude_spectrum = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)

img_back = cv.idft(f_ishift)

img_back[np.where(img_back[:, :, 0] < 0)] = 0
img_back[np.where(img_back[:, :, 0] != 0)] = 1
img_back[np.where(img_back[:, :, 1] < 0)] = 0
img_back[np.where(img_back[:, :, 1] != 0)] = 1

img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

fig, ax = plt.subplots(1, 4)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('原始图像频谱图')
ax[1].imshow(magnitude_spectrum, cmap='gray')
ax[1].axis('off')

ax[2].set_title('布特沃斯高通滤波后频谱图')
ax[2].imshow(fshift_magnitude_spectrum, cmap='gray')
ax[2].axis('off')

ax[3].set_title('布特沃斯高通滤波处理后')
ax[3].imshow(img_back, cmap='gray')
ax[3].axis('off')

plt.show()
