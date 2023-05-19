import cv2 as cv
import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH04/Fig0441(a)(characters_test_pattern).tif', 0)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.ones((rows, cols, 2), np.uint8)
cv.circle(mask, (crow, ccol), 100, (0, 0), -1)  # 理想高通滤波器
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

ax[2].set_title('理想高通滤波后频谱图')
ax[2].imshow(fshift_magnitude_spectrum, cmap='gray')
ax[2].axis('off')

ax[3].set_title('理想高通滤波处理后')
ax[3].imshow(img_back, cmap='gray')
ax[3].axis('off')

plt.show()
