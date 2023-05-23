import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH06/Fig0638(a)(lenna_RGB).tif', 1)  # 读取彩色BGR图像
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsi = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

fig, ax = plt.subplots(1, 4)
ax[0].set_title('原始图像')
ax[0].imshow(img_rgb)
ax[0].axis('off')

ax[1].set_title('H')
ax[1].imshow(img_hsi[:, :, 0], cmap='gray')
ax[1].axis('off')

ax[2].set_title('S')
ax[2].imshow(img_hsi[:, :, 1], cmap='gray')
ax[2].axis('off')

ax[3].set_title('V')
ax[3].imshow(img_hsi[:, :, 2], cmap='gray')
ax[3].axis('off')

plt.show()
