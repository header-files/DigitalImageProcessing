import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH06/Fig0638(a)(lenna_RGB).tif', 1)  # 读取彩色BGR图像
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_mean = cv.blur(img_rgb, (5, 5))

fig, ax = plt.subplots(1, 2)
ax[0].set_title('原始图像')
ax[0].imshow(img_rgb)
ax[0].axis('off')

ax[1].set_title('均值滤波器平滑图像')
ax[1].imshow(img_mean)
ax[1].axis('off')

plt.show()
