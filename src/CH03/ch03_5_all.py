import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif', 0)
img_mean = cv.blur(img, (3, 3))
img_median = cv.medianBlur(img, 3)

fig, ax = plt.subplots(1, 3)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('3x3均值滤波')
ax[1].imshow(img_mean, cmap='gray')
ax[1].axis('off')

ax[2].set_title('3x3中值滤波')
ax[2].imshow(img_median, cmap='gray')
ax[2].axis('off')

plt.show()
