import cv2 as cv
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img1 = cv.imread('../../img/DIP3E_Original_Images_CH09/Fig0905(a)(wirebond-mask).tif', 0)
img1 = (img1 / 255).astype(np.uint8)

kernel = np.ones((15, 15), np.uint8)
img_erosion = cv.erode(img1, kernel, iterations=1)

img2 = cv.imread('../../img/DIP3E_Original_Images_CH09/Fig0907(a)(text_gaps_1_and_2_pixels).tif', 0)
img2 = (img2 / 255).astype(np.uint8)

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
img_dilation = cv.dilate(img2, kernel, iterations=1)

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title('原始图像')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].axis('off')

ax[0, 1].set_title('腐蚀操作')
ax[0, 1].imshow(img_erosion, cmap='gray')
ax[0, 1].axis('off')

ax[1, 0].imshow(img2, cmap='gray')
ax[1, 0].axis('off')

ax[1, 1].set_title('膨胀操作')
ax[1, 1].imshow(img_dilation, cmap='gray')
ax[1, 1].axis('off')

plt.show()
