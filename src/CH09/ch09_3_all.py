import cv2 as cv
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH09/Fig0911(a)(noisy_fingerprint).tif', 0)
img = (img / 255).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)
img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
img_opening_closing = cv.morphologyEx(img_opening, cv.MORPH_CLOSE, kernel)

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title('原始图像')
ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].axis('off')

ax[0, 1].set_title('开操作')
ax[0, 1].imshow(img_opening, cmap='gray')
ax[0, 1].axis('off')

ax[1, 0].set_title('闭操作')
ax[1, 0].imshow(img_closing, cmap='gray')
ax[1, 0].axis('off')

ax[1, 1].set_title('开操作后的闭操作')
ax[1, 1].imshow(img_opening_closing, cmap='gray')
ax[1, 1].axis('off')

plt.show()
