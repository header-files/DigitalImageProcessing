import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img1 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0320(1)(top_left).tif', 0)
img2 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0320(2)(2nd_from_top).tif', 0)

img1_equal = cv.equalizeHist(img1)
img2_equal = cv.equalizeHist(img2)

fig, ax = plt.subplots(2, 4)

ax[0, 0].set_title('原始图像')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].axis('off')
ax[1, 0].imshow(img2, cmap='gray')
ax[1, 0].axis('off')

ax[0, 1].set_title('均衡化后图像')
ax[0, 1].imshow(img1_equal, cmap='gray')
ax[0, 1].axis('off')
ax[1, 1].imshow(img2_equal, cmap='gray')
ax[1, 1].axis('off')

ax[0, 2].set_title('原始图像灰度图')
ax[0, 2].hist(img1.ravel(), bins=256)
ax[1, 2].hist(img2.ravel(), bins=256)

ax[0, 3].set_title('均衡化后图像灰度图')
ax[0, 3].hist(img1_equal.ravel(), bins=256)
ax[1, 3].hist(img2_equal.ravel(), bins=256)

plt.show()
