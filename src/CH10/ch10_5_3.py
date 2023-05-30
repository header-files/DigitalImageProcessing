import cv2 as cv
import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH10/Fig1056(a)(blob_original).tif', 0)
thres = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)[1]  # 二值化

kernel = np.ones((3, 3), np.uint8)
fg = cv.erode(thres, kernel, iterations=8)  # 腐蚀操作，使目标区域缩小并分开不同的区域，得到真实前景
bg = cv.dilate(thres, kernel, iterations=1)  # 膨胀操作，得到真实背景
unknown = cv.subtract(bg, fg)  # 未知区域，即边界
ret, markers = cv.connectedComponents(fg)  # 计算连通区域
markers = markers + 1
markers[unknown == 255] = 0

img1 = cv.imread('../../img/DIP3E_Original_Images_CH10/Fig1056(a)(blob_original).tif', 1)
markers = cv.watershed(img1, markers)
img1[markers == -1] = [255, 0, 0]

fig, ax = plt.subplots(1, 2)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('分水岭分割')
ax[1].imshow(img1)
ax[1].axis('off')

plt.show()
