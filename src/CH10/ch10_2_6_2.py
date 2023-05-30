import cv2 as cv
import matplotlib as mpl

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH10/Fig1025(a)(building_original).tif', 0)
canny = cv.Canny(img, 200, 255)
edges = cv.threshold(canny, 200, 255, cv.THRESH_BINARY)[1]

fig, ax = plt.subplots(1, 2)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('Canny边缘检测')
ax[1].imshow(edges, cmap='gray')
ax[1].axis('off')

plt.show()
