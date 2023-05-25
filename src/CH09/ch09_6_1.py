import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH09/Fig0935(a)(ckt_board_section).tif', 0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
img_erosion = cv.erode(img, kernel, iterations=1)
img_dilation = cv.dilate(img, kernel, iterations=1)

fig, ax = plt.subplots(1, 3)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('腐蚀操作')
ax[1].imshow(img_erosion, cmap='gray')
ax[1].axis('off')

ax[2].set_title('膨胀操作')
ax[2].imshow(img_dilation, cmap='gray')
ax[2].axis('off')

plt.show()
