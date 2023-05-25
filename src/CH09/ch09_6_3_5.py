import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH09/Fig0943(a)(dark_blobs_on_light_background).tif', 0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (120, 120))
img_opening = cv.morphologyEx(img_closing, cv.MORPH_OPEN, kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
edge = cv.morphologyEx(img_opening, cv.MORPH_GRADIENT, kernel)
img_edge = cv.addWeighted(img, 0.5, edge, 0.5, 0)

fig, ax = plt.subplots(1, 4)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('闭操作')
ax[1].imshow(img_closing, cmap='gray')
ax[1].axis('off')

ax[2].set_title('闭操作后开操作')
ax[2].imshow(img_opening, cmap='gray')
ax[2].axis('off')

ax[3].set_title('叠加边界')
ax[3].imshow(img_edge, cmap='gray')
ax[3].axis('off')

plt.show()
