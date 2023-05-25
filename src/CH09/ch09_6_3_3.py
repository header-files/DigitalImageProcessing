import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH09/Fig0940(a)(rice_image_with_intensity_gradient).tif', 0)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (80, 80))

img_tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

fig, ax = plt.subplots(1, 2)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('顶帽变换')
ax[1].imshow(img_tophat, cmap='gray')
ax[1].axis('off')

plt.show()
