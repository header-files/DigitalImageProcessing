import cv2 as cv
import matplotlib as mpl

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH10/Fig1039(a)(polymersomes).tif', 0)
thres = cv.threshold(img, 0, 255, cv.THRESH_OTSU)[1]

fig, ax = plt.subplots(1, 2)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('Otsu阈值处理')
ax[1].imshow(thres, cmap='gray')
ax[1].axis('off')

plt.show()
