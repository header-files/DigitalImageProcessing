import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img1 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0326(a)(embedded_square_noisy_512).tif', 0)
img2 = cv.imread('../../img/DIP3E_Original_Images_CH03/Fig0327(a)(tungsten_original).tif', 0)

img1_equal = cv.equalizeHist(img1)
clashe = cv.createCLAHE(clipLimit=20, tileGridSize=(3, 3))
img1_clashe = clashe.apply(img1)

img2_equal = cv.equalizeHist(img2)
clashe = cv.createCLAHE(clipLimit=50, tileGridSize=(3, 3))
img2_clashe = clashe.apply(img2)

fig, ax = plt.subplots(2, 3)
ax[0, 0].set_title('原始图像')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].axis('off')
ax[1, 0].imshow(img2, cmap='gray')
ax[1, 0].axis('off')

ax[0, 1].set_title('全局直方图平衡')
ax[0, 1].imshow(img1_equal, cmap='gray')
ax[0, 1].axis('off')
ax[1, 1].imshow(img2_equal, cmap='gray')
ax[1, 1].axis('off')

ax[0, 2].set_title('局部直方图平衡')
ax[0, 2].imshow(img1_clashe, cmap='gray')
ax[0, 2].axis('off')
ax[1, 2].imshow(img2_clashe, cmap='gray')
ax[1, 2].axis('off')

plt.show()
