import cv2 as cv
import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'

img = cv.imread('../../img/DIP3E_Original_Images_CH10/Fig1034(a)(marion_airport).tif', 0)
canny = cv.Canny(img, 200, 255)
edges = cv.threshold(canny, 200, 255, cv.THRESH_BINARY)[1]
lines = cv.HoughLines(edges, 1, np.pi / 180, 220)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 10)

fig, ax = plt.subplots(1, 2)
ax[0].set_title('原始图像')
ax[0].imshow(img, cmap='gray')
ax[0].axis('off')

ax[1].set_title('边缘检测')
ax[1].imshow(edges, cmap='gray')
ax[1].axis('off')

plt.show()
