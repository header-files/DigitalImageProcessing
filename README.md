# 数字图像处理实验

《数字图像处理（冈萨雷斯，第三版）》中部分图像处理实验，基于OpenCV-Python实现。

## 参考资料

1. 《数字图像处理_第三版_冈萨雷斯》

   [电子图书](./book/数字图像处理_第三版_冈萨雷斯.pdf)
   [书内配图](https://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm)
2. OpenCV

   [官方文档](https://docs.opencv.org/4.7.0/d6/d00/tutorial_py_root.html)
   [中文文档](./book/opencv4.1中文文档.pdf)

## 实验

### [第三章](./src/CH03)

| 代码                                                             | 原文      | 主要函数                                                 | 备注                                                 |
|----------------------------------------------------------------|---------|------------------------------------------------------|----------------------------------------------------|
| [伽马校正](./src/CH03/ch03_2_3.py)                                 | 67、68页  | 自己实现                                                 | -                                                  |
| [直方图均衡化](./src/CH03/ch03_3_1.py)                               | 77页     | cv.equalizeHist()                                    | OpenCV具体实现与公式略有不同：当图像只有一种灰度值时（比如纯黑图像），不再均衡，直接取原灰度值 |
| [直方图匹配](./src/CH03/ch03_3_2.py)                                | 83页     | 自己实现                                                 | -                                                  |
| [对比度受限的自适应直方图均衡化(CLAHE)](./src/CH03/ch03_3_3.py)               | 85、87页  | cv.createCLAHE()                                     | -                                                  |
| [平滑空间滤波器：均值滤波器、中值滤波器](./src/CH03/ch03_5_all.py)                | 93~97 页 | cv.blur、cv.medianBlur                                | -                                                  |         
| [锐化空间滤波器：拉普拉斯算子、非锐化掩蔽和高提升滤波、Sobel算子](./src/CH03/ch03_6_all.py) | 97~103页 | cv.Laplacian、cv.GaussianBlur、cv.Sobel、cv.addWeighted | -                                                  |
