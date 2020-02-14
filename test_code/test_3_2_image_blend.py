#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图片混合
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.07
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.07
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
img1 = cv.imread("./test_image/boat.bmp")
img2 = cv.imread("./test_image/dollar.bmp")

# 图片按照不同的权重进行相加
dst = cv.addWeighted(img1, 0.8, img2, 0.2, 0)

# 显示图片
plt.subplot(131),plt.imshow(img1,'gray'),plt.title('img1')
plt.subplot(132),plt.imshow(img2,'gray'),plt.title('img2')
plt.subplot(133),plt.imshow(dst,'gray'),plt.title('dst')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()