#!/usr/bin/env python
# coding=utf-8
'''
@描述: 为图片添加边框
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
img1 = cv.imread("./opencv_manual/test_image/OpenCVLogo.jpg")

# 为图片添加不同类型的边框
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
BLUE = [255,0,0]
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)

# 显示不同类型的边框图片
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()