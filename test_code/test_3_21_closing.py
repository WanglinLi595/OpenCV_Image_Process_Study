#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图像闭运算
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.13
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.13
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./opencv_manual/test_image/closing.bmp',0)
kernel = np.ones((10,10), np.uint8)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

plt.subplot(121),plt.imshow(img, 'gray'),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(closing, 'gray'),plt.title('closing')
plt.xticks([]), plt.yticks([])
plt.show()