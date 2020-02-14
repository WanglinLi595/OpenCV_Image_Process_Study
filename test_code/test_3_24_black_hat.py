#!/usr/bin/env python
# coding=utf-8
'''
@描述: 黑帽运算
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.13
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.13
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./test_image/gradient.bmp',0)
kernel = np.ones((10,10), np.uint8)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

plt.subplot(121),plt.imshow(img, 'gray'),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blackhat, 'gray'),plt.title('blackhat')
plt.xticks([]), plt.yticks([])
plt.show()