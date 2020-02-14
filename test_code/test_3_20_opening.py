#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图像开运算
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.13
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.13
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./test_image/opening.bmp',0)
kernel = np.ones((5,5), np.uint8)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

plt.subplot(121),plt.imshow(img, 'gray'),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opening, 'gray'),plt.title('opening')
plt.xticks([]), plt.yticks([])
plt.show()