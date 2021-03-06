#!/usr/bin/env python
# coding=utf-8
'''
@描述: 高斯滤波
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.12
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.12
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./test_image/opencv-logo.png')
blur = cv.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()