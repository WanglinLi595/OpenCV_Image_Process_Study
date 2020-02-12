#!/usr/bin/env python
# coding=utf-8
'''
@描述: 均值滤波
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.12
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.12
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./opencv_manual/test_image/opencv-logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()