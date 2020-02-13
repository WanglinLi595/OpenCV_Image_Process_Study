#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图像膨胀
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.13
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.13
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./opencv_manual/test_image/j.png',0)
kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img, 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dilation, 'gray'),plt.title('dilation')
plt.xticks([]), plt.yticks([])
plt.show()