#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图片简单阈值化
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.11
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.11
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./test_image/gradient.png',0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()