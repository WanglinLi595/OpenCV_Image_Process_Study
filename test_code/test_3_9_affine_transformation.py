#!/usr/bin/env python
# coding=utf-8
'''
@描述: 仿射变换
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.10
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.10
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./opencv_manual/test_image/sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()