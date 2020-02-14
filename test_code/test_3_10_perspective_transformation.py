#!/usr/bin/env python
# coding=utf-8
'''
@描述: 投射变换
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.10
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.10
'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('./test_image/sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()