#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图片按位运算
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.07
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.07
'''

import cv2 as cv
import numpy as np

# 读取两张图片
img1 = cv.imread('./opencv_manual/test_image/messi5.jpg')
img2 = cv.imread('./opencv_manual/test_image/opencv-logo-white.png')
# 
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]
# 
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()