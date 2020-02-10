#!/usr/bin/env python
# coding=utf-8
'''
@描述: 进行图片的旋转
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.10
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.10
'''

import cv2 as cv

img = cv.imread('./opencv_manual/test_image/messi5.jpg',0)
rows,cols = img.shape
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()