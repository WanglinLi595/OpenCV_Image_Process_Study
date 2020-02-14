#!/usr/bin/env python
# coding=utf-8
'''
@描述: 实现图片缩放
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.09
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.09
'''

import numpy as np
import cv2 as cv
img = cv.imread('./opencv_manual/test_image/messi5.jpg')
# 图片扩大两倍
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

cv.imshow("original", img)
cv.imshow("res", res)

cv.waitKey(0)
cv.destroyAllWindows()