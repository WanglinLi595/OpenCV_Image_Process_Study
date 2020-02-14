#!/usr/bin/env python
# coding=utf-8
'''
@描述: 
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.09
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.09
'''

import numpy as np
import cv2 as cv
img = cv.imread('./test_image/messi5.jpg',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()