#!/usr/bin/env python
# coding=utf-8
'''
@描述: 绘制多边形
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import cv2 as cv
import numpy as np

# 确定两个背景板，形成对比
img = np.zeros((512,512,3), np.uint8)

# 确定 4 个点，四个点按照顺序连接
pts = np.array([[50,15],[120,30],[170,50],[30,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img, [pts], True, (0,0,255))

cv.imshow("polygon", img)

cv.waitKey(0)
cv.destroyAllWindows()

