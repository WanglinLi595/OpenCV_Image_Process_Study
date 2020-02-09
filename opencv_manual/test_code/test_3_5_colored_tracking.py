#!/usr/bin/env python
# coding=utf-8
'''
@描述: 颜色追踪
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.09
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.09
'''

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):
    # 提取每一帧
    _, frame = cap.read()
    # 色彩空间转换
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 设定在 HSV 中蓝色的取值范围
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # 进行二值化处理
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # 按位运算，得到蓝色区域
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()


