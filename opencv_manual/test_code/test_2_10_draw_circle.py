#!/usr/bin/env python
# coding=utf-8
'''
@描述: 绘制圆形
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''
import cv2 as cv
import numpy as np

# 创建一个黑色图框
img = np.zeros((512,512,3), np.uint8)

# 绘制一个红色的圆
cv.circle(img, (120,120), 63, (0,0,255), -1)

cv.imshow("circle", img)         # 显示图片

cv.waitKey(0)

cv.destroyAllWindows()