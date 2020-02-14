#!/usr/bin/env python
# coding=utf-8
'''
@描述: 绘制椭圆
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''


import cv2 as cv
import numpy as np

# 确定两个背景板，形成对比
ellipse_1 = np.zeros((512,512,3), np.uint8)
ellipse_2 = np.zeros((512,512,3), np.uint8)

# 两者的起始角不同，一个为 0° ，一个为 90°
cv.ellipse(ellipse_1, (256,256), (100,50), 0, 0, 360, 255, -1)
cv.ellipse(ellipse_2, (256,256), (100,50), 0, 90, 360, 255, -1)

# 显示图像
cv.imshow("ellipse_1", ellipse_1)
cv.imshow("ellipse_2", ellipse_2)

cv.waitKey()

cv.destroyAllWindows()