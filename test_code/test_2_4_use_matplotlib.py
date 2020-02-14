#!/usr/bin/env python
# coding=utf-8
'''
@描述: 使用 Matplotlib 显示图片
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED) # 读取图片
# 设置图片在 matplotlib 的显示方式
plt.imshow(img, cmap="gray", interpolation='bicubic')
plt.show()          # 显示图像