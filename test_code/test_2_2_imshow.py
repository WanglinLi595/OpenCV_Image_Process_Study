#!/usr/bin/env python
# coding=utf-8
'''
@描述: 读取并显示图片
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''
import cv2 as cv

image = cv.imread("../test_image/lena512.bmp", cv.IMREAD_UNCHANGED)  # 读取图片
cv.imshow("Test_3", image)     # 开辟一个窗口显示图片
cv.waitKey(0)                  # 等待用户按下按键
cv.destroyAllWindows()         # 释放所有窗口