#!/usr/bin/env python
# coding=utf-8
'''
@描述: 读取图片数据
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import cv2 as cv

image = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED)
print("image的类型为：", type(image))# 打印图片类型
print("image = \n", image)# 打印图片数据