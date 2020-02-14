#!/usr/bin/env python
# coding=utf-8
'''
@描述: 测试 OpenCV-Python 包是否有效
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.05
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.05
'''

import cv2 as cv

image_data = cv.imread("../test_image/lenacolor.png", cv.IMREAD_COLOR) # 读取图片数据
cv.imshow("Demo1", image_data)    # 显示图片
cv.waitKey(0)   # 等待用户按下按键
cv.destroyAllWindows()  # 摧毁所有显示图片的窗口