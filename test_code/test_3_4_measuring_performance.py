#!/usr/bin/env python
# coding=utf-8
'''
@描述: 演示时间戳功能
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.08
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.08
'''

import cv2 as cv

# 获取运行前的时间戳
start_tick = cv.getTickCount()

# 从 0 加到 1000
y = 0
for i in range(1001):
    y += i

# 获取运行后的时间戳
end_tick = cv.getTickCount()

# 打印 CPU 频率
print("TickFrequency :", cv.getTickFrequency())

# 获取从 0 加到 1000 所用的时间
print((end_tick - start_tick)/cv.getTickFrequency())