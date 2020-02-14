#!/usr/bin/env python
# coding=utf-8
'''
@描述: 播放视频
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''
import numpy as np
import cv2 as cv

# 创建 VideoCapture 类
cap = cv.VideoCapture("/test_video/vtest.avi")  

while cap.isOpened():       # 视频播放完毕，退出循环
    ret, frame = cap.read()     # 读取视频数据

    if not ret:                 # 读取数据出错
        print("视频解析失败，退出中 ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow("frame", gray)    # 显示图片
    if cv.waitKey(25) == ord('q'):   # 控制播放速度，按 Q 键退出
        break
cap.release()       # 关闭视频
cv.destroyAllWindows()  # 摧毁所有创建的窗口
