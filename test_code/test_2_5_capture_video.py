#!/usr/bin/env python
# coding=utf-8
'''
@描述: 捕获视频
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)    # 创建 VideoCapture 对象
if not cap.isOpened():       # 相机打开失败
    print("相机打开失败")
    exit()                  # 退出程序

while True:
    # 一帧一帧的捕获，如果正确读取帧，ret 返回 True
    ret, frame = cap.read()     

    if not ret:       # 读取帧出错
        print("不能接收帧，退出中 ...")
        break

    # 显示图像，循环显示，相当于播放视频
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord('q'):       # 按下 Q 键退出
        break
cap.release()       # 在结尾的时候，一定要释放捕获
cv.destroyAllWindows()      # 摧毁所有创建的窗口