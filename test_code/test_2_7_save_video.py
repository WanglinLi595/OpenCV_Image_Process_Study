#!/usr/bin/env python
# coding=utf-8
'''
@描述: 保存视频
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)        # 打开相机

fourcc = cv.VideoWriter_fourcc(*'XVID')     # 定义编码对象
# 创建 VideoWriter 对象
out = cv.VideoWriter("output.avi", fourcc, 20.0 ,(640, 480))

while cap.isOpened():
    ret, frame = cap.read() # 读取帧

    if not ret:         # 帧的读取结果有误
        print("不能接受数据帧。退出中 ...")
        exit

    out.write(frame)        # 写入帧

    cv.imshow("frame", frame)   # 显示图像

    if cv.waitKey(1) == ord('q'):   # 按 Q 退出
        break

# 工作完成后，释放所有内容
cap.release()
out.release()

cv.destroyAllWindows()  # 摧毁窗口