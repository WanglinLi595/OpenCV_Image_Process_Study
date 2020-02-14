#!/usr/bin/env python
# coding=utf-8
'''
@描述: 保存图片
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''
import numpy as np
import cv2 as cv

img = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED)    # 读取图片
cv.imshow("image", img)     # 显示图片
k = cv.waitKey(0)       # 等待用户按键
if(k == 27):            # 按下 ESC 键摧毁窗口
    cv.destroyAllWindows()
elif(k == ord('s')):     # 按下 S 键保存图片并退出
    cv.imwrite("messigray.png", img)
    cv.destroyAllWindows()