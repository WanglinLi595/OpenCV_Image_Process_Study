#!/usr/bin/env python
# coding=utf-8
'''
@描述: 画线
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import cv2 as cv
import numpy as np

img = np.zeros((512, 512, 3), dtype=np.uint8)  # 创建黑色背景图
cv.imshow("img", img)   # 显示原图

result = cv.line(img, (0 ,0), (511, 511), (0, 255, 0), 1)   # 在背景图上绘制线条
cv.imshow("result", result)         # 显示处理结果图

cv.waitKey(0)       # 等待按键
cv.destroyAllWindows()  # 摧毁窗口