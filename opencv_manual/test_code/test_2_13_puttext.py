#!/usr/bin/env python
# coding=utf-8
'''
@描述: 在图片中添加文本
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import cv2 as cv
import numpy as np

img = np.zeros((512,512,3), np.uint8)

font = cv.FONT_HERSHEY_SIMPLEX  # 确定字体类型
cv.putText(img, 'OpenCV', (10,400), font, 4, (255,255,255), 2 , cv.LINE_AA)

cv.imshow("text", img)      # 图片显示

cv.waitKey(0)
cv.destroyAllWindows()