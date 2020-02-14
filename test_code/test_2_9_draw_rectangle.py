
#!/usr/bin/env python
# coding=utf-8
'''
@描述: 绘制矩形
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.06
@最后编辑人: LiWanglin
@最后编辑时间: 2020.02.06
'''

import numpy as np
import cv2 as cv

# 创建一个黑色图框
img = np.zeros((512,512,3), np.uint8)

# 绘制一个蓝色的矩形框
cv.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)

cv.imshow("rectangle", img)         # 显示图片

cv.waitKey(0)

cv.destroyAllWindows()