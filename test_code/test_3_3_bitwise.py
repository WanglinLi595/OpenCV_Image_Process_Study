#!/usr/bin/env python
# coding=utf-8
'''
@描述: 图片按位运算
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.02.07
@最后编辑人: LiWanglin
@最后编辑时间: 2020.03.27
'''

    import cv2 as cv
    import numpy as np

    # 先定义像个数，用来执行操作
    a = np.uint8([11]) # 二进制：0000 1011
    b = np.uint8([13]) # 二进制：0000 1101

    # 按位与
    # b'0000 1011 & b'0000 1101 = b'0000 1001 = 9
    res_and = cv.bitwise_and(a, b)  
    # 按位或
    # b'0000 1011 | b'0000 1101 = b'0000 1111 = 15
    res_or = cv.bitwise_or(a, b)
    # 按位非
    #  - b'0000 1011 = b'1111 0100 = 244
    res_not = cv.bitwise_not(a)
    # 按位异或
    # b'0000 1011 xor b'0000 1101 = b'0000 0110 = 6
    res_xor = cv.bitwise_xor(a, b)

    # 打印输出结果
    print("res_and is:",res_and)
    print("res_or is:",res_or)
    print("res_not is:",res_not)
    print("res_xor is:",res_xor)



