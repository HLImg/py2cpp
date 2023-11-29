# -*- coding: utf-8 -*-
# @Time    : 10/26/23 8:37 PM
# @File    : util_test.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import cv2 as cv
import numpy as np



def sr_test_interface(image, scale=2, ctype="all"):
    if ctype == '全色图像':
        ctype = 'pan'
    elif ctype == '多光谱图像':
        ctype = 'msi'
    
    h, w = image.shape[:2]
    return cv.resize(image, (w * scale, h * scale))
    



