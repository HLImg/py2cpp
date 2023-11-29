# -*- coding: utf-8 -*-
# @Time    : 10/26/23 8:37 PM
# @File    : util_test.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import cv2 as cv
import numpy as np

def super_resolution(image, scale=2):
    h, w, c = image.shape
    out = cv.resize(image, (scale * h, scale * w))
    return out
