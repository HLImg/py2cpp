# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/20 15:09:57
# @FileName:  pybind_util.py
# @Contact :  lianghao@whu.edu.cn

import os
import numpy as np
import xml.dom.minidom
import python.utils.utils_sisr as util_sr
import python.utils.utils_image as util_img
import python.utils.utils_deblur as util_deblur


def statis(img):
    print(f"read_gdal : xmin = [{img.min()}], xmax = [{img.max()}], mean = [{img.mean()}], shape = {img.shape}, dtype = {img.dtype}")

class Util:
    def __init__(self) -> None:
        print("initial [Util] for image processing")
    
    def read_gdal_mul(self, path):
        image = util_img.gdal_read(path)
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        
        image = np.transpose(image, (1, 2, 0))
        # important operators
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)   
        return image
    
    def calWeight(self, d, k):
        x = np.arange(-d / 2, d / 2)
        y = 1 / (1 + np.exp(-k * x))
        return y 
    
    def tile(self, weight, k1, k2):
        return np.tile(weight, (k1, k2))
    
    def reshape(self, d, new_h, new_w):
        return np.reshape(d, (new_h, new_w))
    
    
