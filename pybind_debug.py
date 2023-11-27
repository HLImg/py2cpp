# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/11/27 20:00:21
# @FileName:   pybind_debug.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import numpy as np
import python.utils.utils_image as util_img

from scipy.io import savemat

class util:
    def __init__(self) -> None:
        print("initial [util] for image processing")
    
    def read(self, path):
        img = util_img.gdal_read(path)
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        img = np.transpose(img, (1, 2, 0))
        if not img.flags['C_CONTIGUOUS']:
            print("read gdal, but not C_CONTIGUOUS")
            img = np.ascontiguousarray(img)
        print(f"[read] the img is {img.shape}, min {img.min()}, max {img.max()}, mean {img.mean()}")
        for c in range(img.shape[2]):
            
            print(f"ori-band-{c + 1}, max - {img[:, :, c].max()},  mean - {img[:, :, c].mean()}")
        return img
    
    def savemat(self, img):
        print(f"savemat, shape is {img.shape}, min {img.min()}, max {img.max()}, mean {img.mean()}")
        savemat("test/save.mat", {'data': img})
        print("finish save mat")
    