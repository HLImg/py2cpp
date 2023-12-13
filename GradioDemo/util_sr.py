# -*- coding: utf-8 -*-
# @Time    : 10/26/23 8:37 PM
# @File    : util_test.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import gradio
import cv2 as cv
import numpy as np
import util_common

def sr_generate_image(msi, pan, kernel_size, kernel_std, noise, jpeg, output_path, progress=gradio.Progress()):
    print(msi.shape)
    print(pan.shape)
    print("sr_gen")
    print("kernel_size ", kernel_size)
    print("kernel_std ", kernel_std)
    print("noise ", noise)
    print("jped ", jpeg)
    print("output: ", output_path)
    
    import time
    for i in progress.tqdm(range(100)):
        time.sleep(1)
    


def sr_test_interface(image, scale=2, ctype="all"):
    if ctype == '全色图像':
        ctype = 'pan'
    elif ctype == '多光谱图像':
        ctype = 'msi'
    
    h, w = image.shape[:2]
    return cv.resize(image, (w * scale, h * scale))
    



