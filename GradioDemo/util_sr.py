# -*- coding: utf-8 -*-
# @Time    : 10/26/23 8:37 PM
# @File    : util_test.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import gradio
import cv2 as cv
import numpy as np
import util_common

from method.super_resolution.sr_main import SR_Inference

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
    
def sr_train(msi, pan, method, img_type, lr, save_freq, save_dir, progress=gradio.Progress()):
    if img_type == '全色图像':
        lq = pan
    elif img_type == '多光谱图像':
        lq = msi
    
    if method == '卷积神经网络':
        use_method = 'cnn'
    elif method == '生成对抗网络':
        use_method = 'gan'
    else:
        use_method = 'cubic'
    
    save_freq = int(save_freq)
    
    
    import time
    for i in progress.tqdm(range(100)):
        time.sleep(1)

def sr_test_interface(msi, pan, method, img_type, ckpt_path, sr_scale):
    
    if img_type == '全色图像':
        lq_path = msi
        img_type = "pan"
    elif img_type == '多光谱图像':
        lq_path = msi
        img_type = "msi"
    
    
    if method == '卷积神经网络':
        use_method = 'cnn'
    elif method == '生成对抗网络':
        use_method = 'gan'
    else:
        use_method = 'cubic'
    
    sr_scale = int(sr_scale)
    
    # sr_inference = SR_Inference(lq_path, use_method, img_type, 
    #                             ckpt_path, sr_scale)
    
    lq = util_common.gdal_read(lq_path)
    lq = np.transpose(lq, (1, 2, 0))
    print("#####################  ", lq_path)
    print("#####################  ", np.mean(lq))
    res = lq[::2, ::2, :3] if img_type == "msi" else lq[::2, ::2]
    return res / res.max(), util_common.download_tif(lq)
    
    



