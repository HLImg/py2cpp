# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/12/13 19:00:38
# @FileName:  metric.py
# @Contact :  lianghao@whu.edu.cn

import numpy as np
import scipy.ndimage
import imgvision as iv

def calculate_image_gradient(image):
    # 计算水平和垂直方向上的梯度
    gradient_x = scipy.ndimage.sobel(image, axis=0, mode='constant')
    gradient_y = scipy.ndimage.sobel(image, axis=1, mode='constant')

    # 计算梯度幅值
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 返回平均梯度
    return np.mean(magnitude)

def cal_psnr(lq, hq):
    metric = iv.spectra_metric(lq, hq)
    return metric.PSNR()

def cal_ssim(lq, hq):
    metric = iv.spectra_metric(lq, hq)
    return metric.SSIM()

def cal_ergas(lq, hq):
    metric = iv.spectra_metric(lq, hq)
    return metric.ERGAS()

def cal_rmse(lq, hq):
    mse = np.mean((lq - hq) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def cal_avg_grad(im_lq, im_hq):
    res = f"({calculate_image_gradient(im_lq):.4f}, "
    res = res + f"{calculate_image_gradient(im_hq):.4f})"
    return res