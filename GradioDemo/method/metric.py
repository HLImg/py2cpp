# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/12/13 19:00:38
# @FileName:  metric.py
# @Contact :  lianghao@whu.edu.cn

import cv2
import numpy as np
import scipy.ndimage
import imgvision as iv

from skimage.metrics import peak_signal_noise_ratio as cal_psnr

def calculate_image_gradient(image):
    # 计算水平和垂直方向上的梯度
    gradient_x = scipy.ndimage.sobel(image, axis=0, mode='constant')
    gradient_y = scipy.ndimage.sobel(image, axis=1, mode='constant')

    # 计算梯度幅值
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 返回平均梯度
    return np.mean(magnitude)

# def cal_psnr(lq, hq):
#     metric = iv.spectra_metric(lq, hq)
#     return metric.PSNR()

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

def cal_qnr(ps,l_ms,pan):
    D1 = D_lamda(ps, l_ms)
    D2 = D_s(ps, l_ms, pan)

    return round(((1 - D1) * (1 - D2)), 3)

def Q(a, b):
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    temp = np.cov(a, b)
    d1 = temp[0, 0]
    cov = temp[0, 1]
    d2 = temp[1, 1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4 * cov * m1 * m2 / (d1 + d2) / (m1 ** 2 + m2 ** 2)

    return Q

def D_lamda(ps, l_ms):
    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                # print(np.abs(Q(ps[:, :, i], ms[:, :, j]) - Q(l_ps[:, :, i], l_ms[:, :, j])))
                sum += np.abs(Q(ps[:, :, i], ps[:, :, j]) - Q(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)


def D_s(ps, l_ms, pan):
    L = ps.shape[2]
    # h, w = pan.shape
    # l_pan = cv2.resize(pan, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    l_pan = cv2.pyrDown(pan)
    l_pan = cv2.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(Q(ps[:, :, i], pan) - Q(l_ms[:, :, i], l_pan))
    return sum / L
