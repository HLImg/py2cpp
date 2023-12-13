# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/11/29 21:40:19
# @FileName:   util_metric.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import random

from method.metric import *
   
metrices = {
    'psnr': cal_psnr,
    'ssim': cal_ssim,
    'rmse': cal_rmse,
    'ergas': cal_ergas,
    'avg_grad': cal_avg_grad
}

def metric_interface(im_lq, im_gt, selected_metrics):
    results = []
    
    for metric in selected_metrics:
        metric_res = metrices[metric.lower()](im_lq, im_gt)
        results.append([metric, metric_res])
    
    return results
