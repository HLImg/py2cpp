# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/11/29 21:40:19
# @FileName:   util_metric.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import random

from method.metric import *
from util_common import is_pan

   
metrices = {
    'psnr': cal_psnr,
    'ssim': cal_ssim,
    'rmse': cal_rmse,
    'ergas': cal_ergas,
    'avg_grad': cal_avg_grad,
    'qnr': cal_qnr
}


def metric_interface(msi_lq_path, pan_hq_path, fusion_path, selected_metrics):
    """
    需要读取图像
    """
    msi_lq = np.random.randn(128, 128, 3)
    pan_hq = np.random.randn(128, 128, 3)
    fusion = np.random.randn(128, 128, 3)
    
    print(msi_lq.shape, pan_hq.shape)
    
    results = []
    for metric in selected_metrics:
        if metric.lower() == 'qnr':
            metric_res = metrices[metric.lower()](msi_lq, pan_hq, fusion)
        else:
            metric_res = metrices[metric.lower()](msi_lq, pan_hq)
        results.append([metric, metric_res])
    return results
