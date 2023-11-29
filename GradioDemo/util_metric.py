# -*- coding: utf-8 -*-
# @Author  :   Hao Liang
# @Time    :   2023/11/29 21:40:19
# @FileName:   util_metric.py
# @Contact :   lianghao@whu.edu.cn 
# @Device  :   private

import random

def metric_interface(selected_metrics):
    
    results = []

    for metric in selected_metrics:
        results.append([metric, round(random.uniform(0, 100), 2)])
    
    return results