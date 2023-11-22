# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/19 20:57:53
# @FileName:  pybind_model.py
# @Contact :  lianghao@whu.edu.cn

import torch
import numpy as np

import python.utils.utils_image as util_img
from python.model.network_usrnet_v1 import USRNet as net

class Model:
    def __init__(self, k, sr_parser, xmax):
        print("start initialize [inference model]")
        torch.cuda.empty_cache()
        
        if 'tiny' in sr_parser.model_name:
            self.model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        else:
            self.model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(sr_parser.abs_model_path), strict=True)
        
        for _, v in self.model.named_parameters():
            v.requires_grad = True
        
        self.kernel = util_img.single2tensor4(k[..., np.newaxis])
        self.sigma = torch.tensor(sr_parser.noise_level_model).float().view([1, 1, 1, 1])
        
        self.model = self.model.to(self.device)
        self.kernel = self.kernel.to(self.device)
        self.sigma = self.sigma.to(self.device)
        
        self.xmax = xmax
        self.scale = sr_parser.sr_scale
        
        print("end initialize [inference model]")
        
    def inference(self, img_lq, w, h):
        self.model.eval()
        img_lq = util_img.single2tensor4(img_lq)
        img_lq = img_lq.to(self.device)
        img_e = self.model(img_lq, self.kernel, self.scale, self.sigma)
        img_e = util_img.tensor2uint(img_e, self.xmax)[:self.scale * w, :self.scale * h, ...]
        return img_e