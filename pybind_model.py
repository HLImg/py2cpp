# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/19 20:57:53
# @FileName:  pybind_model.py
# @Contact :  lianghao@whu.edu.cn

import cv2
import torch
import numpy as np

import python.utils.utils_sisr as util_sr
import python.utils.utils_image as util_img
import python.utils.utils_deblur as util_deblur

from scipy.io import savemat
from python.model.network_usrnet_v1 import USRNet as net

class Model:
    def __init__(self, k, sr_parser, xmax):
        print("start initialize [inference model]")
        torch.cuda.empty_cache()
        self.name = sr_parser.model_name
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
        
        self.xmax = xmax
        self.scale = sr_parser.sr_scale
        self.boarder = sr_parser.boarder_handling
        
        print("end initialize [inference model]")
        
    def inference(self, img_lq, i):
        
        # print("[DEBUG] start inference, the shape is ", img_lq.shape, f"py:arr {img_lq.min()}, {img_lq.max()}, mean = {img_lq.mean()}")
        savemat(f"test/test_{i}.mat", {"data": img_lq})
        
        # previous works
        xmin_lq, xmax_lq = img_lq.min(), img_lq.max()
        img_lq = np.float32(img_lq / self.xmax)
        w, h = img_lq.shape[:2]
        img = cv2.resize(img_lq, (self.scale * h, self.scale * w), interpolation=cv2.INTER_NEAREST) 
        img = util_deblur.wrap_boundary_liu(img, [int(np.ceil(self.scale * w / self.boarder + 2) * self.boarder),
                                                 int(np.ceil(self.scale * h / self.boarder + 2) * self.boarder)])
        img_wrap= util_sr.downsample_np(img, self.scale, center=False)
        img_wrap[:w, :h] = img_lq
        img_lq = np.float32(img_wrap)
        img_lq  = cv2.cvtColor(img_lq , cv2.COLOR_GRAY2RGB)
        img_lq = util_img.single2tensor4(img_lq)
        img_lq = img_lq.to(self.device)
        # print(img_lq.mean(), self.kernel.mean(), self.scale, self.sigma)
        self.model = self.model.to(self.device)
        self.kernel = self.kernel.to(self.device)
        self.sigma = self.sigma.to(self.device)
        self.model.eval()
        import pdb ; pdb.set_trace()
        torch.cuda.empty_cache()
        img_e = self.model(img_lq, self.kernel, self.scale, self.sigma)
        img_e = util_img.tensor2uint(img_e, self.xmax)[:self.scale * w, :self.scale * h, ...]
        img_e = img_e[:, :, 0]
        img_e = np.clip(img_e, xmin_lq, xmax_lq)
        # print("[DEBUG] finish inference, the shape is ", img_e.shape, f"min = {img_e.min()}, max = {img_e.max()},  mean = {img_e.mean()}")
        savemat(f"test/test_sr{i}.mat", {"data": img_e})
        return img_e
    