# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/19 20:57:53
# @FileName:  pybind_model.py
# @Contact :  lianghao@whu.edu.cn

import os
import cv2
import torch
import numpy as np
import xml.dom.minidom

import python.utils.utils_sisr as util_sr
import python.utils.utils_image as util_img
import python.utils.utils_deblur as util_deblur

from scipy.io import savemat
from python.model.network_usrnet_v1 import USRNet as net

class Model:
    def __init__(self, k, sr_parser, h, w, channel=1):
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
        
        # self.xmax = xmax
        self.scale = sr_parser.sr_scale
        self.boarder = sr_parser.boarder_handling

        self.res = np.zeros((h * self.scale, w * self.scale, channel), dtype=np.uint16)
        self.last_image = None
        print("end initialize [inference model]")
    
    def calWeight(self, d, k):
        '''
        :param d: 融合重叠部分直径
        :param k: 融合计算权重参数
        '''
        x = np.arange(-d / 2, d / 2)
        y = 1 / (1 + np.exp(-k * x))
        return y
    
    def set(self, sr_xheight, sr_xwidth, sr_oriheight, 
            sr_oriwidth, sr_overlap, sr_height, sr_width,
            overlap_downedge, overlap_rightedge):
        
        self.sr_xheight = sr_xheight
        self.sr_xwidth = sr_xwidth
        self.sr_oriheight = sr_oriheight
        self.sr_oriwidth = sr_oriwidth
        self.sr_overlap = sr_overlap
        self.sr_height = sr_height
        self.sr_width = sr_width

        w_lr = self.calWeight(sr_overlap, 0.1)
        self.w_left_right0 = np.tile(w_lr, (sr_xheight, 1))
        self.w_left_right = np.tile(w_lr, (sr_oriheight, 1))
        self.w_left_right_downedge = np.tile(w_lr, (overlap_downedge * self.scale - sr_overlap, 1))
        self.w_left_right_corner = np.tile(w_lr, (sr_overlap, 1))

        w_ud = self.calWeight(sr_overlap, 0.1)
        self.w_ud = np.reshape(w_ud, (sr_overlap, 1))
        self.w_up_down = np.tile(self.w_ud, (1, sr_oriwidth))
        self.w_up_down_rightedge = np.tile(self.w_ud, (1, overlap_rightedge * self.scale))

        print("[model-set] finish")

        
    def inference(self, img_lq, xmax, c, i, num_row, num_col):
        img_lq = img_lq.squeeze()
        # print("[DEBUG] start inference, the shape is ", img_lq.shape, f"py:arr {img_lq.min()}, {img_lq.max()}, mean = {img_lq.mean()}")
        # savemat(f"test/test_{i}.mat", {"data": img_lq})
        
        # previous works
        xmin_lq, xmax_lq = img_lq.min(), img_lq.max()
        img_lq = np.float32(img_lq / xmax)
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
        # import pdb ; pdb.set_trace()
        torch.cuda.empty_cache()
        with torch.no_grad():
            img_e = self.model(img_lq, self.kernel, self.scale, self.sigma)
        img_e = util_img.tensor2uint(img_e, xmax)[:self.scale * w, :self.scale * h, ...]
        img_e = img_e[:, :, 0]
        img_e = np.clip(img_e, xmin_lq, xmax_lq)
        # print("[DEBUG] finish inference, the shape is ", img_e.shape, f"min = {img_e.min()}, max = {img_e.max()},  mean = {img_e.mean()}")
        # savemat(f"test/test_sr{i}.mat", {"data": img_e})

        numr = i % num_row
        numcol = i // num_row

        if numcol == 0:
            if numr == 0:
                self.res[:self.sr_xheight, :self.sr_oriwidth, c] = img_e[:, :self.sr_oriwidth]
            elif numr < num_row - 1:
                self.res[:self.sr_xheight, (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_overlap), c] = np.uint16(
                    (1 - self.w_left_right0) * self.last_image[:, self.sr_oriwidth:self.sr_xwidth] + self.w_left_right0 * img_e[:, : self.sr_overlap])
                self.res[:self.sr_xheight, 
                         (numr * self.sr_oriwidth + self.sr_overlap):(numr + 1) * self.sr_oriwidth, c] = img_e[:, self.sr_overlap:self.sr_oriwidth]
            elif numr == num_row - 1:
                self.res[:self.sr_xheight, (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_overlap), c] = np.uint16(
                    (1 - self.w_left_right0) * self.last_image[:, self.sr_oriwidth:self.sr_xwidth] + self.w_left_right0 * img_e[:,
                                                                                                         :self.sr_overlap]
                )
                self.res[:self.sr_xheight, (numr * self.sr_oriwidth + self.sr_overlap):self.sr_width, c] = img_e[:, self.sr_overlap:]
        
        elif numcol < num_col - 1:
            if numr == 0 :
                patch_up = self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap), :self.sr_oriwidth, c]
                self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap), :self.sr_oriwidth, c] = np.uint16(
                            (1 - self.w_up_down) * patch_up + self.w_up_down * img_e[:self.sr_overlap, :self.sr_oriwidth])
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):(numcol * self.sr_oriheight + self.sr_xheight),
                        :self.sr_oriwidth, c] = img_e[self.sr_overlap:, :self.sr_oriwidth]
            
            elif numr < num_row - 1:
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):(numcol * self.sr_oriheight + self.sr_xheight),
                        (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_overlap), c] = np.uint16(
                            (1 - self.w_left_right) * self.last_image[self.sr_overlap:, 
                                                                      self.sr_oriwidth:] + self.w_left_right * img_e[
                                                                                                        self.sr_overlap:,
                                                                                                        :self.sr_overlap])
                img = img_e[:self.sr_overlap, :self.sr_oriwidth]
                img[:, :self.sr_overlap] = np.uint16((1 - self.w_left_right_corner) * self.last_image[:self.sr_overlap,
                                                                                    self.sr_oriwidth:] + self.w_left_right_corner * img_e[
                                                                                                                          :self.sr_overlap,
                                                                                                                          :self.sr_overlap])
                patch_up = self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                                   (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_oriwidth), c]
                self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                        (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_oriwidth), c] = np.uint16(
                            (1 - self.w_up_down) * patch_up + self.w_up_down * img)
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):(numcol * self.sr_oriheight + self.sr_xheight),
                        (numr * self.sr_oriwidth + self.sr_overlap):(numr * self.sr_oriwidth + self.sr_oriwidth), c] = img_e[self.sr_overlap:,
                                                                                                self.sr_overlap:self.sr_oriwidth]
            elif numr == num_row - 1:
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):(numcol * self.sr_oriheight + self.sr_xheight),
                        (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_overlap), c] = np.uint16(
                            (1 - self.w_left_right) * self.last_image[self.sr_overlap:, 
                                                                      self.sr_oriwidth:] + self.w_left_right * img_e[
                                                                                                        self.sr_overlap:,
                                                                                                        :self.sr_overlap])
                img = img_e[:self.sr_overlap, :]
                img[:, :self.sr_overlap] = np.uint16((1 - self.w_left_right_corner) * self.last_image[:self.sr_overlap,
                                                            self.sr_oriwidth:] + self.w_left_right_corner * img_e[:self.sr_overlap,
                                                                                                                  :self.sr_overlap])
                patch_up = self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                                   (numr * self.sr_oriwidth):self.sr_width, c]
                self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                        (numr * self.sr_oriwidth):self.sr_width, c] = np.uint16(
                            (1 - self.w_up_down_rightedge) * patch_up + self.w_up_down_rightedge * img)
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):(numcol * self.sr_oriheight + self.sr_xheight),
                        (numr * self.sr_oriwidth + self.sr_overlap):self.sr_width, c] = img_e[self.sr_overlap:, self.sr_overlap:]
            
        elif numcol == numcol - 1:
            if numr == 0:
                patch_up = self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap), :self.sr_oriwidth]
                self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap), :self.sr_oriwidth, c] = np.uint16(
                            (1 - self.w_up_down) * patch_up + self.w_up_down * img_e[:self.sr_overlap, :self.sr_oriwidth])
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):self.sr_height, :self.sr_oriwidth, c] = img_e[self.sr_overlap:,
                                                                                             :self.sr_oriwidth]
            elif numr < num_row - 1:
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):self.sr_height,
                        (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_overlap), c] = np.uint16(
                            (1 - self.w_left_right_downedge) * self.last_image[self.sr_overlap:,
                                    self.sr_oriwidth:] + self.w_left_right_downedge * img_e[self.sr_overlap:,
                                                                                                  :self.sr_overlap]
                        )
                img = img_e[:self.sr_overlap, :self.sr_oriwidth]
                img[:, :self.sr_overlap] = np.uint16((1 - self.w_left_right_corner) * self.last_image[:self.sr_overlap,
                                                    self.sr_oriwidth:] + self.w_left_right_corner * img_e[:self.sr_overlap,
                                                                                                          :self.sr_overlap])
                patch_up = self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                                   (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_oriwidth), c]
                self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                        (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_oriwidth), c] = np.uint16(
                            (1 - self.w_up_down) * patch_up + self.w_up_down * img)
                
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):self.sr_height,
                        (numr * self.sr_oriwidth + self.sr_overlap):
                        (numr * self.sr_oriwidth + self.sr_oriwidth), c] = img_e[self.sr_overlap:,
                                                                            self.sr_overlap:self.sr_oriwidth]
            elif numr == num_row - 1:
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):self.sr_height,
                        (numr * self.sr_oriwidth):(numr * self.sr_oriwidth + self.sr_overlap)] = np.uint16(
                            (1 - self.w_left_right_downedge) * self.last_image[self.sr_overlap:,
                                        self.sr_oriwidth:] + self.w_left_right_downedge * img_e[self.sr_overlap:,
                                                                                                  :self.sr_overlap]
                        )
                img = img_e[:self.sr_overlap, :]
                img[:, :self.sr_overlap] = np.uint16((1 - self.w_left_right_corner) * self.last_image[:self.sr_overlap,
                                            self.sr_oriwidth:] + self.w_left_right_corner * img_e[:self.sr_overlap,
                                                                                                   :self.sr_overlap])
                patch_up = self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                                   (numr * self.sr_oriwidth):self.sr_width, c]
                self.res[(numcol * self.sr_oriheight):(numcol * self.sr_oriheight + self.sr_overlap),
                        (numr * self.sr_oriwidth):self.sr_width, c] = np.uint16(
                            (1 - self.w_up_down_rightedge) * patch_up + self.w_up_down_rightedge * img)
                self.res[(numcol * self.sr_oriheight + self.sr_overlap):self.sr_height,
                        (numr * self.sr_oriwidth + self.sr_overlap):self.sr_width] = img_e[self.sr_overlap:, self.sr_overlap:]
        self.last_image = img_e

        savemat(f"test/test_sr{i}.mat", {"data": self.res})
    
    def save_tiff(self, path):
        self.res = self.res.squeeze()
        util_img.savetiff(path, self.res)
        del self.res

    def finish(self, filepath_abs_SRPAN):
        doc = xml.dom.minidom.Document()
        item = doc.createElement('DOCUMENT')
        item.setAttribute('content_method', "full")
        doc.appendChild(item)
        flag = doc.createElement('flag')
        flag_text = doc.createTextNode('over')
        flag.appendChild(flag_text)
        item.appendChild(flag)
        xml_save_path = os.path.join(filepath_abs_SRPAN, 'flag.xml')
        f = open(xml_save_path, 'w')
        doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()