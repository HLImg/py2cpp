import cv2
import numpy as np
import torch
from method.super_resolution.utils import utils_sr
import random
# from models.network_usrnet import USRNet as net   # for pytorch version <= 1.7.1
from method.super_resolution.network.network_usrnet_v1 import USRNet  # for pytorch version >= 1.8.1
from method.super_resolution.network.network_rrdbnet import RRDBNet
from method.super_resolution.network.network_cmdsr import CMDSRnet
from method.super_resolution.network.network_realesrgan import RRDBNet_E, RealESRGANer
from method.super_resolution.network.network_swinir import SwinIR


class SR_Inference:
    def __init__(self, lq, use_method, img_type, ckpt_path, sr_scale):
        self.lq = lq
        self.scale = sr_scale
        self.img_type = img_type
        self.use_method = use_method
        self.ckpt = ckpt_path
        
        print(f"the shape of lq is {lq.shape}")
        print(f"use-method {use_method}, img_type {img_type}")
        print(f"scale = {self.scale}, ckpt is {self.ckpt}")