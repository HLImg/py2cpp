# -*- coding: utf-8 -*-
# @Time    : 11/19/23 10:56 PM
# @File    : source.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os.path
import os
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,200).__str__()
# os.environ["OPENCV_IO_MAX_IMAGE_WIDTH"] = pow(2,200).__str__()
# os.environ["OPENCV_IO_MAX_IMAGE_HEIGHT"] = pow(2,200).__str__()
import cv2
# import logging
import xml.dom.minidom
import numpy as np
import torch
from utils import utils_deblur
# from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from models.network_usrnet_v1 import USRNet as net  # for pytorch version >= 1.8.1
import math
import scipy.spatial.transform._rotation_groups
import scipy.special.cython_special
import warnings

warnings.filterwarnings('ignore')
import sys, time
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument("--input_pan", type=str, default="data/PAM/PAN.tiff", help="The filePath of PAN")
parser.add_argument("--input_mux", type=str, default="data/MSI/MSI.tiff", help="The filePath of MUX")
parser.add_argument("--out_pan", type=str, default="data/PAM/PAN.tiff", help="The output filePath of PAN")
parser.add_argument("--out_mux", type=str, help="The out filepath of MUX")

args = parser.parse_args(args=sys.argv[1:])
'''
parser = argparse.ArgumentParser()
parser.add_argument("--input_xml", type=str, help="The path of input xml")
args = parser.parse_args(args=sys.argv[1:])


def main(input_xml):
    #
    dom0 = xml.dom.minidom.parse(input_xml)
    root0 = dom0.documentElement

    input_path_cut = root0.getElementsByTagName('input_mux')
    cut_input_path = input_path_cut[0].firstChild.data

    input_path_cut_PAN = root0.getElementsByTagName('input_pan')
    cut_input_path_PAN = input_path_cut_PAN[0].firstChild.data

    SRPAN_tmp = root0.getElementsByTagName('output_pan')
    SRPAN = SRPAN_tmp[0].firstChild.data

    SRMUX_tmp = root0.getElementsByTagName('output_mux')
    SRMUX = SRMUX_tmp[0].firstChild.data
    # 先左右再上下
    # TODO : c++实现
    output = [SRPAN, SRMUX]
    # 读取xml文件
    dom = xml.dom.minidom.parse('./SuperOption.xml')
    root = dom.documentElement
    # print(cut_input_path, cut_input_path_PAN)
    # 读取多光谱输入路径
    '''
    input_path_cut = root.getElementsByTagName('abs_path_input')
    cut_input_path = input_path_cut[0].firstChild.data
    '''
    # 获取文件名
    filepath_abs, tempfilename_abs = os.path.split(cut_input_path)
    filename_abs_name, extension_abs = os.path.splitext(tempfilename_abs)
    '''
    # 读取全色输入路径
    input_path_cut_PAN = root.getElementsByTagName('abs_path_input_PAN')
    cut_input_path_PAN = input_path_cut_PAN[0].firstChild.data
   '''
    # 获取文件名
    filepath_abs_PAN, tempfilename_abs_PAN = os.path.split(cut_input_path_PAN)
    filename_abs_name_PAN, extension_abs_PAN = os.path.splitext(tempfilename_abs_PAN)

    # 读取log保存路径
    filepath_abs_SRPAN, tempfilename_abs_SRPAN = os.path.split(SRPAN)
    filename_abs_name_SRPAN, extension_abs_SRPAN = os.path.splitext(tempfilename_abs_SRPAN)
    '''
    log_save_path = filepath_abs_SRPAN
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    # log文件命名
    save_log_name = filename_abs_name_SRPAN[:-3] + '.log'
    log_save_path = os.path.join(log_save_path, save_log_name)
    # 写入log
    utils_logger.logger_info('cnn_sr_log', log_path=log_save_path)
    logger = logging.getLogger('cnn_sr_log')
    '''
    # 获取最后输出路径
    """
    output_path_sr = root.getElementsByTagName('abs_path_output')
    sr_output_path = output_path_sr[0].firstChild.data
    """
    # if not os.path.exists(sr_output_path):
    #   os.makedirs(sr_output_path)
    '''
    logger.info('{:>16s} '.format('START!'))
    logger.info('{:>16s} : {:s}'.format('Input Path', cut_input_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', sr_output_path))
    '''
    # 获取模型所在路径
    mode_path = root.getElementsByTagName('mode_path')
    abs_model_path = mode_path[0].firstChild.data
    # 获取模型名字和路径
    model_path, model_name = os.path.split(abs_model_path)

    # 获取超分倍数
    mode_scale = root.getElementsByTagName('mode_scale')
    mode_scale_str = mode_scale[0].firstChild.data
    sf = int(mode_scale_str)

    # 获取噪声等级
    noise_level_model = root.getElementsByTagName('noise_level_model')
    noise_level_img_str = noise_level_model[0].firstChild.data
    noise_level_model = float(noise_level_img_str)

    # 获取核1
    kernel_width_default_c_1 = root.getElementsByTagName('kernel_width_default_c_1')
    kernel_width_default_c_1_str = kernel_width_default_c_1[0].firstChild.data
    kernel_width_default_c_1 = float(kernel_width_default_c_1_str)

    # 获取核2
    kernel_width_default_c_2 = root.getElementsByTagName('kernel_width_default_c_2')
    kernel_width_default_c_2_str = kernel_width_default_c_2[0].firstChild.data
    kernel_width_default_c_2 = float(kernel_width_default_c_2_str)

    # 获取核3
    kernel_width_default_c_3 = root.getElementsByTagName('kernel_width_default_c_3')
    kernel_width_default_c_3_str = kernel_width_default_c_3[0].firstChild.data
    kernel_width_default_c_3 = float(kernel_width_default_c_3_str)

    # 获取核4
    kernel_width_default_c_4 = root.getElementsByTagName('kernel_width_default_c_4')
    kernel_width_default_c_4_str = kernel_width_default_c_4[0].firstChild.data
    kernel_width_default_c_4 = float(kernel_width_default_c_4_str)
    # 获取滤波
    fspecial_filter_type = root.getElementsByTagName('fspecial_filter_type')
    fspecial_filter_type = fspecial_filter_type[0].firstChild.data
    # 获取滤波窗口
    fspecial_filter_hsize = root.getElementsByTagName('fspecial_filter_hsize')
    fspecial_filter_hsize_str = fspecial_filter_hsize[0].firstChild.data
    fspecial_filter_hsize = int(fspecial_filter_hsize_str)
    # boarder_handling
    boarder_handling = root.getElementsByTagName('boarder_handling')
    boarder_handling_str = boarder_handling[0].firstChild.data
    boarder_handling = int(boarder_handling_str)

    # 读取裁切尺寸
    cut_size_height = root.getElementsByTagName('cut_size_height')
    cut_size_height_str = cut_size_height[0].firstChild.data
    xheight = int(cut_size_height_str)
    cut_size_width = root.getElementsByTagName('cut_size_width')
    cut_size_width_str = cut_size_width[0].firstChild.data
    xwidth = int(cut_size_width_str)
    # TODO
    for ii in range(2):
        if ii == 0:
            # TODO
            IMG = util.gdal_read(cut_input_path_PAN)
            if len(IMG.shape) == 3:
                IMG = np.transpose(IMG, (1, 2, 0))
            elif len(IMG.shape) == 2:
                IMG = np.expand_dims(IMG, -1)
            bands = IMG.shape[2]
            print("Start SR-PAN .... ")
            # logger.info('{:>16s} '.format('Start SR-PAN ....'))
        else:
            IMG = util.gdal_read(cut_input_path)
            if len(IMG.shape) == 3:
                IMG = np.transpose(IMG, (1, 2, 0))
            elif len(IMG.shape) == 2:
                IMG = np.expand_dims(IMG, -1)
            height0, width0, bands = IMG.shape
            print("Start SR-MUX ....")
            # logger.info('{:>16s} '.format('Finish SR-PAN ....'))
            # logger.info('{:>16s} '.format('Start SR-MUX ....'))
            SMUX = np.zeros((height0 * sf, width0 * sf, bands), np.uint16)
        for iii in range(bands):
            print('{:s}{:d} '.format('Start band', iii + 1))
            cut_img = IMG[:, :, iii]
            # 获取全色图像的高、宽
            height, width = cut_img.shape[:2]
            # 获取一景全色图像的最大值
            xmax = cut_img.max()

            # 确定裁切的行列块数
            overlap = 200
            oriheight = xheight - overlap
            oriwidth = xwidth - overlap
            max_heinum = math.ceil((height - xheight) / oriheight)
            max_widnum = math.ceil((width - xwidth) / oriwidth)

            if height - max_heinum * oriheight > 0:
                xheinum1 = max_heinum + 1
            else:
                xheinum1 = max_heinum
            if width - max_widnum * oriwidth > 0:
                xwidnum1 = max_widnum + 1
            else:
                xwidnum1 = max_widnum

            # 开始裁剪
            i = 0
            smalldata_tensor = {}

            for h in range(max_heinum):
                for w in range(max_widnum):
                    crop_img = cut_img[(oriheight * h):(oriheight * h + xheight),
                               (oriwidth * w):(oriwidth * w + xwidth)]  # 从一景图像中选取patch
                    i = i + 1
                    smalldata_tensor[i] = crop_img

                    if w == (max_widnum - 1):  # 裁切图右边的不满足设置patch大小的剩余部分
                        if width - max_widnum * oriwidth > 0:
                            crop_img = cut_img[(oriheight * h):(oriheight * h + xheight), (oriwidth * (w + 1)):width]
                            overlap_rightedge = width - oriwidth * (w + 1)
                            i = i + 1
                            smalldata_tensor[i] = crop_img

            if height - max_heinum * oriheight > 0:  # 裁切图下面的不满足设置patch大小的剩余部分
                for w in range(max_widnum):
                    crop_img = cut_img[(oriheight * max_heinum):height, (oriwidth * w):(oriwidth * w + xwidth)]
                    overlap_downedge = height - (oriheight * max_heinum)
                    i = i + 1
                    smalldata_tensor[i] = crop_img

                    # 裁切右下角的剩余部分
                    if w == (max_widnum - 1):
                        if width - max_widnum * oriwidth > 0:
                            crop_img = cut_img[(oriheight * max_heinum):height, (oriwidth * (w + 1)):width]
                            i = i + 1
                            smalldata_tensor[i] = crop_img
            del cut_img

            # TODO

            """超分部分"""
            # ----------------------------------------
            # Preparation
            # ----------------------------------------
            kernel_width_default_x1234 = [kernel_width_default_c_1, kernel_width_default_c_2, kernel_width_default_c_3,
                                          kernel_width_default_c_4]  # default Gaussian kernel widths of clean/sharp images for x1, x2, x3, x4
            kernel_width = kernel_width_default_x1234[sf - 1]
            k = utils_deblur.fspecial(fspecial_filter_type, fspecial_filter_hsize, kernel_width)
            k = sr.shift_pixel(k, sf)  # shift the kernel

            # TODO: CPP
            k /= np.sum(k)

            # TODO: CPP
            kernel = util.single2tensor4(k[..., np.newaxis])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 确定使用gpu还是cpu
            torch.cuda.empty_cache()

            # TODO

            # ----------------------------------------
            # load model
            # ----------------------------------------
            if 'tiny' in model_name:
                model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
            else:
                model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
            model.load_state_dict(torch.load(abs_model_path), strict=True)  # 加载模型参数
            model.eval()  # 模型加载完成的标记
            for key, v in model.named_parameters():
                v.requires_grad = False

            model = model.to(device)

            # TODO
            dic_len = len(smalldata_tensor)

            boarder = boarder_handling  # default setting for kernel size 25x25
            sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])

            """拼接"""
            sr_xheight = xheight * sf
            sr_xwidth = xwidth * sf
            sr_oriwidth = oriwidth * sf
            sr_oriheight = oriheight * sf
            sr_overlap = overlap * sf
            sr_height = height * sf
            sr_width = width * sf

            mage = np.zeros((height * sf, width * sf), np.uint16)  # 设置大图初始值
            numrow = xwidnum1  # 每行的张数
            numcolunm = xheinum1  # 每列的张数
            w_lr = util.calWeight(sr_overlap, 0.1)
            w_left_right0 = np.tile(w_lr, (sr_xheight, 1))
            w_left_right = np.tile(w_lr, (sr_oriheight, 1))
            w_left_right_downedge = np.tile(w_lr, (overlap_downedge * sf - sr_overlap, 1))
            w_left_right_corner = np.tile(w_lr, (sr_overlap, 1))

            w_ud = util.calWeight(sr_overlap, 0.1)
            w_ud = np.reshape(w_ud, (sr_overlap, 1))
            w_up_down = np.tile(w_ud, (1, sr_oriwidth))
            w_up_down_rightedge = np.tile(w_ud, (1, overlap_rightedge * sf))

            # TODO
            for i in range(dic_len):  ###patch超分
                if i % 10 == 0:
                    # logger.info('{:>16s} : {:d}/{:d}'.format('finished', i, dic_len))
                    print("finished:{0}/{1}".format(i, dic_len))
                # ------------------------------------
                # (1) 加载及处理patch
                # ------------------------------------
                img_L = smalldata_tensor[i + 1]  # 读取patch
                xmin_L = img_L.min()
                xmax_L = img_L.max()
                # img_L = cv2.cvtColor(img_L, cv2.COLOR_GRAY2RGB)  # GGG
                img_L = util.uint2single(img_L, xmax)  # patch归一化

                w, h = img_L.shape[:2]  # patch尺寸
                img = cv2.resize(img_L, (sf * h, sf * w), interpolation=cv2.INTER_NEAREST)  # patch上采样
                img = utils_deblur.wrap_boundary_liu(img, [int(np.ceil(sf * w / boarder + 2) * boarder),
                                                           int(np.ceil(sf * h / boarder + 2) * boarder)])

                img_wrap = sr.downsample_np(img, sf, center=False)  # 下采样
                img_wrap[:w, :h] = img_L
                img_L = np.float32(img_wrap)  ##3通道
                img_L = cv2.cvtColor(img_L, cv2.COLOR_GRAY2RGB)

                # TODO
                img_L = util.single2tensor4(img_L)  ##转成tensor数据类型
                img_L = img_L.to(device)  ##将数据导入gpu显存

                # ------------------------------------
                # (2) 输出超分后patch
                # ------------------------------------
                [img_L, kernel, sigma] = [el.to(device) for el in [img_L, kernel, sigma]]  ##将参数导入显存计算
                img_E = model(img_L, kernel, sf, sigma)  ##patch超分
                img_E = util.tensor2uint(img_E, xmax)[:sf * w, :sf * h, ...]  ##patch数据类型转为uint16

                # TODO
                img_E = img_E[:, :, 0]
                img_E[img_E < xmin_L] = xmin_L
                img_E[img_E > xmax_L] = xmax_L

                numr = i % numrow
                numcol = i // numrow
                #  ## 将超分后patch填充进大图
                if numcol == 0:
                    if numr == 0:
                        mage[:sr_xheight, :sr_oriwidth] = img_E[:, :sr_oriwidth]
                    elif numr < (numrow - 1):
                        mage[:sr_xheight, (numr * sr_oriwidth):(numr * sr_oriwidth + sr_overlap)] = np.uint16(
                            (1 - w_left_right0) * last_image[:, sr_oriwidth:sr_xwidth] + w_left_right0 * img_E[:,
                                                                                                         :sr_overlap])
                        mage[:sr_xheight, (numr * sr_oriwidth + sr_overlap):(numr + 1) * sr_oriwidth] = img_E[:,
                                                                                                        sr_overlap:sr_oriwidth]
                    elif numr == (numrow - 1):
                        mage[:sr_xheight, (numr * sr_oriwidth):(numr * sr_oriwidth + sr_overlap)] = np.uint16(
                            (1 - w_left_right0) * last_image[:, sr_oriwidth:sr_xwidth] + w_left_right0 * img_E[:,
                                                                                                         :sr_overlap])
                        mage[:sr_xheight, (numr * sr_oriwidth + sr_overlap):sr_width] = img_E[:, sr_overlap:]
                elif numcol < (numcolunm - 1):
                    if numr == 0:
                        patch_up = mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap), :sr_oriwidth]
                        mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap), :sr_oriwidth] = np.uint16(
                            (1 - w_up_down) * patch_up + w_up_down * img_E[:sr_overlap, :sr_oriwidth])
                        mage[(numcol * sr_oriheight + sr_overlap):(numcol * sr_oriheight + sr_xheight),
                        :sr_oriwidth] = img_E[sr_overlap:, :sr_oriwidth]
                    elif numr < (numrow - 1):
                        mage[(numcol * sr_oriheight + sr_overlap):(numcol * sr_oriheight + sr_xheight),
                        (numr * sr_oriwidth):(numr * sr_oriwidth + sr_overlap)] = np.uint16(
                            (1 - w_left_right) * last_image[sr_overlap:, sr_oriwidth:] + w_left_right * img_E[
                                                                                                        sr_overlap:,
                                                                                                        :sr_overlap])
                        img = img_E[:sr_overlap, :sr_oriwidth]
                        img[:, :sr_overlap] = np.uint16((1 - w_left_right_corner) * last_image[:sr_overlap,
                                                                                    sr_oriwidth:] + w_left_right_corner * img_E[
                                                                                                                          :sr_overlap,
                                                                                                                          :sr_overlap])
                        patch_up = mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                                   (numr * sr_oriwidth):(numr * sr_oriwidth + sr_oriwidth)]
                        mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                        (numr * sr_oriwidth):(numr * sr_oriwidth + sr_oriwidth)] = np.uint16(
                            (1 - w_up_down) * patch_up + w_up_down * img)
                        mage[(numcol * sr_oriheight + sr_overlap):(numcol * sr_oriheight + sr_xheight),
                        (numr * sr_oriwidth + sr_overlap):(numr * sr_oriwidth + sr_oriwidth)] = img_E[sr_overlap:,
                                                                                                sr_overlap:sr_oriwidth]
                    elif numr == (numrow - 1):
                        mage[(numcol * sr_oriheight + sr_overlap):(numcol * sr_oriheight + sr_xheight),
                        (numr * sr_oriwidth):(numr * sr_oriwidth + sr_overlap)] = np.uint16(
                            (1 - w_left_right) * last_image[sr_overlap:, sr_oriwidth:] + w_left_right * img_E[
                                                                                                        sr_overlap:,
                                                                                                        :sr_overlap])
                        img = img_E[:sr_overlap, :]
                        img[:, :sr_overlap] = np.uint16((1 - w_left_right_corner) * last_image[:sr_overlap,
                                                                                    sr_oriwidth:] + w_left_right_corner * img_E[
                                                                                                                          :sr_overlap,
                                                                                                                          :sr_overlap])
                        patch_up = mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                                   (numr * sr_oriwidth):sr_width]
                        mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                        (numr * sr_oriwidth):sr_width] = np.uint16(
                            (1 - w_up_down_rightedge) * patch_up + w_up_down_rightedge * img)
                        mage[(numcol * sr_oriheight + sr_overlap):(numcol * sr_oriheight + sr_xheight),
                        (numr * sr_oriwidth + sr_overlap):sr_width] = img_E[sr_overlap:, sr_overlap:]
                elif numcol == (numcolunm - 1):
                    if numr == 0:
                        patch_up = mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap), :sr_oriwidth]
                        mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap), :sr_oriwidth] = np.uint16(
                            (1 - w_up_down) * patch_up + w_up_down * img_E[:sr_overlap, :sr_oriwidth])
                        mage[(numcol * sr_oriheight + sr_overlap):sr_height, :sr_oriwidth] = img_E[sr_overlap:,
                                                                                             :sr_oriwidth]
                    elif numr < (numrow - 1):
                        mage[(numcol * sr_oriheight + sr_overlap):sr_height,
                        (numr * sr_oriwidth):(numr * sr_oriwidth + sr_overlap)] = np.uint16(
                            (1 - w_left_right_downedge) * last_image[sr_overlap:,
                                                          sr_oriwidth:] + w_left_right_downedge * img_E[sr_overlap:,
                                                                                                  :sr_overlap])
                        img = img_E[:sr_overlap, :sr_oriwidth]
                        img[:, :sr_overlap] = np.uint16((1 - w_left_right_corner) * last_image[:sr_overlap,
                                                                                    sr_oriwidth:] + w_left_right_corner * img_E[
                                                                                                                          :sr_overlap,
                                                                                                                          :sr_overlap])
                        patch_up = mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                                   (numr * sr_oriwidth):(numr * sr_oriwidth + sr_oriwidth)]
                        mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                        (numr * sr_oriwidth):(numr * sr_oriwidth + sr_oriwidth)] = np.uint16(
                            (1 - w_up_down) * patch_up + w_up_down * img)
                        mage[(numcol * sr_oriheight + sr_overlap):sr_height,
                        (numr * sr_oriwidth + sr_overlap):(numr * sr_oriwidth + sr_oriwidth)] = img_E[sr_overlap:,
                                                                                                sr_overlap:sr_oriwidth]
                    elif numr == (numrow - 1):
                        mage[(numcol * sr_oriheight + sr_overlap):sr_height,
                        (numr * sr_oriwidth):(numr * sr_oriwidth + sr_overlap)] = np.uint16(
                            (1 - w_left_right_downedge) * last_image[sr_overlap:,
                                                          sr_oriwidth:] + w_left_right_downedge * img_E[sr_overlap:,
                                                                                                  :sr_overlap])
                        img = img_E[:sr_overlap, :]
                        img[:, :sr_overlap] = np.uint16((1 - w_left_right_corner) * last_image[:sr_overlap,
                                                                                    sr_oriwidth:] + w_left_right_corner * img_E[
                                                                                                                          :sr_overlap,
                                                                                                                          :sr_overlap])
                        patch_up = mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                                   (numr * sr_oriwidth):sr_width]
                        mage[(numcol * sr_oriheight):(numcol * sr_oriheight + sr_overlap),
                        (numr * sr_oriwidth):sr_width] = np.uint16(
                            (1 - w_up_down_rightedge) * patch_up + w_up_down_rightedge * img)
                        mage[(numcol * sr_oriheight + sr_overlap):sr_height,
                        (numr * sr_oriwidth + sr_overlap):sr_width] = img_E[sr_overlap:, sr_overlap:]
                last_image = img_E
            if ii == 1:
                SMUX[:, :, iii] = mage
            # SMUX.append(mage)
        del IMG
        # SMUX = np.uint16(SMUX)
        # SMUX = np.transpose(SMUX, (1, 2, 0))

        # dstimage_path = os.path.join(sr_output_path[ii], tempfilename[ii]) ##设置大图存储路径
        # dstimage_path = os.path.join(sr_output_path, tempfilename_abs)  ##设置大图存储路径
        if ii == 0:
            mage = np.expand_dims(mage, -1)
            util.savetiff(output[ii], mage)  ####  存储大图
            del mage
        elif ii == 1:
            util.savetiff(output[ii], SMUX)
            del SMUX

    #
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
    #
    print('Finish!')
    # logger.info('{:>16s} '.format('Finish!'))


if __name__ == '__main__':
    # args = parser.parse_args(args=[])
    start = time.time()
    main(args.input_xml)
    end = time.time()

    print("Total : {0} min".format((end - start) / 60))
