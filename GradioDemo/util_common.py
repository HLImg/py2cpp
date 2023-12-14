# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/12/13 20:07:12
# @FileName:  util_common.py
# @Contact :  lianghao@whu.edu.cn

import os
import time
import random
import string
import tempfile
import numpy as np
from osgeo import gdal

def gdal_read(path):
    data = gdal.Open(path)
    width = data.RasterXSize
    height = data.RasterYSize
    out = data.ReadAsArray(0, 0, width, height)
    return out

def savetiff(path, img):
    pixelWidth = 1.0
    pixelHeight = -1.0

    cols = img.shape[1]
    rows = img.shape[0]
    if len(img.shape) == 3:
        bands = img.shape[2]
    else:
        bands = 1
    originX = 0
    originY = 0
    driver = gdal.GetDriverByName('GTiff')

    outRaster = driver.Create(path, cols, rows, bands, gdal.GDT_UInt16)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    #开始写入
    if bands==1:
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(img[:, :,0])
    else:
        for i in range(bands):
            outRaster.GetRasterBand(i + 1).WriteArray(img[:,:,i])

def upload_ckpt(files):
    return files    

def upload_tif(files):
    img = gdal_read(files)
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    else:
        img = np.repeat(img[:, :, np.newaxis], 4, axis=2)
        
    show_img = img.copy()
    show_img = show_img[:, :, :3]
    show_img = show_img / show_img.max()
    
    return show_img, files

def is_pan(image):
    if (image[0] == image[2]).all():
        return image[:, :, 0]
    else:
        return image

def get_tmp_name(length=40):
    letters_and_digits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def get_tmp_dir(length=40):
    dir = tempfile.gettempdir()
    random_string = get_tmp_name()
    temp_dir_path = os.path.join(dir, random_string)
    
    if not os.path.exists(temp_dir_path):
        os.mkdir(temp_dir_path)
    
    return temp_dir_path
    
def download_tif(image):
    temp_dir = get_tmp_dir(40)
    temp_name = get_tmp_name(10) + ".tiff"
    temp_file_path = os.path.join(temp_dir, temp_name)
    
    savetiff(temp_file_path, image)
    
    print("保存文件成功")
    return temp_file_path