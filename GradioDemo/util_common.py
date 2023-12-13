# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/12/13 20:07:12
# @FileName:  util_common.py
# @Contact :  lianghao@whu.edu.cn

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
            

def upload_msi(files):
    img = gdal_read(files)
    img = np.transpose(img, (1, 2, 0))
    img = img / img.max()
    return img[:, :, :3], img

def upload_pan(files):
    img = gdal_read(files)
    img = img / img.max()
    return img