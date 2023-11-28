# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/11/19 19:53:40
# @FileName:  pybind_xml.py
# @Contact :  lianghao@whu.edu.cn

import os
import numpy as np
import xml.dom.minidom
import python.utils.utils_sisr as util_sr
import python.utils.utils_image as util_img
import python.utils.utils_deblur as util_deblur


class XMLParserInput:
    def __init__(self, input_xml) -> None:
        print("start parsing the XML file of the [input data]")
        dom0 = xml.dom.minidom.parse(input_xml)
        self.root0 = dom0.documentElement
        
        input_path_cut = self.root0.getElementsByTagName('input_mux')
        self.inp_mux = input_path_cut[0].firstChild.data
        
        _, tempfilename_abs = os.path.split(self.inp_mux)
        self.filename_abs_name, self.extension_abs = os.path.splitext(tempfilename_abs)
        
        input_path_cut_PAN = self.root0.getElementsByTagName('input_pan')
        self.inp_pan = input_path_cut_PAN[0].firstChild.data
        
        _, tempfilename_abs_PAN = os.path.split(self.inp_pan)
        self.filename_abs_name_PAN, self.extension_abs_PAN = os.path.splitext(tempfilename_abs_PAN)
        
        SRPAN_tmp = self.root0.getElementsByTagName('output_pan')
        self.out_pan = SRPAN_tmp[0].firstChild.data
        
        self.filepath_abs_SRPAN, tempfilename_abs_SRPAN = os.path.split(self.out_pan)
        self.filename_abs_name_SRPAN, self.extension_abs_SRPAN = os.path.splitext(tempfilename_abs_SRPAN)
        
        SRMUX_tmp = self.root0.getElementsByTagName('output_mux')
        self.out_mux = SRMUX_tmp[0].firstChild.data
        
        self.output = [self.out_pan, self.out_mux]
        
        print("end parsing the XML file of the [input data]")
    
    def save_tiff(self, mage, index):
        if index == 0:
            mage = np.expand_dims(mage, -1)
            util_img.savetiff(self.output[index], mage)
        else:
            util_img.savetiff(self.output[index], mage)
    
    def finish(self):
        doc = xml.dom.minidom.Document()
        item = doc.createElement('DOCUMENT')
        item.setAttribute('content_method', "full")
        doc.appendChild(item)
        flag = doc.createElement('flag')
        flag_text = doc.createTextNode('over')
        flag.appendChild(flag_text)
        item.appendChild(flag)
        xml_save_path = os.path.join(self.filepath_abs_SRPAN, 'flag.xml')
        f = open(xml_save_path, 'w')
        doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()
        
        
class XMLParserSR:
    def __init__(self, input_xml) -> None:
        print("start parsing the XML file of the [super resolution]")
        dom = xml.dom.minidom.parse(input_xml)
        self.root = dom.documentElement
        
        mode_path = self.root.getElementsByTagName('mode_path')
        self.abs_model_path = mode_path[0].firstChild.data
        self.model_path, self.model_name = os.path.split(self.abs_model_path)
        
        mode_scale = self.root.getElementsByTagName('mode_scale')
        mode_scale_str = mode_scale[0].firstChild.data
        self.sr_scale = int(mode_scale_str)
        
        noise_level_model = self.root.getElementsByTagName('noise_level_model')
        noise_level_img_str = noise_level_model[0].firstChild.data
        self.noise_level_model = float(noise_level_img_str) 
        
        kernel_width_default_c_1 = self.root.getElementsByTagName('kernel_width_default_c_1')
        kernel_width_default_c_1_str = kernel_width_default_c_1[0].firstChild.data
        self.kernel_width_default_c_1 = float(kernel_width_default_c_1_str)
        
        kernel_width_default_c_2 = self.root.getElementsByTagName('kernel_width_default_c_2')
        kernel_width_default_c_2_str = kernel_width_default_c_2[0].firstChild.data
        self.kernel_width_default_c_2 = float(kernel_width_default_c_2_str)
        
        kernel_width_default_c_3 = self.root.getElementsByTagName('kernel_width_default_c_3')
        kernel_width_default_c_3_str = kernel_width_default_c_3[0].firstChild.data
        self.kernel_width_default_c_3 = float(kernel_width_default_c_3_str)
        
        kernel_width_default_c_4 = self.root.getElementsByTagName('kernel_width_default_c_4')
        kernel_width_default_c_4_str = kernel_width_default_c_4[0].firstChild.data
        self.kernel_width_default_c_4 = float(kernel_width_default_c_4_str)
        
        fspecial_filter_type = self.root.getElementsByTagName('fspecial_filter_type')
        self.fspecial_filter_type = fspecial_filter_type[0].firstChild.data
        
        fspecial_filter_hsize = self.root.getElementsByTagName('fspecial_filter_hsize')
        fspecial_filter_hsize_str = fspecial_filter_hsize[0].firstChild.data
        self.fspecial_filter_hsize = int(fspecial_filter_hsize_str)
        
        boarder_handling = self.root.getElementsByTagName('boarder_handling')
        boarder_handling_str = boarder_handling[0].firstChild.data
        self.boarder_handling = int(boarder_handling_str)
        
        cut_size_height = self.root.getElementsByTagName('cut_size_height')
        cut_size_height_str = cut_size_height[0].firstChild.data
        self.xheight = int(cut_size_height_str)
        
        cut_size_width = self.root.getElementsByTagName('cut_size_width')
        cut_size_width_str = cut_size_width[0].firstChild.data
        self.xwidth = int(cut_size_width_str)
        
        
        self.kernel_width_default_x1234 = [self.kernel_width_default_c_1, self.kernel_width_default_c_2, 
                                           self.kernel_width_default_c_3, self.kernel_width_default_c_4] 
        
        
        self.kernel_width = self.kernel_width_default_x1234[self.sr_scale - 1]
        
        k = util_deblur.fspecial(self.fspecial_filter_type, self.fspecial_filter_hsize, self.kernel_width)
        self.k = util_sr.shift_pixel(k, self.sr_scale)
        #  TODO k的总和为1， 判断是否需要归一化
        # k /= np.sum(k)
        print("end parsing the XML file of the [super resolution]")
