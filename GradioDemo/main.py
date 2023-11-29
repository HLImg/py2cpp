# -*- coding: utf-8 -*-
# @Time    : 10/24/23 3:44 PM
# @File    : main.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import gradio
import util_test as test
from gradio.components import Image

# 主题设置
# theme = gradio.themes.Soft()
theme = gradio.themes.Base()
# 图像显示自适应屏幕
image_adaptive_style = {"max-width": "100%", "max-height": "100%"}

# 读取css文件
css_file = 'css/style.css'

with gradio.Blocks(title='图像处理', theme=theme, css=css_file) as demo:
    # gradio.Markdown(f'<style>{custom_css}</style>')
    gradio.Markdown('# <center style="font-family: Arial; font-size: 32px;">商业遥感卫星系统项目地面段-影像超分处理</center>')
    with gradio.Tabs():
        with gradio.TabItem('全色多光谱图像融合', elem_id="function-tab-1"):
            with gradio.Row():
                inp_pan_id_1 = gradio.Image(source='upload')
                inp_msi_id_1 = gradio.Image(source='upload')
            with gradio.Row():
                out_fusion_id_1 = gradio.Image(label='download')

            with gradio.Row():
                with gradio.Column():
                    gradio.Markdown('### <center>参数设置</center>')
                    with gradio.Tab('全色多光谱图像融合'):
                        with gradio.Row():
                            setting_1_fusion_weight = gradio.Slider(minimum=0, maximum=1,
                                                                    step=0.05, label='融合权重')
                            setting_2_fusion_weight = gradio.Slider(minimum=0, maximum=1,
                                                                    step=0.05, label='alpha')
                    with gradio.Tab('卷积神经网络'):
                        with gradio.Row():
                            setting_2_fusion_weight = gradio.Slider(minimum=0, maximum=1,
                                                                    step=0.05, label='融合权重')
                with gradio.Column():
                    gradio.Markdown('#### <center>选择显示波段</center>')
                    select_band_r_id1 = gradio.Textbox(label='R')
                    select_band_g_id1 = gradio.Textbox(label='G')
                    select_band_b_id1 = gradio.Textbox(label='B')
                    # 进度条
                    button_fusion_id1 = gradio.Button('开始融合')

            with gradio.Row():
                out_info_id1 = gradio.Textbox()

        with gradio.TabItem('图像超分辨率', elem_id="function-tab-2"):
            with gradio.Row():
                sr_msi_id_2 = gradio.Image(source='upload', label='多光谱图像', show_label=True)
                sr_pan_id_2 = gradio.Image(source='upload', label='全色图像', show_label=True)
            with gradio.Row():
                sr_res_id_2 = Image(show_label=True, label='超分结果')
            with gradio.Row():
                with gradio.Tab('数据库'):
                    sr_data_blur_kernel_size = gradio.Slider(0, 31, step=1, label='模糊核尺寸')
                    sr_data_blur_kernel_sigma = gradio.Slider(0, 50, step=0.5, label='模糊和标准差')
                    sr_data_noise_level = gradio.Slider(0, 100, step=0.5, label='噪声等级')
                    sr_data_jpeg_level = gradio.Slider(0, 100, step=0.5, label='JPEG压缩')
                    sr_data_output_dir = gradio.Text(lines=1, label='输出路径', show_label=True, show_copy_button=True)
                    sr_data_gen_button = gradio.Button('开始生成')
                    sr_data_process_bar = gradio.Textbox(label='数据生成进度', show_label=True)
                with gradio.Tab('训练'):
                    with gradio.Row():
                        with gradio.Column():
                            sr_train_method = gradio.Dropdown(choices=['生成对抗网络', '卷积神经网络', '双三次线性插值'],
                                                             label='超分方法选择')
                        with gradio.Column():
                            sr_train_img_type = gradio.Dropdown(choices=['全色图像', '多光谱图像'], label='超分图像类型')

                    sr_train_lr = gradio.Slider(1e-6, 1, step=1e-6, label='学习率', show_label=True)
                    sr_train_save_freq = gradio.Slider(0, 100, step=1, label='模型保存频次', show_label=True)
                    sr_train_save_dir = gradio.Text(lines=1, label='模型保存目录', show_label=True)
                    sr_train_button = gradio.Button('开始训练')
                    sr_train_process_bar = gradio.Textbox(label='训练进度')

                with gradio.Tab('测试'):
                    with gradio.Row():
                        with gradio.Column():
                            sr_test_method = gradio.Dropdown(choices=['生成对抗网络', '卷积神经网络', '双三次线性插值'],
                                                             label='超分方法选择')
                        with gradio.Column():
                            sr_test_img_type = gradio.Dropdown(choices=['全色图像', '多光谱图像'], label='超分图像类型')
                    sr_nn_ckpt_path = gradio.Text(lines=1, label='模型参数文件地址', show_copy_button=True)
                    sr_scale = gradio.Slider(minimum=1, maximum=4, step=1, label='超分倍数')

                    sr_button_test = gradio.Button('开始超分')
                    sr_button_test.click(fn=test.super_resolution,
                                         inputs=[sr_pan_id_2, sr_scale],
                                         outputs=[sr_res_id_2])


        with gradio.TabItem('数据定量分析', elem_id="function-tab-3"):
            pass
        with gradio.TabItem('关于', elem_id="function-tab-4"):
            gradio.Markdown('## 介绍')



demo.queue()
demo.launch()
