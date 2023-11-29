# -*- coding: utf-8 -*-
# @Time    : 10/24/23 3:44 PM
# @File    : main.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import gradio
import util_sr
from gradio.components import Image

# 主题设置
# theme = gradio.themes.Soft()
theme = gradio.themes.Base()
# 图像显示自适应屏幕
image_adaptive_style = {"max-width": "100%", "max-height": "100%"}

# 读取css文件
css_file = 'css/style.css'

alpha_label = "&#945; (alpha)"  # 使用 HTML 实体
beta_label = "&#946; (beta)"    # 使用 HTML 实体
gamma_label = "&#947; (gamma)"  # 使用 HTML 实体

with gradio.Blocks(title='图像处理', theme=theme, css=css_file) as demo:
    # gradio.Markdown(f'<style>{custom_css}</style>')
    gradio.Markdown('# <center style="font-family: Arial; font-size: 32px;">商业遥感卫星系统项目地面段-影像超分处理</center>')
    with gradio.Tabs():
        with gradio.TabItem('全色多光谱图像融合', elem_id="function-tab-1"):
            with gradio.Row():
                ps_msi_id_1 = gradio.Image(sources='upload', label="多光谱图像", show_label=True)
                ps_pan_id_1 = gradio.Image(sources='upload', label="全色图像", show_label=True)
            with gradio.Row():
                ps_res_id_1 = gradio.Image(label='融合结果', show_label=True)
            with gradio.Row():
                with gradio.Tab("数据库"):
                    ps_data_blur_kernel_size = gradio.Slider(0, 31, step=1, label="模糊核尺寸")
                    ps_data_blur_kernel_sigma = gradio.Slider(0, 50, step=0.5, label="模糊核标准差")
                    ps_data_noise_level = gradio.Slider(0, 100, step=0.5, label="噪声等级")
                    ps_data_output_dir = gradio.Text(lines=1, label="输出路径", show_label=True, show_copy_button=True)
                    ps_data_gen_button = gradio.Button("开始生成")
                    ps_data_process_bar = gradio.Textbox(label="数据生成进度", show_label=True)
                
                with gradio.Tab("训练"):
                    ps_train_method = gradio.Dropdown(choices=['模型类方法', '卷积神经网络'],
                                                          label="选择图像融合方法")
                    with gradio.Row():
                        with gradio.Column():
                            ps_train_lr = gradio.Number(label="学习率", show_label=True)
                        with gradio.Column():
                            ps_train_save_freq = gradio.Number(step=1, label="模型保存频次")
                    ps_train_save_dir = gradio.Text(lines=1, label='模型保存目录', show_label=True)
                    ps_data_blur_kernel_sigma_train_button = gradio.Button('开始训练')
                    ps_train_process_bar = gradio.Textbox(label='训练进度')
                
                with gradio.Tab("测试"):
                    ps_test_method = gradio.Dropdown(choices=['模型类方法', '卷积神经网络'],
                                                          label="选择图像融合方法")
                    ps_nn_ckpt_path = gradio.Text(lines=1, label='预训练模型地址', show_copy_button=True)
                    
                    with gradio.Row():
                        with gradio.Column():
                            ps_test_model_alpha = gradio.Number(label="a", show_label=True)
                        with gradio.Column():
                            ps_test_model_beta = gradio.Number(label="b", show_label=True)
                        with gradio.Column():
                            ps_test_model_gamma = gradio.Number(label="g", show_label=True)
                    
                    ps_test_save_path = gradio.Text(lines=1, label="融合结果保存地址", show_label=True)
                
                    ps_button_test = gradio.Button('开始融合')
        
        """
        超分辩率
        """

        with gradio.TabItem('图像超分辨率', elem_id="function-tab-2"):
            with gradio.Row():
                sr_msi_id_2 = gradio.Image(sources='upload', label='多光谱图像', show_label=True)
                sr_pan_id_2 = gradio.Image(sources='upload', label='全色图像', show_label=True)
            with gradio.Row():
                sr_res_id_2 = Image(show_label=True, label='超分结果')
            with gradio.Row():
                with gradio.Tab('数据库'):
                    sr_data_blur_kernel_size = gradio.Slider(0, 31, step=1, label='模糊核尺寸')
                    sr_data_blur_kernel_sigma = gradio.Slider(0, 50, step=0.5, label='模糊核标准差')
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
                    
                    with gradio.Row():
                        with gradio.Column():
                            sr_train_lr = gradio.Number(label='学习率', show_label=True)
                        with gradio.Column():
                            sr_train_save_freq = gradio.Number(label='模型保存频次', show_label=True)

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
                    sr_button_test.click(fn=util_sr.sr_test_interface,
                                         inputs=[sr_pan_id_2, sr_scale],
                                         outputs=[sr_res_id_2])


        with gradio.TabItem('数据定量分析', elem_id="function-tab-3"):
            pass
        with gradio.TabItem('关于', elem_id="function-tab-4"):
            gradio.Markdown('## 介绍')



demo.queue()
demo.launch()
