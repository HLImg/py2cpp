# -*- coding: utf-8 -*-
# @Time    : 10/24/23 3:44 PM
# @File    : main.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import gradio
import util_ps
import util_sr
import util_metric
import util_common
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
                    ps_nn_ckpt_path = gradio.File()
                    upload_ps_ckpt_button = gradio.UploadButton("点击上传模型参数文件")
                    upload_ps_ckpt_button.upload(fn=util_common.upload_ckpt, inputs=upload_ps_ckpt_button, outputs=ps_nn_ckpt_path)
                    
                    with gradio.Row():
                        with gradio.Column():
                            ps_test_model_alpha = gradio.Number(label="alpha", show_label=True)
                        with gradio.Column():
                            ps_test_model_beta = gradio.Number(label="beta", show_label=True)
                        with gradio.Column():
                            ps_test_model_gamma = gradio.Number(label="gamma", show_label=True)
                    
                    ps_test_save_path = gradio.Text(lines=1, label="融合结果保存地址", show_label=True)
                
                    ps_button_test = gradio.Button('开始融合')
        
        """
        超分辩率
        """

        with gradio.TabItem('图像超分辨率', elem_id="function-tab-2"):
            with gradio.Row():
                with gradio.Column():
                    sr_msi_upload = gradio.UploadButton(label="上传多光谱图像")
                    sr_msi_id_show = gradio.Image(label='多光谱图像', show_label=True, visible=True)
                    sr_msi_id_2 = gradio.File(visible=False)
                    sr_msi_upload.upload(util_common.upload_tif, sr_msi_upload, outputs=[sr_msi_id_show, sr_msi_id_2])

                with gradio.Column():
                    sr_pan_upload = gradio.UploadButton(label="上传全色图像")
                    sr_pan_id_show = gradio.Image(label='全色图像', show_label=True, visible=True)
                    sr_pan_id_2 = gradio.File(visible=False)
                    sr_pan_upload.upload(util_common.upload_tif, sr_pan_upload, outputs=[sr_pan_id_show, sr_pan_id_2])
                    
            with gradio.Row():
                sr_res_id_show = gradio.Image(show_label=True, label='超分结果')
                sr_res_id_2 = gradio.File(label="超分结果", height=0.2, min_width=10, scale=0.2, interactive=False)
                
            with gradio.Row():
                with gradio.Tab('数据库'):
                    sr_data_blur_kernel_size = gradio.Slider(0, 31, step=1, label='模糊核尺寸')
                    sr_data_blur_kernel_sigma = gradio.Slider(0, 50, step=0.5, label='模糊核标准差')
                    sr_data_noise_level = gradio.Slider(0, 100, step=0.5, label='噪声等级')
                    sr_data_jpeg_level = gradio.Slider(0, 100, step=0.5, label='JPEG压缩')
                    sr_data_output_dir = gradio.Text(lines=1, label='输出路径', show_label=True, show_copy_button=True)
                    sr_data_gen_button = gradio.Button('开始生成')
                    sr_data_process_bar = gradio.Textbox(label='数据生成进度', show_label=True)
                    
                    # button
                    
                    sr_data_gen_button.click(fn=util_sr.sr_generate_image, 
                                             inputs=[sr_msi_id_2, sr_pan_id_2, 
                                                     sr_data_blur_kernel_size, 
                                                     sr_data_blur_kernel_sigma,
                                                     sr_data_noise_level,
                                                     sr_data_jpeg_level,
                                                     sr_data_output_dir], outputs=sr_data_process_bar)
                    
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
                    sr_train_button.click(fn=util_sr.sr_train, 
                                          inputs=[sr_msi_id_2, sr_pan_id_2, 
                                                  sr_train_method, sr_train_img_type, 
                                                  sr_train_lr, sr_train_save_freq, sr_train_save_dir],
                                          outputs=sr_train_process_bar)

                with gradio.Tab('测试'):
                    with gradio.Row():
                        with gradio.Column():
                            sr_test_method = gradio.Dropdown(choices=['生成对抗网络', '卷积神经网络', '双三次线性插值'],
                                                             label='超分方法选择')
                        with gradio.Column():
                            sr_test_img_type = gradio.Dropdown(choices=['全色图像', '多光谱图像'], label='超分图像类型')
                    # sr_nn_ckpt_path = gradio.Text(lines=1, label='模型参数文件地址', show_copy_button=True)
                    sr_nn_ckpt_path = gradio.File(height=10)
                    upload_sr_ckpt_button = gradio.UploadButton("点击上传模型参数文件")
                    upload_sr_ckpt_button.upload(fn=util_common.upload_ckpt, inputs=upload_sr_ckpt_button, outputs=sr_nn_ckpt_path)
                    
                    sr_scale = gradio.Slider(minimum=1, maximum=4, step=1, label='超分倍数')

                    sr_button_test = gradio.Button('开始超分')
                    sr_button_test.click(fn=util_sr.sr_test_interface,
                                         inputs=[sr_msi_id_2, sr_pan_id_2, sr_test_method,
                                                 sr_test_img_type, sr_nn_ckpt_path, sr_scale],
                                         outputs=[sr_res_id_show, sr_res_id_2])


        with gradio.TabItem('数据定量分析', elem_id="function-tab-3"):
            with gradio.Row():
                with gradio.Column():
                    metric_msi_upload = gradio.UploadButton(label="上传多光谱（低分辨率）图像")
                    metric_msi_show = gradio.Image(label='多光谱（低分辨率）图像', show_label=True, visible=True)
                    metric_msi_id_3 = gradio.File(visible=False)
                    metric_msi_upload.upload(util_common.upload_tif, metric_msi_upload, outputs=[metric_msi_show, metric_msi_id_3])
                with gradio.Column():
                    metric_pan_upload = gradio.UploadButton(label="上传全色（高分辨率）图像")
                    metric_pan_show = gradio.Image(label='全色（高分辨率）图像', show_label=True, visible=True)
                    metric_pan_id_3 = gradio.File(visible=False)
                    metric_pan_upload.upload(util_common.upload_tif, metric_pan_upload, outputs=[metric_pan_show, metric_pan_id_3])
                    
                with gradio.Column():
                    metric_fusion_upload = gradio.UploadButton(label="上传融合图像")
                    metric_fusion_show = gradio.Image(label="融合图像", show_label=True, visible=True)
                    metric_fusion_id_3 = gradio.File(visible=False)
                    metric_fusion_upload.upload(util_common.upload_tif, metric_fusion_upload, outputs=[metric_fusion_show, metric_fusion_id_3])
                    
            with gradio.Row():
                with gradio.Column():
                    metric_inp = gradio.CheckboxGroup(["PSNR", "RMSE", "SSIM", 
                                                        "ERGAS", "QNR", "AVG_GRAD"], 
                                                        label="定量分析指标")
                    metric_button = gradio.Button("开始分析")
                with gradio.Column():
                    metric_out = gradio.DataFrame(label="定量分析结果",
                                                   headers=["method", "value"],
                                                   datatype=["str", "str"], 
                                                   interactive=False, wrap=True)
                
                metric_button.click(util_metric.metric_interface, inputs=[metric_msi_id_3, metric_pan_id_3, metric_fusion_id_3,metric_inp], outputs=metric_out)
        with gradio.TabItem('关于', elem_id="function-tab-4"):
            gradio.Markdown('## 介绍')



demo.queue()
demo.launch()
