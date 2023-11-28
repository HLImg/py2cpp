/**
 * @brief: super resolution
 * @author: eis.whu
*/

# include <pybind11/pybind11.h>
# include <pybind11/embed.h>
# include <pybind11/numpy.h>
# include <iostream>
# include <string>
# include <vector>
# include <map>
# include <math.h>
# include <opencv2/opencv.hpp>
# include <opencv2/highgui.hpp>

# include <typeinfo>

namespace py = pybind11 ;


cv::Mat np2mat_u16(py::array_t<u_int16_t>& img){
    py::buffer_info buf_info = img.request();
    size_t h, w, channels ;
    h = buf_info.shape[0] ;
    w = buf_info.shape[1] ;
    channels = buf_info.shape[2] ;

    int dtype = CV_MAKETYPE(CV_16U, channels);
    // 重塑flat_mat以匹配原始图像的维度和通道
    cv::Mat reshaped_mat(h, w, dtype, img.mutable_data());

    std::cout << "converted  " << reshaped_mat.rows <<"  " <<reshaped_mat.cols << "   "<< reshaped_mat.channels() << std::endl;
    return reshaped_mat ;
}

template <typename T>
py::array_t<T> to_py_arr(const cv::Mat& img) {
    // 创建一个匹配cv::Mat尺寸和通道数的numpy数组
    py::array_t<T> py_arr({img.rows, img.cols, img.channels()}, {sizeof(T)*img.cols*img.channels(), sizeof(T)*img.channels(), sizeof(T)});
    
    // 获取numpy数组的mutable视图用于修改数据
    py::buffer_info buf = py_arr.request();
    
    for (int i = 0; i < img.rows; ++i) {
        // 计算每行的指针位置
        auto ptr_src = img.ptr<T>(i);
        auto ptr_dst = static_cast<T*>(buf.ptr) + i * img.cols * img.channels();
        
        // 复制当前行
        memcpy(ptr_dst, ptr_src, sizeof(T) * img.cols * img.channels());
    }
    
    return py_arr;
}


int main(int argc, char *argv[]){

    // setting cuda device
    // setenv("CUDA_VISIBLE_DEVICES", "-1", 1);

    std::string inp_xml = "./input.xml" ;
    std::string sr_xml = "./super_option.xml" ;

    if (argc > 3){
        std::cerr << "Too many arguments. Expected  2, recevied " << argc - 1 << ".\n" ;
        return 1 ;
    }
    else if (argc == 3){
        inp_xml, sr_xml = argv[1], argv[2] ;
    } 
    else if (argc == 2){
        inp_xml = argv[1] ;
    }

    /*------------------------------ program entry --------------------------------------*/
    py::scoped_interpreter guard{} ; // initial interpreter of python

    /**
     * @brief: parsing the xml file
     * @date: Nov 19 22:52
     * @details : OK
    */
    py::module py_module_xml = py::module::import("pybind_xml") ;
    py::object inp_args = py_module_xml.attr("XMLParserInput")(inp_xml) ;
    py::object sr_args = py_module_xml.attr("XMLParserSR")(sr_xml) ;

    /**
     * @brief : crop image
    */

    py::module py_module_model = py::module::import("pybind_model") ;

    py::module py_module_util = py::module::import("pybind_util") ;
    py::object py_util = py_module_util.attr("Util")();
    
    std::string inp_pan_path = inp_args.attr("inp_pan").cast<std::string>() ;
    std::string inp_mux_path = inp_args.attr("inp_mux").cast<std::string>() ;

    int sr_scale = sr_args.attr("sr_scale").cast<int>() ;
    int xheight = sr_args.attr("xheight").cast<int>() ;
    int xwidth = sr_args.attr("xwidth").cast<int>() ;

    std::cout << "sr-scale is " << sr_scale << std::endl ;
    std::cout << "input pan path is " << inp_pan_path << std::endl ;
    std::cout << "input mux path is " << inp_mux_path << std::endl ;
    std::cout << "crop setting [xheight]=" << xheight << " [xwidth]=" << xwidth << std::endl;

    for (int ii = 1; ii < 2 ; ii++){
        std::string inp_data;
        if (ii == 0){
            inp_data = inp_pan_path ;
            std::cout << "############ [ start SR-PAN .... ] ############" << std::endl;
        }
        else{
            inp_data = inp_mux_path ;
            std::cout << "############ [ start SR-MUX .... ] ############" << std::endl;
        }
        auto image = py_util.attr("read_gdal_mul")(inp_data).cast<py::array_t<uint16_t>>() ;
        auto mat_hwc = np2mat_u16(image) ;

        size_t inp_h = mat_hwc.rows ;
        size_t inp_w = mat_hwc.cols ;

        size_t overlap = 200 ;
        size_t num_band = mat_hwc.channels();


        auto k  = sr_args.attr("k").cast<py::array_t<_Float32>>() ;
        py::object py_model = py_module_model.attr("Model")(k, sr_args, inp_h, inp_w, num_band);

        for (int iii = 0 ; iii < num_band; iii++){
            std::cout << ii <<" - start band  " << iii + 1 << std::endl ; 
            cv::Mat cur_img = cv::Mat::zeros(mat_hwc.rows, mat_hwc.cols, CV_16UC1);

            // cv::Mat cur_img(mat_hwc.rows, mat_hwc.cols, CV_16UC1);
            int from_to[] = { iii, 0 };
            cv::mixChannels(&mat_hwc , 1, &cur_img, 1, from_to, 1); 

            size_t height = cur_img.rows ;
            size_t width = cur_img.cols ;

            double minVal;
            double xmax = 0 ;
            cv::minMaxLoc(cur_img, &minVal, &xmax);
            
            int oriheight = xheight - overlap ;
            int oriwidth = xwidth - overlap ;

            
            int max_heinum = std::ceil((height - xheight) / static_cast<double>(oriheight));
            int max_widnum = std::ceil((width - xwidth) / static_cast<double>(oriwidth)) ;
            
            int xheinum1 = max_heinum ;
            int xwidnum1 = max_widnum ;
            if (height - max_heinum * oriheight > 0)
                xheinum1 = max_heinum + 1 ;
            
            if (width - max_widnum * oriwidth > 0)
                xwidnum1 = max_widnum + 1 ; 
        

            // start to crop image to patches
           
            int i  = 0 ;
            int overlap_rightedge, overlap_downedge ;
            std::map<int, cv::Mat> smalldata_tensor ;

            std::cout << "start to crop" << std::endl;
            
            for (size_t h = 0 ; h < max_heinum ; h++){
                for (size_t w = 0; w < max_widnum; w++){
                    cv::Rect roi(oriwidth * w, oriheight * h, xwidth, xheight) ;
                    cv::Mat crop_img = cur_img(roi) ;
                    
                    i = i + 1 ;
                    smalldata_tensor[i] = crop_img.clone();

                    if (w == max_widnum - 1 && width - max_widnum * oriwidth > 0){
                        i = i + 1 ;
                        cv::Rect roi_r(oriwidth * (w + 1), oriheight * h, width - oriwidth * (w + 1), xheight);
                        cv::Mat crop_img_r = cur_img(roi_r);
                        smalldata_tensor[i] = crop_img_r.clone();

                        overlap_rightedge = width - oriwidth * (w + 1);  
                    }
                }
            }

            if (height - max_heinum * oriheight > 0){
                for (size_t w = 0 ; w < max_widnum ; w++){
                    i = i + 1 ;
                    cv::Rect roi(oriwidth * w, oriheight * max_heinum, xwidth, height - oriheight * max_heinum);
                    cv::Mat crop_img = cur_img(roi);
                    overlap_downedge = height - (oriheight * max_heinum);
                    smalldata_tensor[i] = crop_img.clone();

                    if (w == max_widnum - 1 &&  width - max_widnum * oriwidth > 0){
                        i = i + 1 ;
                        cv::Rect roi_r(oriwidth * (w + 1), oriheight * max_heinum, 
                                        width - oriwidth * (w + 1), height - oriheight * max_heinum) ;
                        cv::Mat crop_img_r = cur_img(roi_r);
                        smalldata_tensor[i] = crop_img_r.clone();

                        double min2;
                        double max2 = 0 ;
                        cv::minMaxLoc(smalldata_tensor[i], &min2, &max2);
                        // std::cout << "[debug : crop_img], ori " <<  min2 << ", " << max2 << std::endl ; 
                        // std::cout << "[debug]  channel " << cur_img.channels() <<std::endl;
                    }
                }
            }

            // std::cout << "crop number is " << i << std::endl ;
            cur_img.release() ;

            /**
             * @brief : super resolution
             * @author : lianghao@whu.edu.cn
             * @date : Nov 20 23:16
            */
            
            // auto k  = sr_args.attr("k").cast<py::array_t<_Float32>>() ;
            // py::object py_model = py_module_model.attr("Model")(k, sr_args, xmax, );

            int dic_len = smalldata_tensor.size() ;
            int boarder_handling = sr_args.attr("boarder_handling").cast<int>() ;

            size_t sr_xheight = xheight * sr_scale ;
            size_t sr_xwidth = xwidth * sr_scale ;
            size_t sr_oriwidth = oriwidth * sr_scale ;
            size_t sr_oriheight = oriheight * sr_scale ;
            size_t sr_overlap = overlap * sr_scale ; 
            size_t sr_height = height * sr_scale ;
            size_t sr_width = width * sr_scale ;

            // cv::Mat mage = cv::Mat::zeros(sr_height, sr_width, CV_16U);

            size_t num_row = xwidnum1 ;
            size_t num_col = xheinum1 ;

            py_model.attr("set")(sr_xheight, sr_xwidth, sr_oriheight, 
            sr_oriwidth, sr_overlap, sr_height, sr_width, overlap_downedge, overlap_rightedge) ;

            // auto tmp = py_util.attr("calWeight")(sr_overlap, 0.1);
            // auto w_lr = tmp.cast<py::array_t<float>>();
            
            // auto w_left_right0 = py_util.attr("tile")(w_lr, sr_xheight, 1) ;
            // auto w_left_right = py_util.attr("tile")(w_lr, sr_oriheight, 1) ;
            // auto w_left_right_downedge = py_util.attr("tile")(w_lr, overlap_downedge * sr_scale - sr_overlap, 1) ;
            // auto w_left_right_corner = py_util.attr("tile")(w_lr, sr_overlap, 1) ;
            
            // auto w_ud = py_util.attr("reshape")(w_lr, sr_overlap, 1)  ;
            // auto w_up_down = py_util.attr("tile")(w_ud, 1, sr_oriwidth) ;
            // auto w_up_down_rightedge = py_util.attr("tile")(w_ud, 1, 
            //                                     overlap_rightedge * sr_scale) ;

            for (size_t i = 0; i < dic_len ; i++){
                // if (i % 10 == 0)
                //     std::cout << "finished: [" << i << "]/[" << dic_len <<"]"<< std::endl ;

                std::cout << "finished: [" << i << "]/[" << dic_len <<"]"<< std::endl ;
                
                double min2;
                double max2 = 0 ;
                auto img = smalldata_tensor[i + 1] ;

                cv::minMaxLoc(cur_img, &min2, &max2);
                auto tmp = to_py_arr<u_int16_t>(img) ;
                // std::cout << "[debug : to_py_arr], ori " <<  min2 << ", " << max2 << std::endl ; 
                auto img_e = py_model.attr("inference")(tmp, xmax, iii, i, num_row, num_col).cast<py::array_t<float>>();
            }

        }

        mat_hwc.release() ;

        std::string path ;
        if (ii == 0){
            path = inp_args.attr("out_pan").cast<std::string>() ;
        }else{
            path = inp_args.attr("out_mux").cast<std::string>() ;
           
        }

        py_model.attr("save_tiff")(path) ;
        std::cout << "###################################" << std::endl ;
        std::cout << ii <<"  saved on "<< path << std::endl ;
        std::cout << "###################################" << std::endl ;
    }

    inp_args.attr("finish")() ;
    std::cout << "###################################" << std::endl ;
    std::cout << "All Finish" << std::endl ;
    return 0 ;
}



