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


void statis(const cv::Mat& img){
    double mu = 0 ;
    double xmax = 0 ;
    double xmin = 20000 ;
    double count = 0 ;
    for (size_t i = 0 ; i < img.rows ; i++)
        for (size_t j = 0 ; j < img.cols ; j++){
            int pixel = img.at<uint16_t>(i, j) ;
            if (pixel < xmin)
                xmin = pixel ;
            if (pixel > xmax)
                xmax = pixel ;
            mu = mu + pixel ;
            count = count + 1;
        }
    
    std::cout << "hw1 : xmin = [" << xmin << "] , xmax = [" << xmax << "], mean = [" << mu / count << " ]" << std::endl ;
}

void statis_mul(const cv::Mat& img){
    double mu = 0 ;
    int xmax = 0 ;
    int xmin = 20000 ;
    double count = 0 ;
    std::cout << "mul-statis-c" << img.channels() << std::endl ;
    for (size_t i = 0 ; i < img.rows ; i++)
        for (size_t j = 0 ; j < img.cols ; j++)
             {cv::Vec4w pixel = img.at<cv::Vec4w>(i, j);  
            for (size_t c = 0 ; c < img.channels() ; c++){
                if (pixel[c] < xmin)
                xmin = pixel[c] ;
                if (pixel[c] > xmax)
                xmax = pixel[c] ;
             mu = mu + pixel[c] ;
            count = count + 1;
        }
             }
    
    std::cout << "hwc : xmin = [" << xmin << "] , xmax = [" << xmax << "], mean = [" << mu / count << " ]" << std::endl ;
}


cv::Mat np2mat_u16(py::array_t<uint16_t>& img){
    py::buffer_info buf_info = img.request();
    size_t h, w, channels ;
    h = buf_info.shape[0] ;
    w = buf_info.shape[1] ;
    channels = buf_info.shape[2] ;
    cv::Mat flat_mat(h * w * channels, 1, CV_16UC1, img.mutable_data());

    // 重塑flat_mat以匹配原始图像的维度和通道
    cv::Mat reshaped_mat = flat_mat.reshape(channels, h);
    return reshaped_mat ;
}
cv::Mat np2mat_mul(py::array_t<uint16_t>& img){
    py::buffer_info buf_info = img.request();
    size_t h, w, channels ;
    h = buf_info.shape[0] ;
    w = buf_info.shape[1] ;
    channels = buf_info.shape[2] ;

    // 重塑flat_mat以匹配原始图像的维度和通道
    cv::Mat reshaped_mat(h, w, CV_16UC4, img.mutable_data());

    std::cout << "converted  " << reshaped_mat.rows <<"  " <<reshaped_mat.cols << "   "<< reshaped_mat.channels() << std::endl;
    return reshaped_mat ;
}


template <typename T>
cv::Mat to_mat_mul(const py::array_t<T>& img) {
    auto r = img.template unchecked<3>();
    std::cout << r.shape(2) << std::endl ;
    // 创建4个颜色通道，应该使用 CV_16UC4
    size_t dtype =  CV_MAKETYPE(CV_16U, r.shape(2));

  
    cv::Mat mat(r.shape(0), r.shape(1), dtype);
    double max_py = 0;
    double max_mat = 0 ;
    for (size_t c = 0; c < r.shape(2); c++){
        cv::Mat channel = cv::Mat::zeros(r.shape(0), r.shape(1), CV_16U);
        for (size_t i = 0; i < r.shape(0); i++) {
            for (size_t j = 0; j < r.shape(1); j++) {
                channel.at<uint16_t>(i, j) = r(i, j, c);
                if (r(i, j, c) >max_py)
                    max_py =  r(i, j, c) ;
                if (channel.at<uint16_t>(i, j) > max_mat)
                    max_mat = channel.at<uint16_t>(i, j) ;
            }
        }
        int from_to[] = {0, int(c)} ;
        cv::mixChannels(&channel, 1, &mat, 1, from_to, 1);
    }


    double ch_max = 0;

    for (size_t c = 0; c < r.shape(2); c++){
        for (size_t i = 0; i < r.shape(0); i++) {
            for (size_t j = 0; j < r.shape(1); j++) {
                if (mat.at<uint16_t>(i, j) > ch_max)
                    ch_max = mat.at<uint16_t>(i, j) ;
            }
        }
    }
    
    std::cout << "convert : py-xmax  " <<max_py  <<" | cv-xmax  " << max_mat<<" | check-xmax  " << ch_max<< std::endl ;
    return mat;
}

template <typename T>
cv::Mat to_mat_float(const py::array_t<T>& img){
    auto r = img.template unchecked<2>();
    // std::cout << "dtype is " << cv::DataType<T>::type << std::endl ;
    cv::Mat mat(r.shape(0), r.shape(1), CV_32F);

    for (size_t i = 0 ; i < r.shape(0); i++){
        for (size_t j = 0; j < r.shape(1) ; j++)
            mat.at<T>(i, j) = r(i, j) ;
    }
    return mat ;
}

template <typename T>
py::array_t<T> to_py_arr(const cv::Mat& img) {
    if (img.channels() == 1) {
        py::array_t<T> py_arr = py::array_t<T>({img.rows, img.cols});
        memcpy(py_arr.request().ptr, img.ptr(),
               sizeof(T) * img.total());
        return py_arr;
    } else {
        py::array_t<T> py_arr =
                py::array_t<T>({img.rows, img.cols, img.channels()});
        memcpy(py_arr.request().ptr, img.ptr(),
               sizeof(T) * img.total());
        return py_arr;
    }
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
        // auto mat_hwc = to_mat_mul(image) ;  // u16int, (h, w, c)
        // auto mat_hwc = np2mat_u16(image) ;
        auto mat_hwc = np2mat_u16(image) ;
        statis_mul(mat_hwc) ;

        // 检查每个通道的最大值
        for (size_t c = 0 ; c < 4 ; c++){
            int cmax = 0;
            double mu = 0;
            for (size_t i = 0 ; i < mat_hwc.rows ; i++)
                for (size_t j = 0; j < mat_hwc.cols;j++){
                    cv::Vec<uint16_t, 4> pixel = mat_hwc.at<cv::Vec<uint16_t, 4>>(i, j);
                    if (cmax < pixel[c])
                    cmax = pixel[c];
                mu = mu + pixel[c];
                }
    
            std::cout << "band-" << c + 1 << " , max is " << cmax << " , mu is" << mu / (mat_hwc.rows * mat_hwc.cols) << std::endl ;
        }


        size_t overlap = 200 ;
        size_t num_band = mat_hwc.channels();


        // TODO
        //cv::Mat smux = cv::Mat::zeros(height0, width0, CV_16UC4);

        for (int iii = 0 ; iii < num_band; iii++){
            std::cout << "start band  " << iii + 1 << std::endl ; 
            cv::Mat cur_img = cv::Mat::zeros(mat_hwc.rows, mat_hwc.cols, CV_16UC1);

            // cv::Mat cur_img(mat_hwc.rows, mat_hwc.cols, CV_16UC1);
            int from_to[] = { iii, 0 };
            cv::mixChannels(&mat_hwc , 1, &cur_img, 1, from_to, 1); 

            std::cout << "cur-img - band : " << iii + 1 << " / " << num_band << std::endl ;
            statis(cur_img) ;

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
                    statis(smalldata_tensor[i]) ;
                    if (w == max_widnum - 1 && width - max_widnum * oriwidth > 0){
                        i = i + 1 ;
                        cv::Rect roi_r(oriwidth * (w + 1), oriheight * h, width - oriwidth * (w + 1), xheight);
                        cv::Mat crop_img_r = cur_img(roi_r);
                        smalldata_tensor[i] = crop_img_r.clone();
                        statis(smalldata_tensor[i]) ;
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
                    statis(smalldata_tensor[i]) ;
                    if (w == max_widnum - 1 &&  width - max_widnum * oriwidth > 0){
                        i = i + 1 ;
                        cv::Rect roi_r(oriwidth * (w + 1), oriheight * max_heinum, 
                                        width - oriwidth * (w + 1), height - oriheight * max_heinum) ;
                        cv::Mat crop_img_r = cur_img(roi_r);
                        smalldata_tensor[i] = crop_img_r.clone();
                        statis(smalldata_tensor[i]) ;
                        double min2;
                        double max2 = 0 ;
                        cv::minMaxLoc(smalldata_tensor[i], &min2, &max2);
                        std::cout << "[debug : crop_img], ori " <<  min2 << ", " << max2 << std::endl ; 
                        std::cout << "[debug]  channel " << cur_img.channels() <<std::endl;
                    }
                }
            }

            std::cout << "crop number is " << i << std::endl ;
            cur_img.release() ;

        

            /**
             * @brief : super resolution
             * @author : lianghao@whu.edu.cn
             * @date : Nov 20 23:16
            */
            
            auto k  = sr_args.attr("k").cast<py::array_t<_Float32>>() ;
            py::object py_model = py_module_model.attr("Model")(k, sr_args, xmax);

            int dic_len = smalldata_tensor.size() ;
            int boarder_handling = sr_args.attr("boarder_handling").cast<int>() ;

            size_t sr_xheight = xheight * sr_scale ;
            size_t sr_xwidth = xwidth * sr_scale ;
            size_t sr_oriwidth = oriwidth * sr_scale ;
            size_t sr_oriheight = oriheight * sr_scale ;
            size_t sr_overlap = overlap * sr_scale ; 
            size_t sr_height = height * sr_scale ;
            size_t sr_width = width * sr_scale ;

            cv::Mat mage = cv::Mat::zeros(sr_height, sr_width, CV_16U);

            size_t num_row = xwidnum1 ;
            size_t num_col = xheinum1 ;

            auto tmp = py_util.attr("calWeight")(sr_overlap, 0.1);
            auto w_lr = tmp.cast<py::array_t<float>>();
            
            auto w_left_right0 = py_util.attr("tile")(w_lr, sr_xheight, 1) ;
            auto w_left_right = py_util.attr("tile")(w_lr, sr_oriheight, 1) ;
            auto w_left_right_downedge = py_util.attr("tile")(w_lr, overlap_downedge * sr_scale - sr_overlap, 1) ;
            auto w_left_right_corner = py_util.attr("tile")(w_lr, sr_overlap, 1) ;
            
            auto w_ud = py_util.attr("reshape")(w_lr, sr_overlap, 1)  ;
            auto w_up_down = py_util.attr("tile")(w_ud, 1, sr_oriwidth) ;
            auto w_up_down_rightedge = py_util.attr("tile")(w_ud, 1, 
                                                overlap_rightedge * sr_scale) ;\

            for (size_t i = 0; i < dic_len ; i++){
                // if (i % 10 == 0)
                //     std::cout << "finished: [" << i << "]/[" << dic_len <<"]"<< std::endl ;

                std::cout << "finished: [" << i << "]/[" << dic_len <<"]"<< std::endl ;
                
                double min2;
                double max2 = 0 ;
                auto img = smalldata_tensor[i + 1] ;
                statis(img) ;
                cv::minMaxLoc(cur_img, &min2, &max2);
                auto tmp = to_py_arr<u_int16_t>(img) ;
                std::cout << "[debug : to_py_arr], ori " <<  min2 << ", " << max2 << std::endl ; 
                auto img_e = py_model.attr("inference")(tmp, i).cast<py::array_t<float>>();
                auto shape = img_e.shape() ;
                std::cout << "the shape of img_e is " << shape[0] << ", " << shape[1] << std::endl ;
            }
        }
        
    }

    return 0 ;
}



