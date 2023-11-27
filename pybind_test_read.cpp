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

int main(){
    
    py::scoped_interpreter guard{} ;
    py::module py_test = py::module::import("pybind_debug") ;
    py::object py_util = py_test.attr("util")() ;

    std::string mux_path = "/data/dataset/project/1-PAN.TIF" ;

    // read multi spectral image
    auto image = py_util.attr("read")(mux_path).cast<py::array_t<uint16_t>>() ;
    cv::Mat mat_hwc = np2mat_u16(image) ;
    
    for (size_t c = 0 ; c < mat_hwc.channels() ; c++){
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

    std::cout << "(" << mat_hwc.rows << ", " << mat_hwc.cols << ", " << mat_hwc.channels() << " )" << std::endl;
    

    py::array_t<uint16_t> py_arr = to_py_arr<uint16_t>(mat_hwc);

    py_util.attr("savemat")(py_arr) ;
    return 0 ;
}