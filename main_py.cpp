/**
 * @brief: super resolution with py_array
 * @author: eis.whu
*/

# include <pybind11/pybind11.h>
# include <pybind11/embed.h>
# include <pybind11/numpy.h>
# include <iostream>
# include <string>
# include <vector>
# include <map>
# include <typeinfo>

namespace py = pybind11 ;

template <typename T>
std::vector<size_t> get_array_shape(const py::array_t<T>& img){
    if (img.ndim() == 3) {
        auto r = img.template unchecked<3>();
        return {r.shape(0), r.shape(1), r.shape(2)};
    } else if (img.ndim() == 2) {
        auto r = img.template unchecked<2>();
        return {r.shape(0), r.shape(1), 1};
    } else {
        throw std::runtime_error("Array must be 2D or 3D.");
    }
}

template <typename T>
py::array_t<T> py_crop(py::array_t<T>& img, )

int main(int argc, char *argv){
    // set cuda device
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1) ;
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

for (int ii = 0; ii < 2 ; ii++){
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
    std::vector<size_t> ori_shape = get_array_shape(image);
    size_t overlap = 200 ;
    size_t num_band = ori_shape[2];

    for (int iii = 0 ; iii < num_band; iii++){
        std::cout << "start band  " << iii + 1 << std::endl ; 
        py::array_t<uint16_t> cut_img({ori_shape[0], ori_shape[1], 1});
        
        // deepcopy
        for (int y = 0; y < ori_shape[0]; ++y) {
            for (int x = 0; x < ori_shape[1]; ++x) {
                cut_img.mutable_at(y, x, 0) = image.at(y, x, iii);
        
        size_t height = ori_shape[0] ;
        size_t width = ori_shape[1] ;

        double xmax = py::max(cut_img).cast<double>();

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

        int i = 0 ;
        int overlap_rightedge, overlap_downedge ;
        std::map<int, cv::Mat> smalldata_tensor ;

        for (size_t h = 0 ; h < max_heinum; h++){
            for (size_t w = 0; w < max_widnum; w++){
                // Assuming oriheight, h, xheight, oriwidth, w and xwidth are properly defined
                // 拷贝
                py::slice row_slice(oriheight * h, oriheight * h + xheight, 1);
                py::slice col_slice(oriwidth * w, oriwidth * w + xwidth, 1);
                py::array_t<uint16_t> crop_img = cut_img[row_slice][col_slice];

            }
        }

    }
}

    }

}