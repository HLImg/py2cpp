本程序的主体使用c++实现，关于模型推理或其他解析的工作使用python实现，二者通过pybind11交互

### 配置

```shell
# 下载pybind11
git clone https://github.com/pybind/pybind11.git
# 链接python库
sudo apt-get install python3.8-dev
# 编译命令
g++ main.cpp -o main -std=c++11 -fPIC `python3 -m pybind11 --includes` `pkg-config --cflags --libs opencv4` -DPYTHON_API -lpython3.8
```

### cpp调用python

```python
class Inference:
    def __init__(self, in_dim, embed_dim):
        self.in_dim = in_dim
        self.model = PlainNet(in_dim, embed_dim)
        print(f"initial Inference, [in_dim]={in_dim}, [embed_dim]={embed_dim}")
    
    def inference(self):
        x = torch.randn(3, self.in_dim)
        y = self.model(x)
        print(f"inference : {y.shape}")
        return True
    
    def sequare(self, x):
        return x ** 2
        
```

```cpp
# include <pybind11/embed.h>
# include <iostream>

namespace py = pybind11 ;


int main(){
    // initial interpreter of python
    py::scoped_interpreter guard{} ; // 初始化Python解释器

    try{
        // import module
        py::module m = py::module::import("infer") ;
        // import class or function
        py::object inference = m.attr("Inference")(3, 10) ;
        std::cout << "initial basic class named <Inference>" << std::endl ;

        std::cout << "----------------------------------------------" << std::endl ;
        inference.attr("inference")() ;
        std::cout << "execute inference" << std::endl ;
        std::cout << "----------------------------------------------" << std::endl ;
        auto result = inference.attr("sequare")(2).cast<int>() ;
        std::cout << "the sequare of number-2 is " << result << std::endl ;


    }catch (const py::error_already_set& e){
        std::cerr << "Exception occurred: " << e.what() << std::endl ;
    }

    return 0 ;
}
```
### 更新日志

1. [XML文件解析-Python](./pybind_xml.py)：输入文件XML解析和配置文件XML解析 (Nov 19 22：51 解析成功)
2. [模型推理-Python](pybind_model.py): 使用python调用torch进行推理 ()
3. [图像切割部分-C++](main.cpp) (Nov 20 23:17)

### 其他

1. 安装gdal
```shell
pip install proj
pip install geos 
pip install gdal==3.4.3  # 版本与系统版本一致
conda install libffi==3.3 # from osgeo import gdal导入错误，No module named ‘_gdal, libffi
```

2. PyTorch版本要与CUDA保持一致