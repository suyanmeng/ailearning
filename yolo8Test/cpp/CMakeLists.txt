cmake_minimum_required(VERSION 3.18)
project(yolov8_trt)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT 路径（你已安装）
include_directories(/usr/local/TensorRT-8.6.1.6/include)
link_directories(/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib)

# 编译
add_executable(yolov8_trt main.cpp)
target_link_libraries(yolov8_trt
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)