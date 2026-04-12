cmake_minimum_required(VERSION 3.10)
project(yolov8_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -arch=sm_89)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_89)
# CUDA
find_package(CUDA REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT 路径（你已安装）
include_directories(/usr/local/TensorRT-8.6.1.6/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

link_directories(/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib)
cuda_add_library(preprocess_kernel STATIC src/preprocess_kernel.cu)
cuda_add_library(postprocess_kernel STATIC src/postprocess_kernel.cu)
# 编译
add_executable(yolov8_trt src/main.cpp)
target_link_libraries(yolov8_trt
    preprocess_kernel
    postprocess_kernel
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)