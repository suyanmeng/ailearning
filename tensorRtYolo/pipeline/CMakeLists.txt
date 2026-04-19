cmake_minimum_required(VERSION 3.10)
project(tensorRtYolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -arch=sm_89)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_89)
# CUDA
find_package(CUDA REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT 路径（你已安装）
include_directories(${CMAKE_SOURCE_DIR}/common ${CMAKE_SOURCE_DIR}/include)

cuda_add_library(preprocesskernel STATIC cuda/preprocesskernel.cu)
cuda_add_library(postprocesskernel STATIC cuda/postprocesskernel.cu)
# 编译
add_executable(${PROJECT_NAME} src/main.cpp src/pipeline.cpp src/postprocessor.cpp src/preprocessor.cpp src/trtengine.cpp src/visualizer.cpp)
target_link_libraries(${PROJECT_NAME}
    preprocesskernel
    postprocesskernel
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)
