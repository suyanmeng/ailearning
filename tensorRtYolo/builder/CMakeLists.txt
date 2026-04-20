cmake_minimum_required(VERSION 3.10)
project(builder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -arch=sm_89)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_89)
# CUDA
find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

# 编译
add_executable(${PROJECT_NAME} src/main.cpp src/builder.cpp)
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)
