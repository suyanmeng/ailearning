cmake_minimum_required(VERSION 3.10)
project(PreprocessTest)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 强制要求 Debug 构建类型（如果没有指定，默认设为 Debug）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 查找CUDA和OpenCV
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# 注意：即使处于 Debug 模式，也建议保留 -arch=sm_89(40显卡架构），否则可能使用默认架构导致性能/兼容问题
# 因此将架构标志同时加到 Debug 中
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -arch=sm_89)
endif()

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 编译CUDA Kernel文件（生成对象库）
cuda_add_library(preprocess_kernel STATIC
    src/preprocess_kernel.cu
)

# 生成可执行文件
add_executable(preprocess_test
    src/preprocess_test.cpp
)

# 链接库
target_link_libraries(preprocess_test
    preprocess_kernel
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

target_compile_options(preprocess_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>
)