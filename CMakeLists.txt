cmake_minimum_required(VERSION 3.18)
project(cuda_demo LANGUAGES CXX CUDA)

# 强制 Debug，不可更改
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)

# C++ / CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 🔥🔥🔥 核函数调试必须加的参数（唯一正确写法）
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# 输出到 bin
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 找CUDA
find_package(CUDAToolkit REQUIRED)

# 编译你的 cu 文件
add_executable(${PROJECT_NAME} src/main.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)