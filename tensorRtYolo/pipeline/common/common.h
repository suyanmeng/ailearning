#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)
namespace cv {
class Mat;
}
namespace TensorRTYolo {
namespace fs = std::filesystem;
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kERROR)
            printf("[TRT ERROR] %s\n", msg);
    }
};

inline Logger gLogger;

// 单张图像数据
struct ImageData {
    std::string name;
    std::shared_ptr<cv::Mat> mat;
};
struct GPUBuffer {
    float* gpu_input = nullptr;
    float* gpu_output = nullptr;
    bool used = false;
};
// 批量输入数据（送给推理）
struct BatchData {
    std::vector<ImageData> images;
    int src_w = 0;       // 原图宽
    int src_h = 0;       // 原图高
    int dst_w = 640;     // 预处理后宽
    int dst_h = 640;     // 预处理后高
    float scale = 1.0f;  // 缩放比例 (new / original)
    int pad_w = 0;       // 左右 padding
    int pad_h = 0;       // 上下 padding
    GPUBuffer* gpu_buf = nullptr;
};
// 检测框结构体
struct BoxResult {
    float x1 = 0.0;
    float y1 = 0.0;
    float x2 = 0.0;
    float y2 = 0.0;
    float score = 0;
    int class_id = -1;
};
// 一个图像的输出结果（推理后）
struct ImageResult {
    std::vector<BoxResult> boxes;
};

// 批量输出结果（推理后）
struct BatchResult {
    std::vector<ImageResult> imgs_ret;
};

}  // namespace TensorRTYolo