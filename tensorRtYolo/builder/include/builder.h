#pragma once    
#include <string>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kERROR)
            printf("[TRT ERROR] %s\n", msg);
    }
};
namespace TensorRTYolo {

class EngineBuilder {
public:
    // 这里才接收 ONNX 路径
    bool buildOnnxToEngine(
        const std::string& onnx_path,
        const std::string& save_engine_path
    );
};
} // namespace TensorRTYolo