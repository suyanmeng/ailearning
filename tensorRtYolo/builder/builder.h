#pragma once    
#include <string>
namespace TensorRTYolo {
class EngineBuilder {
public:
    // 这里才接收 ONNX 路径
    bool build_onnx_to_engine(
        const std::string& onnx_path,
        const std::string& save_engine_path
    );
};
} // namespace TensorRTYolo