#include  "builder.h"

int main() {
    const std::string onnx_path = "/work/cuda/tensorRtYolo/models/onnx/yolov8n_fp16.onnx";
    const std::string engine_save_path =
        "/work/cuda/tensorRtYolo/models/engine/yolov8n_b16_fp16.engine";
    TensorRTYolo::EngineBuilder builder;
    builder.buildOnnxToEngine(onnx_path, engine_save_path);
    return 0;
}