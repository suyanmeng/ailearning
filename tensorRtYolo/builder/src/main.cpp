#include  "builder.h"

int main() {
    const std::string onnx_path = "/work/cuda/tensorRtYolo/models/onnx/yolov8n.onnx";
    const std::string engine_save_path =
        "/work/cuda/tensorRtYolo/models/engine/yolov8n.engine";
    TensorRTYolo::EngineBuilder builder;
    builder.buildOnnxToEngine(onnx_path, engine_save_path);
    return 0;
}