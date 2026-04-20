#include "trtengine.h"

namespace TensorRTYolo {
TrtEngine::~TrtEngine() {}

bool TrtEngine::load_engine(const std::string& engine_path) {
    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file.is_open()) {
        std::cerr << "错误：无法打开 engine 文件 → " << engine_path
                  << std::endl;
        return false;
    }

    std::streampos engine_size = engine_file.tellg();
    if (engine_size <= 0) {
        std::cerr << "错误：engine 文件为空 → " << engine_path << std::endl;
        engine_file.close();
        return false;
    }
    std::vector<char> engine_buf(engine_size);
    engine_file.seekg(0);
    engine_file.read(engine_buf.data(), engine_size);

    runtime_.reset(nvinfer1::createInferRuntime(
        gLogger));  // 工厂 创建 TensorRT 运行时环境(日志、资源)
    engine_.reset(runtime_->deserializeCudaEngine(
        engine_buf.data(), engine_size));  // 机器 把引擎从内存加载到 GPU
    context_.reset(engine_->createExecutionContext());  // 工人
    // 负责执行推理，可创建多个用来并行

    nvinfer1::Dims dims = context_->getTensorShape(engine_->getIOTensorName(0));
    input_channels_ = dims.d[1];
    input_height_ = dims.d[2];
    input_width_ = dims.d[3];
    dims = context_->getTensorShape(engine_->getIOTensorName(1));
    out_height_ = dims.d[1];
    out_width_ = dims.d[2];
    return true;
}

void TrtEngine::set_dynamic_batch(int batch_size) {
    context_->setInputShape(engine_->getIOTensorName(0),
                            nvinfer1::Dims4{batch_size, input_channels_,
                                            input_height_, input_width_});
}

void TrtEngine::infer(const GPUBuffer* buffer) {
    void* buf[2] = {buffer->gpu_input, buffer->gpu_output};
    context_->executeV2(buf);
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA 错误: %d %s\n", err2, cudaGetErrorString(err2));
    }
}
size_t TrtEngine::getInputMaxSize() {
    return MAX_BATCH * input_channels_ * input_width_ * input_height_ * sizeof(float);
}
size_t TrtEngine::getOutputMaxSize() {
    return MAX_BATCH * out_width_ * out_height_ * sizeof(float);
}
}  // namespace TensorRTYolo