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
    dtype_ = engine_->getTensorDataType(engine_->getIOTensorName(0));
    printEngineInfo();
    return true;
}

std::vector<std::unique_ptr<nvinfer1::IExecutionContext>>
TrtEngine::createExecutionContexts(int num_contexts) {
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;
    for (int i = 0; i < num_contexts; ++i) {
        auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!ctx) {
            std::cerr << "错误：创建执行上下文失败 → " << i << std::endl;
            return {};
        }
        contexts.push_back(std::move(ctx));
    }
    return contexts;
}

void TrtEngine::set_dynamic_batch(int batch_size) {
    context_->setInputShape(engine_->getIOTensorName(0),
                            nvinfer1::Dims4{batch_size, input_channels_,
                                            input_height_, input_width_});
}

void TrtEngine::infer(const std::shared_ptr<const BatchData>& batch_data) {
    batch_data->gpu_buf->ctx->setInputShape(
        engine_->getIOTensorName(0),
        nvinfer1::Dims4{(int64_t)batch_data->images.size(), input_channels_,
                        input_height_, input_width_});
    batch_data->gpu_buf->ctx->setInputTensorAddress(
        engine_->getIOTensorName(0), batch_data->gpu_buf->gpu_input);
    batch_data->gpu_buf->ctx->setOutputTensorAddress(
        engine_->getIOTensorName(1), batch_data->gpu_buf->gpu_output);
    batch_data->gpu_buf->ctx->enqueueV3(batch_data->gpu_buf->cuda_stream);
}
void TrtEngine::printEngineInfo() {
    int nbIOTensors = engine_->getNbIOTensors();
    std::cout << "IO张量总数: " << nbIOTensors << "\n";

    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT);
        const char* ioStr = isInput ? "输入" : "输出";
        nvinfer1::DataType dtype = engine_->getTensorDataType(name);
        const char* dtypeStr = "未知";
        if (dtype == nvinfer1::DataType::kFLOAT) dtypeStr = "FP32";
        if (dtype == nvinfer1::DataType::kHALF) dtypeStr = "FP16";
        if (dtype == nvinfer1::DataType::kINT8) dtypeStr = "INT8";
        nvinfer1::Dims dims = context_->getTensorShape(name);
        std::cout << "----------------------------------------\n";
        std::cout << "索引: " << i << "\n";
        std::cout << "名称: " << name << "\n";
        std::cout << "类型: " << ioStr << "\n";
        std::cout << "数据类型: " << dtypeStr << "\n";
        std::cout << "形状: ";
        for (int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "----------------------------------------\n";
}
size_t TrtEngine::getSizeByDataType(nvinfer1::DataType dtype) const {
    if (dtype == nvinfer1::DataType::kFLOAT) return sizeof(float);
    if (dtype == nvinfer1::DataType::kHALF) return sizeof(__half);
    if (dtype == nvinfer1::DataType::kINT8) return sizeof(int8_t);
    return size_t();
}
}  // namespace TensorRTYolo