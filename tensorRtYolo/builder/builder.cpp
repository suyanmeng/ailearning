#include "builder.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>

#include "common.h"
namespace TensorRTYolo {
bool EngineBuilder::build_onnx_to_engine(const std::string& onnx_path,
                                         const std::string& save_engine_path) {
    std::ifstream f(save_engine_path);
    if (!f.good()) {
        std::cout << "引擎不存在，开始从ONNX构建..." << std::endl;

        auto builder = nvinfer1::createInferBuilder(gLogger);
        auto network = builder->createNetworkV2(0U);
        auto parser = nvonnxparser::createParser(*network, gLogger);
        auto config = builder->createBuilderConfig();

        // 读取 ONNX
        std::ifstream onnx_file(onnx_path, std::ios::binary);
        std::vector<char> onnx_buf((std::istreambuf_iterator<char>(onnx_file)),
                                   std::istreambuf_iterator<char>());
        parser->parse(onnx_buf.data(), onnx_buf.size());
        bool is_dynamic_batch =
            (network->getInput(0)->getDimensions().d[0] == -1);

        if (is_dynamic_batch) {
            // ========== 动态 Batch 配置（parse 之后才能写！） ==========
            auto profile = builder->createOptimizationProfile();

            profile->setDimensions(network->getInput(0)->getName(),
                                   nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims4{1, 3, 640, 640});
            profile->setDimensions(network->getInput(0)->getName(),
                                   nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims4{4, 3, 640, 640});
            profile->setDimensions(network->getInput(0)->getName(),
                                   nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims4{8, 3, 640, 640});

            config->addOptimizationProfile(profile);
        }

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   4ULL << 30);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        // 构建引擎
        nvinfer1::IHostMemory* serialized =
            builder->buildSerializedNetwork(*network, *config);

        // 保存引擎
        std::ofstream engine_file(save_engine_path, std::ios::binary);
        engine_file.write((char*)serialized->data(), serialized->size());
        engine_file.close();

        std::cout << "引擎构建完成！" << std::endl;
    } else {
        std::cout << "引擎已存在，跳过构建。" << std::endl;
    }
    return false;
}
}  // namespace TensorRTYolo