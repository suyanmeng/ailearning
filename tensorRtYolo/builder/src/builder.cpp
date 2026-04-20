#include "builder.h"

#include <fstream>
#include <iostream>
#include <vector>
Logger gLogger;
namespace TensorRTYolo {
bool EngineBuilder::buildOnnxToEngine(const std::string& onnx_path,
                                      const std::string& save_engine_path) {
    std::ifstream f(save_engine_path);
    if (f.good()) {
        std::cout << "引擎已存在，跳过构建：" << save_engine_path << std::endl;
        return true;
    }

    std::cout << "引擎不存在，开始从 ONNX 构建引擎..." << std::endl;
    // ======================
    // 1. 创建 TRT 组件
    // ======================
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(0U);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    auto config = builder->createBuilderConfig();

    // ======================
    // 2. 读取并解析 ONNX
    // ======================
    std::ifstream onnx_file(onnx_path, std::ios::binary);
    if (!onnx_file) {
        std::cerr << "打开 ONNX 文件失败：" << onnx_path << std::endl;
        return false;
    }

    std::vector<char> onnx_buf(std::istreambuf_iterator<char>(onnx_file), {});
    bool parsed = parser->parse(onnx_buf.data(), onnx_buf.size());

    if (!parsed) {
        std::cerr << "解析 ONNX 失败！" << std::endl;
        int nb_err = parser->getNbErrors();
        for (int i = 0; i < nb_err; ++i) {
            auto* err = parser->getError(i);
            std::cerr << "ONNX 错误 " << i << ": " << err->desc() << std::endl;
        }
        return false;
    }

    // ======================================================
    // 打印 ONNX 详细信息
    // ======================================================
    std::cout << "\n=============================================" << std::endl;
    std::cout << "              ONNX 模型详细信息              " << std::endl;
    std::cout << "=============================================" << std::endl;

    int nb_inputs = network->getNbInputs();
    std::cout << "输入节点数量: " << nb_inputs << std::endl;
    for (int i = 0; i < nb_inputs; ++i) {
        auto* tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        std::cout << "  输入[" << i << "]: " << tensor->getName() << std::endl;
        std::cout << "    形状: ";
        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << dims.d[d] << " ";
        }
        std::cout << "\n    数据类型: " << (int)tensor->getType()
                  << " (FP32=0, FP16=1, INT8=2)" << std::endl;
    }

    int nb_outputs = network->getNbOutputs();
    std::cout << "\n输出节点数量: " << nb_outputs << std::endl;
    for (int i = 0; i < nb_outputs; ++i) {
        auto* tensor = network->getOutput(i);
        auto dims = tensor->getDimensions();
        std::cout << "  输出[" << i << "]: " << tensor->getName() << std::endl;
        std::cout << "    形状: ";
        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << dims.d[d] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n网络总层数: " << network->getNbLayers() << std::endl;

    bool is_dynamic_batch = false;
    if (nb_inputs > 0) {
        auto dims = network->getInput(0)->getDimensions();
        is_dynamic_batch = (dims.d[0] == -1);
    }
    std::cout << "是否动态 Batch: " << (is_dynamic_batch ? "是 ✅" : "否 ❌")
              << std::endl;
    std::cout << "=============================================\n" << std::endl;

    // ======================
    // 3. 动态 Batch 配置
    // ======================
    if (is_dynamic_batch) {
        std::cout << "配置动态 Batch 优化配置文件..." << std::endl;
        auto profile = builder->createOptimizationProfile();
        const char* name = network->getInput(0)->getName();

        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4{4, 3, 640, 640});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4{8, 3, 640, 640});

        config->addOptimizationProfile(profile);
    }

    // ======================
    // 4. 构建配置
    // ======================
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               4ULL << 30);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // ======================
    // 5. 构建 & 保存引擎
    // ======================
    nvinfer1::IHostMemory* serialized =
        builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        std::cerr << "引擎构建失败！" << std::endl;
        return false;
    }

    std::ofstream engine_file(save_engine_path, std::ios::binary);
    engine_file.write(static_cast<char*>(serialized->data()),
                      serialized->size());
    engine_file.close();

    std::cout << "✅ 引擎构建成功！保存到: " << save_engine_path << std::endl;
    return true;
}
}  // namespace TensorRTYolo