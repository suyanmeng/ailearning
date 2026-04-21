#include "trtenginetest.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>

#include "postprocess.h"
#include "preprocess.h"

trtEngine::trtEngine(const std::string& engine_path,
                     const std::string& onnx_path) {
    buildEngine(engine_path, onnx_path);
    initEngine(engine_path);
    printEngineInfo10x(engine_, context_);
}

vector<vector<Box>> trtEngine::infer(const vector<TrtImage>& imgs) {
    uint8_t* d_src = nullptr;
    const int batch_size = imgs.size();
    int cols = imgs[0].mat.cols;
    int rows = imgs[0].mat.rows;
    const size_t src_bytes = batch_size * cols * rows * 3 * sizeof(uint8_t);
    context_->setInputShape(engine_->getIOTensorName(0),
                            Dims4{batch_size, 3, 640, 640});
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaMemcpy(d_src + i * cols * rows * 3, imgs[i].mat.data,
                              cols * rows * 3 * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
    }
    // CUDA_CHECK(cudaMemcpy(d_src, imgs[0].data, src_bytes,
    // cudaMemcpyHostToDevice));

    float scale;
    int pad_w, pad_h;
    calculate_scale_pad(cols, rows, input_width_, input_height_, scale, pad_w,
                        pad_h);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "scale: " << scale << ", pad_w: " << pad_w << ", pad_h: " << pad_h
         << ", batch_size: " << batch_size << endl;
    // cuda预处理
    launch_preprocess_kernel(d_src, cols, rows, (float*)buffers_[0],
                             input_width_, input_height_, scale, pad_w, pad_h,
                             true, batch_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
    }
    auto t2 = chrono::high_resolution_clock::now();
    float ms1 = chrono::duration<float, milli>(t2 - t1).count();
    float* cpu_input = new float[batch_size * 3 * 640 * 640];
    cudaMemcpy(cpu_input, buffers_[0],
               batch_size * 3 * 640 * 640 * sizeof(float),
               cudaMemcpyDeviceToHost);
    // 打印预处理图片
    for (int b = 0; b < batch_size; b++) {
        // 取出第 b 张预处理后的图：[3, 640, 640] float
        Mat img(640, 640, CV_32FC3);
        float* ptr = cpu_input + b * 3 * 640 * 640;

        // 把 CHW 格式转成 HWC 格式，才能用 OpenCV 显示
        for (int h = 0; h < 640; h++) {
            for (int w = 0; w < 640; w++) {
                float r = ptr[0 * 640 * 640 + h * 640 + w];
                float g = ptr[1 * 640 * 640 + h * 640 + w];
                float b_val = ptr[2 * 640 * 640 + h * 640 + w];

                // 反归一化（如果你预处理做了 /255 就打开）
                r *= 255.0f;
                g *= 255.0f;
                b_val *= 255.0f;

                img.at<Vec3f>(h, w) = Vec3f(b_val, g, r);  // OpenCV: BGR
            }
        }

        // 转成 8bit 图片
        Mat img_8u;
        img.convertTo(img_8u, CV_8UC3);

        // 保存
        char name[64];
        sprintf(name, "debug_preprocessed_batch_%d.jpg", b);
        imwrite(name, img_8u);
        printf("✅ 已保存预处理图片: %s\n", name);
    }
    // 打印前10个值
    // for(int i=0; i<10; i++){
    //     printf("%f ", cpu_input[i*8400]);
    // }
    for (int i = 0; i < batch_size; i++) {
        cout << "第 " << i << " 张图片前10个值: ";
        for (int j = 0; j < 10; j++) {
            printf("%f ",
                   cpu_input[i * 3 * 640 * 640 + 640 * 320 + j]);  // 中间值
        }
        cout << endl;
    }

    // 推理
    t1 = chrono::high_resolution_clock::now();
    context_->executeV2(buffers_);
    t2 = chrono::high_resolution_clock::now();
    auto ms2 = chrono::duration<float, milli>(t2 - t1).count();
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA 错误: %d %s\n%d", err2, cudaGetErrorString(err2));
    }

    // cuda后处理
    const int MAX_BOXES = 1024;
    Box* d_boxes;
    int* d_box_num;
    CUDA_CHECK(cudaMalloc(&d_boxes, batch_size * MAX_BOXES * sizeof(Box)));
    CUDA_CHECK(cudaMalloc(&d_box_num, batch_size * sizeof(int)));
    t1 = chrono::high_resolution_clock::now();
    launch_postprocess_kernel((float*)buffers_[1], out_width_, out_height_ - 4,
                              0.25f, 0.45f, scale, pad_w, pad_h, cols, rows,
                              d_boxes, d_box_num, batch_size);
    t2 = chrono::high_resolution_clock::now();
    auto ms3 = chrono::duration<float, milli>(t2 - t1).count();

    cudaDeviceSynchronize();
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("CUDA 错误: %d %s\n%d", err1, cudaGetErrorString(err1));
    }

    int* box_num = new int[batch_size];
    cudaMemcpy(box_num, d_box_num, batch_size * sizeof(int),
               cudaMemcpyDeviceToHost);
    Box* detections = new Box[batch_size * MAX_BOXES * sizeof(Box)];
    cudaMemcpy(detections, d_boxes, batch_size * MAX_BOXES * sizeof(Box),
               cudaMemcpyDeviceToHost);
    cout << "检测到有效框数量（GPU后处理后）：" << box_num[0] << endl;
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_box_num));
    vector<vector<Box>> results;

    for (int i = 0; i < batch_size; i++) {
        vector<Box> img_boxes;
        for (int j = 0; j < box_num[i]; j++) {
            img_boxes.push_back(detections[i * MAX_BOXES + j]);
        }
        results.push_back(img_boxes);
        cout << "第 " << i << " 张图片检测到 " << box_num[i] << " 个框" << endl;
    }
    return results;
}

void trtEngine::drawDetections(Mat& img, const vector<Box>& detections) {
    for (const auto& det : detections) {
        // 1. 画检测框
        rectangle(img, Point(det.x1, det.y1), Point(det.x2, det.y2),
                  COCO_COLORS[det.class_id], 2);

        // 2. 拼接标签文字
        string label =
            format("%s %.2f", COCO_NAMES[det.class_id].c_str(), det.score);
        int baseline = 0;
        Size label_size =
            getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

        int label_x = det.x1;
        int label_y;
        bool label_inside = false;

        // 如果框太靠上，标签会超出图片 → 把标签画在框内部
        if (det.y1 - label_size.height - 5 < 0) {
            label_y = det.y1 + label_size.height + 5;
            label_inside = true;
        } else {
            label_y = det.y1 - 5;
        }

        // 3. 画标签背景
        rectangle(img, Point(label_x, label_y - label_size.height - 5),
                  Point(label_x + label_size.width, label_y),
                  COCO_COLORS[det.class_id], FILLED);

        // 4. 画文字
        putText(img, label, Point(label_x, label_y - 5), FONT_HERSHEY_SIMPLEX,
                0.6, Scalar(0, 0, 0), 2);
    }
}

trtEngine::~trtEngine() {
    CUDA_CHECK(cudaFree(buffers_[0]));
    CUDA_CHECK(cudaFree(buffers_[1]));
}

void trtEngine::buildEngine(const std::string& engine_path,
                            const std::string& onnx_path) {
    ifstream f(engine_path);
    if (!f.good()) {
        cout << "引擎不存在，开始从ONNX构建..." << endl;

        auto builder = createInferBuilder(gLogger);
        auto network = builder->createNetworkV2(0U);
        auto parser = nvonnxparser::createParser(*network, gLogger);
        auto config = builder->createBuilderConfig();

        // 读取 ONNX
        ifstream onnx_file(onnx_path, ios::binary);
        vector<char> onnx_buf((istreambuf_iterator<char>(onnx_file)),
                              istreambuf_iterator<char>());
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

        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4ULL << 30);
        config->setFlag(BuilderFlag::kFP16);

        // 构建引擎
        IHostMemory* serialized =
            builder->buildSerializedNetwork(*network, *config);

        // 保存引擎
        ofstream engine_file(engine_path, ios::binary);
        engine_file.write((char*)serialized->data(), serialized->size());
        engine_file.close();

        cout << "引擎构建完成！" << endl;
    }
}

void trtEngine::initEngine(const std::string& engine_path) {
    ifstream engine_file(engine_path, ios::binary | ios::ate);
    size_t engine_size = engine_file.tellg();
    vector<char> engine_buf(engine_size);
    engine_file.seekg(0);
    engine_file.read(engine_buf.data(), engine_size);

    runtime_ = createInferRuntime(
        gLogger);  // 工厂 创建 TensorRT 运行时环境(日志、资源)
    engine_ = runtime_->deserializeCudaEngine(
        engine_buf.data(), engine_size);  // 机器 把引擎从内存加载到 GPU
    context_ = engine_->createExecutionContext();  // 工人
    // 负责执行推理，可创建多个用来并行

    Dims dims = context_->getTensorShape(engine_->getIOTensorName(0));
    input_channels_ = dims.d[1];
    input_height_ = dims.d[2];
    input_width_ = dims.d[3];
    dims = context_->getTensorShape(engine_->getIOTensorName(1));
    out_width_ = dims.d[2];
    out_height_ = dims.d[1];
    CUDA_CHECK(cudaMalloc(&buffers_[0], 8 * input_channels_ * input_width_ *
                                            input_height_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&buffers_[1], 8 * out_width_ * out_height_ * sizeof(float)));
}

void trtEngine::printEngineInfo10x(ICudaEngine* engine,
                                   IExecutionContext* ctx) {
    if (!engine || !ctx) return;

    // 1. 获取IO张量总数
    int nbIOTensors = engine->getNbIOTensors();
    std::cout << "IO张量总数: " << nbIOTensors << "\n";

    for (int i = 0; i < nbIOTensors; ++i) {
        // 2. 获取张量名称（核心变化：用name代替index）
        const char* name = engine->getIOTensorName(i);

        // 3. 判断输入/输出
        TensorIOMode mode = engine->getTensorIOMode(name);
        bool isInput = (mode == TensorIOMode::kINPUT);
        const char* ioStr = isInput ? "输入" : "输出";

        // 4. 获取数据类型
        nvinfer1::DataType dtype = engine->getTensorDataType(name);
        const char* dtypeStr = "未知";
        if (dtype == nvinfer1::DataType::kFLOAT) dtypeStr = "FP32";
        if (dtype == nvinfer1::DataType::kHALF) dtypeStr = "FP16";
        if (dtype == nvinfer1::DataType::kINT8) dtypeStr = "INT8";

        // 5. 获取形状（必须从context获取！）
        Dims dims = ctx->getTensorShape(name);

        // 打印
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

void trtEngine::calculate_scale_pad(int src_w, int src_h, int dst_w, int dst_h,
                                    float& scale, int& pad_w, int& pad_h) {
    scale = min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    pad_w = (dst_w - new_w) / 2;
    pad_h = (dst_h - new_h) / 2;
}
