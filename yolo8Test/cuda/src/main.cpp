#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "preprocess.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// COCO 80类标准类别名（和Ultralytics完全一致）
const vector<string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR)
            printf("[TRT ERROR] %s\n", msg);
    }
} gLogger;

// 🔴 关键1：完全对齐Ultralytics的预处理（等比例缩放+灰色填充）
// 🔥 🔥 🔥 终极正确预处理：CHW + 归一化 + 对齐YOLOv8 🔥 🔥 🔥
Mat preprocess(const Mat& img, int input_w, int input_h, float& scale, int& pad_w, int& pad_h) {
    scale = min((float)input_w / img.cols, (float)input_h / img.rows);
    int new_w = cvRound(img.cols * scale);
    int new_h = cvRound(img.rows * scale);
    pad_w = (input_w - new_w) / 2;
    pad_h = (input_h - new_h) / 2;

    Mat resized;
    resize(img, resized, Size(new_w, new_h));
    Mat padded(input_h, input_w, CV_8UC3, Scalar(114, 114, 114));
    resized.copyTo(padded(Rect(pad_w, pad_h, new_w, new_h)));
    imwrite("/work/cuda/yolo8Test/cpp/output/temp1.jpg", padded);
    // BGR -> RGB
    cvtColor(padded, padded, COLOR_BGR2RGB);
    imwrite("/work/cuda/yolo8Test/cpp/output/temp2.jpg", padded);
    padded.convertTo(padded, CV_32F, 1.0f / 255.0f);
    imwrite("/work/cuda/yolo8Test/cpp/output/temp3.jpg", padded);

    // ✅✅✅ 关键：HWC -> CHW（必须加！！！）
    vector<Mat> channels(3);
    split(padded, channels);//3通道变1通道，每个通道都是HWC格式的单通道图像，高度和宽度不变，将3个单通道叠加起来，高度就变为了640*3
    Mat chw;
    vconcat(channels[0], channels[1], chw);//RGBRGBRGB->RRRGGGBBB也就实现了HWC -> CHW
    vconcat(chw, channels[2], chw);
    return chw;
}

// 🔴 关键2：完全对齐Ultralytics的后处理（解析[1,84,8400]输出+NMS+坐标还原）
void postprocess(const float* output, const int input_w, const int input_h,
                  const float scale, const int pad_w, const int pad_h,
                  vector<Detection>& detections, float conf_thresh = 0.25f, float nms_thresh = 0.45f) {
    cout<<input_w<<','<<input_h<<','<<scale<<','<<pad_w<<','<<pad_h<<endl;
    // 打印前10个框的坐标和置信度
    cout << "调试：前10个框的输出值：" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "  框" << i << ": cx=" << output[i] 
             << ", cy=" << output[8400+i] 
             << ", conf=" << output[4*8400+i] << endl;
    }

    vector<Rect> boxes;
    vector<float> scores;
    vector<int> class_ids;

    const int num_anchors = 8400;  // 8400个检测框
    const int num_coords = 4;     // 4个坐标(x,y,w,h)
    const int num_classes = 80;   // 80个类别

    // 遍历8400个检测框
    for (int i = 0; i < num_anchors; i++) {
        // 1. 提取坐标：x_center, y_center, w, h（严格按[1,84,8400]排布）
        const float cx = output[i];
        const float cy = output[num_anchors + i];
        const float w = output[2 * num_anchors + i];
        const float h = output[3 * num_anchors + i];

        // 2. 提取最大类别置信度（遍历80个类别）
        float max_conf = 0.0f;
        int class_id = 0;
        for (int j = 0; j < num_classes; j++) {
            const float conf = output[(num_coords + j) * num_anchors + i];
            if (conf > max_conf) {
                max_conf = conf;
                class_id = j;
            }
        }

        // 3. 过滤低置信度框
        if (max_conf < conf_thresh) continue;

        // 4. 坐标还原：先去填充，再等比例放大到原图尺寸
        float x1 = (cx - w / 2.0f - pad_w) / scale;
        float y1 = (cy - h / 2.0f - pad_h) / scale;
        float x2 = (cx + w / 2.0f - pad_w) / scale;
        float y2 = (cy + h / 2.0f - pad_h) / scale;

        // 5. 边界裁剪，防止框超出图片
        x1 = max(0.0f, x1);
        y1 = max(0.0f, y1);
        x2 = min(static_cast<float>(input_w) / scale, x2);
        y2 = min(static_cast<float>(input_h) / scale, y2);

        // 6. 收集有效框
        boxes.emplace_back(Rect(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(max_conf);
        class_ids.push_back(class_id);
    }

    // 7. NMS非极大值抑制，去除重复框
    vector<int> indices;
    dnn::NMSBoxes(boxes, scores, conf_thresh, nms_thresh, indices);

    // 8. 生成最终检测结果
    for (int idx : indices) {
        detections.push_back({
            static_cast<float>(boxes[idx].x),
            static_cast<float>(boxes[idx].y),
            static_cast<float>(boxes[idx].x + boxes[idx].width),
            static_cast<float>(boxes[idx].y + boxes[idx].height),
            scores[idx],
            class_ids[idx]
        });
    }
}

// 🔴 关键3：画框+标签（和Python效果完全一致）
void drawDetections(Mat& img, const vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 画绿色检测框
        rectangle(img, Point(det.x1, det.y1), Point(det.x2, det.y2), Scalar(0, 255, 0), 2);
        cout<<det.x1<<','<<det.y1<<','<<det.x2<<','<<det.y2<<endl;
        // 画标签+置信度（带背景框，文字清晰）
        string label = format("%s %.2f", COCO_NAMES[det.class_id].c_str(), det.conf);
        int baseline = 0;
        Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        rectangle(img, Point(det.x1, det.y1 - label_size.height - 5),
                  Point(det.x1 + label_size.width, det.y1), Scalar(0, 255, 0), FILLED);
        putText(img, label, Point(det.x1, det.y1 - 5),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    }
}

void buildEngine(const string& engine_path, const string onnx_path)
{
    ifstream f(engine_path);
    if (!f.good()) {
        cout << "引擎不存在，开始从ONNX构建..." << endl;

        // ==================== TRT10 正确写法 ====================
        auto builder = createInferBuilder(gLogger);
        auto network = builder->createNetworkV2(0U);  // TRT10 不需要 kEXPLICIT_BATCH
        auto parser = nvonnxparser::createParser(*network, gLogger);
        auto config = builder->createBuilderConfig();

        // 读取 ONNX
        ifstream onnx_file(onnx_path, ios::binary);
        vector<char> onnx_buf((istreambuf_iterator<char>(onnx_file)), istreambuf_iterator<char>());
        parser->parse(onnx_buf.data(), onnx_buf.size());

        // 配置
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4ULL << 30);
        config->setFlag(BuilderFlag::kFP16);

        // ==================== TRT10 关键 API：直接序列化，不返回 engine ====================
        IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);

        // 保存引擎
        ofstream engine_file(engine_path, ios::binary);
        engine_file.write(reinterpret_cast<char*>(serialized->data()), serialized->size());
        engine_file.close();

        cout << "引擎构建完成！" << endl;
    }
}
void calculate_scale_pad(int src_w, int src_h, int dst_w, int dst_h, 
                         float& scale, int& pad_w, int& pad_h) {
    scale = min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    pad_w = (dst_w - new_w) / 2;
    pad_h = (dst_h - new_h) / 2;
}

void debugPreprocess(int dst_h,int dst_w,float* h_dst)
{
    cv::Mat output_img(dst_h, dst_w, CV_32FC3);//32位浮点型3通道
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < dst_h; ++h) {
            for (int w = 0; w < dst_w; ++w) {
                float val = h_dst[c * dst_w * dst_h + h * dst_w + w];
                if (c == 0) output_img.at<cv::Vec3f>(h, w)[2] = val; // R
                else if (c == 1) output_img.at<cv::Vec3f>(h, w)[1] = val; // G
                else output_img.at<cv::Vec3f>(h, w)[0] = val; // B
            }
        }
    }
    cv::Mat output_vis;
    output_img.convertTo(output_vis, CV_8UC3, 255.0);//8位无符号型3通道，计算要用浮点，写图片要用整形
    cv::imwrite("/work/cuda/yolo8Test/cuda/output/tempp.jpg", output_vis);
    printf("Saved output_preprocessed.jpg\n");
}

int main() {
    const string cur_dir = filesystem::current_path().parent_path();
    const string onnx_path = "/work/cuda/yolo8Test/resource/yolov8n.onnx";
    const string engine_path = "/work/cuda/yolo8Test/resource/yolov8n_cpp.engine";
    const string img_path = "/work/cuda/yolo8Test/resource/bus.jpg";
    const string save_path = "/work/cuda/yolo8Test/cuda/output/result.jpg";
    const string temp_path = "/work/cuda/yolo8Test/cuda/output/temp.jpg";

    // 1. 构建/加载TensorRT引擎
    buildEngine(engine_path, onnx_path);
    filesystem::create_directories(filesystem::path(save_path).parent_path());
    // 2. 加载引擎
    ifstream engine_file(engine_path, ios::binary | ios::ate);
    size_t engine_size = engine_file.tellg();
    vector<char> engine_buf(engine_size);
    engine_file.seekg(0);
    engine_file.read(engine_buf.data(), engine_size);

    IRuntime* runtime = createInferRuntime(gLogger);//工厂 创建 TensorRT 运行时环境(日志、资源)
    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_buf.data(), engine_size);//机器 把引擎从内存加载到 GPU
    IExecutionContext* ctx = engine->createExecutionContext();//工人 负责执行推理，可创建多个用来并行

    // 3. 读取图片+预处理
    Mat img = imread(img_path);
    if (img.empty()) {
        cerr << "❌ 图片读取失败！请检查路径：" << img_path << endl;
        return -1;
    }
    const int input_w = 640, input_h = 640;
    float scale = 1.0f;
    int pad_w = 80, pad_h = 0;
    //Mat input = preprocess(img, input_w, input_h, scale, pad_w, pad_h);
    
    //预处理
    int src_w = img.cols, src_h = img.rows;
    calculate_scale_pad(src_w, src_h, input_w, input_h, scale, pad_w, pad_h);
    printf("Source: %dx%d, Target: %dx%d\n", src_w, src_h, input_w, input_h);

    uint8_t* d_src = nullptr;
    void* buffers[2];
    const size_t src_bytes = src_w * src_h * 3 * sizeof(uint8_t);
    const size_t input_size = 3 * input_w * input_h * sizeof(float);
    const size_t output_size = 84 * 8400 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&buffers[0], input_size));
    CUDA_CHECK(cudaMalloc(&buffers[1], output_size));
    CUDA_CHECK(cudaMemcpy(d_src, img.data, src_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto t3 = chrono::high_resolution_clock::now();
    launch_preprocess_kernel(d_src, src_w, src_h, (float*)buffers[0], input_w, input_h, true, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t4 = chrono::high_resolution_clock::now();
    float ms1 = chrono::duration<float, milli>(t4 - t3).count();
    cout << "\n✅ cuda预处理完成！耗时：" << ms1<< " ms" << endl;
    // float* h_dst = (float*)malloc(input_size);debug
    // CUDA_CHECK(cudaMemcpy(h_dst, buffers[0], input_size, cudaMemcpyDeviceToHost));
    // debugPreprocess(input_h,input_w,h_dst);
    // 4. 分配显存+数据传输

    for(int i = 0; i < 10; i++) { // 预热10次，稳定性能
        ctx->executeV2(buffers);
    }
    
    // 5. 推理计时
    auto t1 = chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++) { // 预热10次，稳定性能
        ctx->executeV2(buffers);
    }
    //ctx->enqueueV2(buffers);异步版本，需配合cudaStreamSynchronize同步
    auto t2 = chrono::high_resolution_clock::now();
    float ms = chrono::duration<float, milli>(t2 - t1).count();

    // 6. 推理结果回传CPU
    vector<float> output(84 * 8400);
    CUDA_CHECK(cudaMemcpy(output.data(), buffers[1], output_size, cudaMemcpyDeviceToHost));

    // 7. 后处理+画框
    vector<Detection> detections;
    postprocess(output.data(), input_w, input_h, scale, pad_w, pad_h, detections);
    drawDetections(img, detections);

    // 8. 保存结果
    imwrite(save_path, img);

    cout << "\n✅ 推理完成！耗时：" << ms << " ms" << endl;
    cout << "✅ 检测到有效框数量：" << detections.size() << endl;
    cout << "✅ 结果已保存至：" << save_path << endl;

    // 9. 资源释放
    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    return 0;
}