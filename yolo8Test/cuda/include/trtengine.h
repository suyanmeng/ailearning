#ifndef TRTENGINE_H
#define TRTENGINE_H

#include <NvInfer.h>

#include <opencv2/opencv.hpp>

#include "postprocess.h"
#include "preprocess.h"
using namespace cv;
using namespace nvinfer1;
using namespace std;
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) printf("[TRT ERROR] %s\n", msg);
    }
};
struct TrtImage {
    string name;
    Mat mat;
};

inline Logger gLogger;
const vector<string> COCO_NAMES = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

const cv::Scalar COCO_COLORS[] = {
    cv::Scalar(0, 255, 0),      // 0: 绿色
    cv::Scalar(255, 0, 0),      // 1: 蓝色
    cv::Scalar(0, 0, 255),      // 2: 红色
    cv::Scalar(255, 255, 0),    // 3: 青色
    cv::Scalar(255, 0, 255),    // 4: 紫色
    cv::Scalar(0, 255, 255),    // 5: 黄色
    cv::Scalar(255, 255, 255),  // 6: 白色
    cv::Scalar(128, 0, 0),     cv::Scalar(0, 128, 0),
    cv::Scalar(0, 0, 128),     cv::Scalar(128, 128, 0),
    cv::Scalar(128, 0, 128),   cv::Scalar(0, 128, 128),
    cv::Scalar(128, 128, 128), cv::Scalar(255, 128, 0),
    cv::Scalar(255, 0, 128),   cv::Scalar(128, 255, 0),
    cv::Scalar(0, 255, 128),   cv::Scalar(128, 0, 255),
    cv::Scalar(0, 128, 255),   cv::Scalar(255, 128, 128),
    cv::Scalar(128, 255, 128), cv::Scalar(128, 128, 255),
    cv::Scalar(255, 255, 128), cv::Scalar(255, 128, 255),
    cv::Scalar(128, 255, 255), cv::Scalar(50, 150, 50),
    cv::Scalar(150, 50, 50),   cv::Scalar(50, 50, 150),
    cv::Scalar(150, 150, 50),  cv::Scalar(150, 50, 150),
    cv::Scalar(50, 150, 150),  cv::Scalar(200, 100, 50),
    cv::Scalar(100, 200, 50),  cv::Scalar(50, 100, 200),
    cv::Scalar(100, 50, 200),  cv::Scalar(200, 50, 100),
    cv::Scalar(50, 200, 100),  cv::Scalar(80, 180, 20),
    cv::Scalar(180, 80, 20),   cv::Scalar(20, 180, 80),
    cv::Scalar(80, 20, 180),   cv::Scalar(180, 20, 80),
    cv::Scalar(20, 80, 180),   cv::Scalar(255, 64, 64),
    cv::Scalar(64, 255, 64),   cv::Scalar(64, 64, 255),
    cv::Scalar(255, 255, 64),  cv::Scalar(255, 64, 255),
    cv::Scalar(64, 255, 255),  cv::Scalar(100, 200, 255),
    cv::Scalar(255, 100, 200), cv::Scalar(200, 255, 100),
    cv::Scalar(100, 200, 200), cv::Scalar(200, 100, 200),
    cv::Scalar(200, 200, 100), cv::Scalar(150, 255, 0),
    cv::Scalar(0, 150, 255),   cv::Scalar(255, 0, 150),
    cv::Scalar(150, 0, 255),   cv::Scalar(255, 150, 0),
    cv::Scalar(0, 255, 150),   cv::Scalar(255, 200, 0),
    cv::Scalar(0, 255, 200),   cv::Scalar(200, 0, 255),
    cv::Scalar(255, 0, 200),   cv::Scalar(200, 255, 0),
    cv::Scalar(0, 200, 255),   cv::Scalar(100, 0, 255),
    cv::Scalar(255, 100, 0),   cv::Scalar(0, 255, 100),
    cv::Scalar(100, 255, 0),   cv::Scalar(0, 100, 255),
    cv::Scalar(255, 0, 100),   cv::Scalar(50, 0, 0),
    cv::Scalar(0, 50, 0),      cv::Scalar(0, 0, 50),
    cv::Scalar(50, 50, 0),     cv::Scalar(50, 0, 50),
    cv::Scalar(0, 50, 50)};

class trtEngine {
   public:
    trtEngine(const std::string& engine_path, const std::string& onnx_path);

    vector<vector<Box>> infer(const vector<TrtImage>& imgs);
    void drawDetections(Mat& img, const vector<Box>& detections);
    ~trtEngine();

   private:
    void buildEngine(const std::string& engine_file,
                     const std::string& onnx_path);
    void initEngine(const std::string& engine_path);
    void printEngineInfo10x(ICudaEngine* engine, IExecutionContext* ctx);
    void calculate_scale_pad(int src_w, int src_h, int dst_w, int dst_h,
                             float& scale, int& pad_w, int& pad_h);
    void* buffers_[2];  // 输入输出缓冲区指针数组
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    int input_channels_ = 3;
    int input_width_ = 640;
    int input_height_ = 640;
    int out_width_ = 8400;
    int out_height_ = 84;
};

#endif  // TRTENGINE_H