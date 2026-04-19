#pragma once
#include <opencv2/opencv.hpp>

#include "common.h"

namespace TensorRTYolo {
class Visualizer {
   public:
    Visualizer() = default;
    ~Visualizer() = default;

    void draw(std::shared_ptr<cv::Mat> img, const std::vector<BoxResult>& detections);

   private:
    const std::vector<std::string> COCO_NAMES = {
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

    const std::vector<cv::Scalar> COCO_COLORS = {
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
};
}  // namespace TensorRTYolo