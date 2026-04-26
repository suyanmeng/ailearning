#include "visualizer.h"
namespace TensorRTYolo {
void TensorRTYolo::Visualizer::draw(std::shared_ptr<cv::Mat> img,
                                    const std::vector<BoxResult>& detections) {
    for (const auto& det : detections) {
        // 1. 画检测框
        // std::cout<<"imgsize="<<img->cols<<"x"<<img->rows<<",Drawing box: (" << det.x1 << ", " << det.y1 << ") - ("
        //          << det.x2 << ", " << det.y2 << "), class_id: "
        //          << det.class_id << ", score: " << det.score << std::endl;
        rectangle(*img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2),
                  COCO_COLORS[det.class_id], 2);

        // 2. 拼接标签文字
        std::string label =
            cv::format("%s %.2f", COCO_NAMES[det.class_id].c_str(), det.score);
        int baseline = 0;
        cv::Size label_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

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
        rectangle(*img, cv::Point(label_x, label_y - label_size.height - 5),
                  cv::Point(label_x + label_size.width, label_y),
                  COCO_COLORS[det.class_id], cv::LineTypes::FILLED);

        // 4. 画文字
        putText(*img, label, cv::Point(label_x, label_y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

}  // namespace TensorRTYolo