#pragma once
#include <opencv2/opencv.hpp>

#include "common.h"
namespace TensorRTYolo {
class PreProcessor {
   public:
    PreProcessor() = default;
    ~PreProcessor() = default;

    // 单图预处理
    //ImageData process(const cv::Mat& img);
    void batchProcess(const BatchData& batch_data);

   private:
    int m_input_w;
    int m_input_h;
};
}  // namespace TensorRTYolo