#pragma once
#include <opencv2/opencv.hpp>

#include "common.h"
namespace TensorRTYolo {
class PreProcessor {
   public:
    PreProcessor() = default;
    ~PreProcessor() = default;

    void batchProcess(const std::shared_ptr<const BatchData>& batch_data);
};
}  // namespace TensorRTYolo