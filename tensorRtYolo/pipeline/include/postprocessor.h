#pragma once
#include <vector>

#include "common.h"
namespace TensorRTYolo {
class PostProcessor {
   public:
    PostProcessor() = default;
    ~PostProcessor() = default;

    BatchResult batchProcess(const BatchData& batch_data, float* gpu_output);

   private:
    void decode_kernel(float* output, BoxResult* boxes, int batch_size,
                       int num_boxes);
    void nms_kernel(std::vector<BoxResult>& boxes);

   private:
    int out_width_ = 8400;
    int out_height_ = 84;
};
}  // namespace TensorRTYolo