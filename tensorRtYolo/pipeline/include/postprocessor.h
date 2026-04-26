#pragma once
#include <vector>

#include "common.h"
namespace TensorRTYolo {
class PostProcessor {
   public:
    PostProcessor() = default;
    ~PostProcessor() = default;

    void batchProcess(const std::shared_ptr<const BatchData>& batch_data);
};
}  // namespace TensorRTYolo