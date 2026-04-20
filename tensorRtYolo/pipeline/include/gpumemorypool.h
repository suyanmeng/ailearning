#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

#include "common.h"

namespace TensorRTYolo {
class GPUMemoryPool {
   public:

    void init(int num_buffers, size_t max_intput_size,
              size_t max_output_size) {
        buffers_.resize(num_buffers);
        for (int i = 0; i < num_buffers; ++i) {
            cudaMalloc(&buffers_[i].gpu_input, max_intput_size);
            cudaMalloc(&buffers_[i].gpu_output, max_output_size);
            buffers_[i].used = false;
        }
    }

    GPUBuffer* allocate() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() {
            for (auto& b : buffers_)
                if (!b.used) return true;
            return false;
        });

        for (auto& b : buffers_) {
            if (!b.used) {
                b.used = true;
                return &b;
            }
        }
        return nullptr;
    }

    void free(GPUBuffer* buf) {
        std::lock_guard<std::mutex> lock(mtx_);
        buf->used = false;
        cv_.notify_one();
    }

   private:
    std::vector<GPUBuffer> buffers_;
    std::mutex mtx_;
    std::condition_variable cv_;
};
}  // namespace TensorRTYolo