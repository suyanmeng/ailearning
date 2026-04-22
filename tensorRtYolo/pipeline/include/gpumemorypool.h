#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

#include "common.h"

namespace TensorRTYolo {
class GPUMemoryPool {
   public:
    ~GPUMemoryPool() {
        for (auto& b : buffers_) {
            if (b.gpu_img_) cudaFree(b.gpu_img_);
            if (b.gpu_input) cudaFree(b.gpu_input);
            if (b.gpu_output) cudaFree(b.gpu_output);
            if (b.d_boxes) cudaFree(b.d_boxes);
            if (b.d_box_num) cudaFree(b.d_box_num);
            if (b.d_last_boxes) cudaFree(b.d_last_boxes);
            if (b.d_last_box_num) cudaFree(b.d_last_box_num);
        }
    }

    void init(int num_buffers, int max_batch, size_t max_img_size,
              size_t max_intput_size, size_t max_output_size,
              size_t max_boxes_size) {
        buffers_.resize(num_buffers);
        for (int i = 0; i < num_buffers; ++i) {
            cudaMalloc(&buffers_[i].gpu_img_, max_batch * max_img_size);
            cudaMalloc(&buffers_[i].gpu_input, max_batch * max_intput_size);
            cudaMalloc(&buffers_[i].gpu_output, max_batch * max_output_size);
            cudaMalloc(&buffers_[i].d_boxes, max_batch * max_boxes_size);
            cudaMalloc(&buffers_[i].d_box_num, max_batch * sizeof(int));
            cudaMalloc(&buffers_[i].d_last_boxes, max_batch * max_boxes_size);
            cudaMalloc(&buffers_[i].d_last_box_num, max_batch * sizeof(int));
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