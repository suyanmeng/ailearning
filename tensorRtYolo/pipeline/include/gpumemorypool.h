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
            if (b.cuda_stream != nullptr) cudaStreamDestroy(b.cuda_stream);
            if (b.cpu_img) cudaFreeHost(b.cpu_img);
            if (b.gpu_img) cudaFree(b.gpu_img);
            if (b.gpu_input) cudaFree(b.gpu_input);
            if (b.gpu_output) cudaFree(b.gpu_output);
            if (b.d_boxes) cudaFree(b.d_boxes);
            if (b.d_box_num) cudaFree(b.d_box_num);
            if (b.d_last_boxes) cudaFree(b.d_last_boxes);
            if (b.d_last_box_num) cudaFree(b.d_last_box_num);
            if (b.h_last_boxes) cudaFreeHost(b.h_last_boxes);
            if (b.h_last_box_num) cudaFreeHost(b.h_last_box_num);
        }
    }

    void init(
        int num_buffers, int max_batch, size_t max_img_size,
        size_t max_intput_size, size_t max_output_size, size_t max_boxes_size,
        std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts) {
        buffers_.resize(num_buffers);
        max_batch_ = max_batch;
        for (int i = 0; i < num_buffers; ++i) {
            CUDA_CHECK(cudaStreamCreate(&buffers_[i].cuda_stream));
            buffers_[i].ctx = std::move(contexts[i]);
            CUDA_CHECK(cudaHostAlloc(&buffers_[i].cpu_img,
                                     max_batch * max_img_size,
                                     cudaHostAllocDefault));
            CUDA_CHECK(
                cudaMalloc(&buffers_[i].gpu_img, max_batch * max_img_size));
            CUDA_CHECK(cudaMalloc(&buffers_[i].gpu_input,
                                  max_batch * max_intput_size));
            CUDA_CHECK(cudaMalloc(&buffers_[i].gpu_output,
                                  max_batch * max_output_size));
            CUDA_CHECK(
                cudaMalloc(&buffers_[i].d_boxes, max_batch * max_boxes_size));
            CUDA_CHECK(
                cudaMalloc(&buffers_[i].d_box_num, max_batch * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&buffers_[i].d_last_boxes,
                                  max_batch * max_boxes_size));
            CUDA_CHECK(cudaMalloc(&buffers_[i].d_last_box_num,
                                  max_batch * sizeof(int)));
            CUDA_CHECK(cudaHostAlloc(&buffers_[i].h_last_boxes,
                                     max_batch * max_boxes_size,
                                     cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&buffers_[i].h_last_box_num,
                                     max_batch * sizeof(int),
                                     cudaHostAllocDefault));
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
    int max_batch_ = 0;
    std::mutex mtx_;
    std::condition_variable cv_;
};
}  // namespace TensorRTYolo