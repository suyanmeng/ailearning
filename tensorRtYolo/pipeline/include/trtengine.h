#pragma once

#include "common.h"
namespace TensorRTYolo {
class TrtEngine {
   public:
    TrtEngine() = default;
    ~TrtEngine();

    bool load_engine(const std::string& engine_path);
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getInputChannels() const { return input_channels_; }
    int getOutputWidth() const { return out_width_; }
    int getOutputHeight() const { return out_height_; }
    void set_dynamic_batch(int batch_size);
    void infer();
    void* getInputBuffer() const { return buffers_[0]; }
    void* getOutputBuffer() const { return buffers_[1]; }

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    void* buffers_[2] = {nullptr};
    int input_channels_ = 3;
    int input_width_ = 640;
    int input_height_ = 640;
    int out_width_ = 8400;
    int out_height_ = 84;
};
}  // namespace TensorRTYolo