#pragma once

#include "common.h"
namespace TensorRTYolo {
class TrtEngine {
   public:
    TrtEngine() = default;
    ~TrtEngine();

    bool load_engine(const std::string& engine_path);
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>>
    createExecutionContexts(int num_contexts);
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getInputChannels() const { return input_channels_; }
    int getOutputWidth() const { return out_width_; }
    int getOutputHeight() const { return out_height_; }
    void set_dynamic_batch(int batch_size);
    void infer(const BatchData& batch_data);
    int getMaxBufferNum() const { return MAX_BUFFER_NUM; }
    int getMaxBatch() const { return MAX_BATCH; }
    size_t getImgMaxSupportSize() const {
        return MAX_IMG_SUPPORT_SIZE * sizeof(uint8_t);
    }
    size_t getInputMaxSize() const {
        return input_channels_ * input_width_ * input_height_ * sizeof(float);
    }
    size_t getOutputMaxSize() const {
        return out_width_ * out_height_ * sizeof(float);
    }
    size_t getMaxBoxesSize() const { return MAX_BOXES * sizeof(BoxResult); }
    size_t getMaxBoxes() const { return MAX_BOXES; }

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    int input_channels_ = 3;
    int input_width_ = 640;
    int input_height_ = 640;
    int out_width_ = 8400;
    int out_height_ = 84;
    const static int MAX_BUFFER_NUM = 6;
    const static int MAX_BATCH = 4;
    const static size_t MAX_IMG_SUPPORT_SIZE =
        1920 * 1080 * 3;  // 支持的最大图片尺寸（字节）
    const static int MAX_BOXES = 1024;
};
}  // namespace TensorRTYolo