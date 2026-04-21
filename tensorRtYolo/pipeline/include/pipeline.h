#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "gpumemorypool.h"
#include "postprocessor.h"
#include "preprocessor.h"
#include "trtengine.h"
#include "visualizer.h"

namespace TensorRTYolo {

class Pipeline {
   public:
    Pipeline(const std::string& engine_path);
    ~Pipeline();

    void run();
    void stop();
    bool setImageDir(const std::string& input_dir, const std::string& output_dir);
    bool setVideoPath(const std::string& video_src_path, const std::string& video_dst_path);

   private:
    // 线程1：读取图像 + 组建Batch
    void threadImageProducer();
    void threadVideoProducer();

    // 线程2：GPU预处理
    void threadPreprocess();

    // 线程3：推理 + 后处理
    void threadInferPost();

    // 工具函数
    void calculateBatchData(BatchData& data);
    void saveResult(BatchData& batch_data, const BatchResult& batch_result);

   private:
    std::string input_dir_;
    std::string output_dir_;
    bool video_flag_ = false;
    cv::VideoCapture cap_;
    cv::VideoWriter writer_;

    std::unique_ptr<PreProcessor> pre_;
    std::unique_ptr<TrtEngine> trt_;
    std::unique_ptr<PostProcessor> post_;
    std::unique_ptr<Visualizer> vis_;

    // GPU 内存池（4块最大显存）
    std::unique_ptr<GPUMemoryPool> gpu_pool_;
    std::thread t_producer_;
    std::thread t_preprocess_;
    std::thread t_infer_post_;

    std::queue<BatchData> batch_queue_;         // 待预处理队列
    std::queue<BatchData> preprocessed_queue_;  // 待推理队列

    std::mutex mtx_batch_;
    std::mutex mtx_prep_;
    std::condition_variable cv_batch_;
    std::condition_variable cv_prep_;

    bool stop_flag_ = false;
};

}  // namespace TensorRTYolo