#pragma once
#include <mutex>
#include <queue>
#include <thread>

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
    bool setInputDir(const std::string& input_dir);
    void setOutputDir(const std::string& output_dir);

   private:
    std::queue<BatchData> getBatchData();
    void calculateScalePad(BatchData& data);
    void saveResult(BatchData& batch_data, const BatchResult& batch_result);

   private:
    std::string input_dir_;
    std::string output_dir_;
    std::unique_ptr<PreProcessor> pre_;
    std::unique_ptr<TrtEngine> trt_;
    std::unique_ptr<PostProcessor> post_;
    std::unique_ptr<Visualizer> vis_;
};
}  // namespace TensorRTYolo