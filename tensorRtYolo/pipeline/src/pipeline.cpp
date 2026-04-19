#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <map>
namespace TensorRTYolo {

Pipeline::Pipeline(const std::string& engine_path) {
    pre_ = std::make_unique<PreProcessor>();
    trt_ = std::make_unique<TrtEngine>();
    trt_->load_engine(engine_path);
    post_ = std::make_unique<PostProcessor>();
    vis_ = std::make_unique<Visualizer>();
}

Pipeline::~Pipeline() {}
void Pipeline::run() {
    auto batch_queue = getBatchData();
    while (!batch_queue.empty()) {
        auto batch_data = batch_queue.front();
        batch_queue.pop();
        pre_->batchProcess(batch_data, (float*)trt_->getInputBuffer());
        trt_->set_dynamic_batch(batch_data.images.size());
        trt_->infer();
        auto result =
            post_->batchProcess(batch_data, (float*)trt_->getOutputBuffer());
        saveResult(batch_data, result);
    }
}
void Pipeline::stop() {}
bool Pipeline::setInputDir(const std::string& input_dir) {
    if (!fs::exists(input_dir)) {
        std::cerr << "输入目录不存在: " << input_dir << std::endl;
        return false;
    }
    input_dir_ = input_dir;
    return true;
}

void Pipeline::setOutputDir(const std::string& output_dir) {
    if (!fs::exists(output_dir)) {
        std::cerr << "输出目录不存在，正在创建：" << output_dir << std::endl;
        fs::create_directories(fs::path(output_dir));
    }
    output_dir_ = output_dir;
}
std::queue<BatchData> Pipeline::getBatchData() {
    if (!fs::exists(input_dir_)) {
        std::cerr << "输入目录不存在: " << input_dir_ << std::endl;
        return {};
    }
    std::map<std::pair<int, int>, std::queue<ImageData>> w_h_imgs;  // 图片分类
    for (auto& entry : fs::directory_iterator(input_dir_)) {
        if (entry.is_regular_file()) {
            auto img = std::make_shared<cv::Mat>(cv::imread(entry.path().string()));
            if (img->empty()) {
                std::cerr << "❌ 图片读取失败！请检查路径："
                          << entry.path().string() << std::endl;
                continue;
            }
            std::string img_name = entry.path().string().substr(
                entry.path().string().find_last_of("/") + 1);
            w_h_imgs[{img->cols, img->rows}].push({img_name, img});
        }
    }
    if (w_h_imgs.empty()) {
        std::cerr << "❌ 没有找到有效图片！请检查目录：" << input_dir_
                  << std::endl;
        return {};
    }

    long long total_images = 0;
    auto start_time = std::chrono::steady_clock::now();
    std::queue<BatchData> batch_queue;
    for (auto& [w_h, imgs] : w_h_imgs) {
        std::cout << "处理尺寸为 " << w_h.first << "x" << w_h.second
                  << " 的图片，共 " << imgs.size() << " 张" << std::endl;
        total_images += imgs.size();
        int batch_size = imgs.size();
        while (!imgs.empty()) {
            BatchData data;
            for (int i = 0; i < 4 && !imgs.empty(); i++) {
                data.images.push_back(imgs.front());
                imgs.pop();
            }
            calculateScalePad(data);
            batch_queue.push(data);
        }
    }
    return batch_queue;
}
void Pipeline::calculateScalePad(BatchData& data) {
    data.origin_h = data.images[0].mat->rows;
    data.origin_w = data.images[0].mat->cols;
    data.dst_w = trt_->getInputWidth();
    data.dst_h = trt_->getInputHeight();
    data.scale = std::min((float)data.dst_w / data.origin_w,
                          (float)data.dst_h / data.origin_h);
    int new_w = (int)(data.origin_w * data.scale);
    int new_h = (int)(data.origin_h * data.scale);
    data.pad_w = (data.dst_w - new_w) / 2;
    data.pad_h = (data.dst_h - new_h) / 2;
}
void Pipeline::saveResult(BatchData& batch_data,
                          const BatchResult& batch_result) {
    for (int i = 0; i < batch_data.images.size(); i++) {
        vis_->draw(batch_data.images[i].mat, batch_result.imgs_ret[i].boxes);
        std::string save_path = output_dir_ + batch_data.images[i].name;
        std::cout << "保存图片: " << save_path << std::endl;
        cv::imwrite(save_path, *(batch_data.images[i].mat));
    }
}
}  // namespace TensorRTYolo