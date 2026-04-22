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

    // 初始化 GPU 内存池（4块最大Batch4显存）
    gpu_pool_ = std::make_unique<GPUMemoryPool>();
    gpu_pool_->init(trt_->getMaxBufferNum(), trt_->getMaxBatchSize(),
                    trt_->getImgMaxSupportSize(), trt_->getInputMaxSize(),
                    trt_->getOutputMaxSize(), trt_->getMaxBoxesSize());
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::run() {
    std::cout << "[Pipeline] 启动 3 线程 GPU 流水线" << std::endl;

    // 启动 3 个线程
    if (video_flag_) {
        t_producer_ = std::thread(&Pipeline::threadVideoProducer, this);
    } else {
        t_producer_ = std::thread(&Pipeline::threadImageProducer, this);
    }
    t_preprocess_ = std::thread(&Pipeline::threadPreprocess, this);
    t_infer_post_ = std::thread(&Pipeline::threadInferPost, this);

    // 等待线程结束
    if (t_producer_.joinable()) t_producer_.join();
    if (t_preprocess_.joinable()) t_preprocess_.join();
    if (t_infer_post_.joinable()) t_infer_post_.join();

    std::cout << "[Pipeline] 全部任务完成" << std::endl;
}

void Pipeline::stop() {
    stop_flag_ = true;
    cv_batch_.notify_all();
    cv_prep_.notify_all();
}

bool Pipeline::setImageDir(const std::string& input_dir,
                           const std::string& output_dir) {
    if (!fs::exists(input_dir)) {
        std::cerr << "输入目录不存在: " << input_dir << std::endl;
        return false;
    }
    if (!fs::exists(output_dir)) {
        if (fs::create_directories(fs::path(output_dir))) {
            std::cout << "输出目录不存在，创建成功: " << output_dir
                      << std::endl;
        } else {
            std::cerr << "输出目录不存在，创建失败: " << output_dir
                      << std::endl;
            return false;
        }
    }
    input_dir_ = input_dir;
    output_dir_ = output_dir;
    video_flag_ = false;
    return true;
}

bool Pipeline::setVideoPath(const std::string& video_src_path,
                            const std::string& video_dst_path) {
    if (!fs::exists(video_src_path)) {
        std::cerr << "视频源路径不存在: " << video_src_path << std::endl;
        return false;
    }
    if (!cap_.open(video_src_path)) {
        std::cerr << "视频打开失败: " << video_src_path << std::endl;
        return false;
    }
    if (!fs::exists(fs::path(video_dst_path).parent_path())) {
        if (fs::create_directories(fs::path(video_dst_path).parent_path())) {
            std::cout << "输出目录不存在，创建成功: " << video_dst_path
                      << std::endl;
        } else {
            std::cerr << "输出目录不存在，创建失败: " << video_dst_path
                      << std::endl;
            return false;
        }
    }
    std::cout << "视频打开成功: " << video_src_path << std::endl;
    int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap_.get(cv::CAP_PROP_FPS);
    int fourcc = cv::VideoWriter::fourcc(
        'm', 'p', '4', 'v');  // mp4v是常用的编码器，兼容性较好
    writer_ =
        cv::VideoWriter(video_dst_path, fourcc, fps, cv::Size(width, height));

    video_flag_ = true;
    return true;
    return false;
}

// 线程1：生产Batch（读图片）
void Pipeline::threadImageProducer() {
    if (!fs::exists(input_dir_)) return;

    std::map<std::pair<int, int>, std::queue<ImageData>> w_h_imgs;
    for (auto& entry : fs::directory_iterator(input_dir_)) {
        if (entry.is_regular_file()) {
            auto img =
                std::make_shared<cv::Mat>(cv::imread(entry.path().string()));
            if (img->empty()) continue;

            std::string img_name = entry.path().string().substr(
                entry.path().string().find_last_of("/") + 1);
            w_h_imgs[{img->cols, img->rows}].push({img_name, img});
        }
    }

    for (auto& [w_h, imgs] : w_h_imgs) {
        while (!imgs.empty()) {
            BatchData data;
            for (int i = 0; i < 4 && !imgs.empty(); i++) {
                data.images.push_back(imgs.front());
                imgs.pop();
            }

            // 推入队列
            std::lock_guard<std::mutex> lock(mtx_batch_);
            batch_queue_.push(data);
            cv_batch_.notify_one();
        }
    }

    // 生产结束
    std::lock_guard<std::mutex> lock(mtx_batch_);
    stop_flag_ = true;
    cv_batch_.notify_all();
    std::cout << "[线程1] 生产完成" << std::endl;
}

void Pipeline::threadVideoProducer() {
    if (!cap_.isOpened()) return;

    cv::Mat frame;
    int cnt = 0;
    std::queue<ImageData> imgs;
    while (cap_.read(frame) && !frame.empty()) {
        cnt++;
        imgs.push(
            {std::to_string(cnt), std::make_shared<cv::Mat>(frame.clone())});
        if (cnt == 4) {
            cnt = 0;
            BatchData data;
            while (!imgs.empty()) {
                data.images.push_back(imgs.front());
                imgs.pop();
            }
            // 推入Batch队列（线程安全）
            {
                std::lock_guard<std::mutex> lock(mtx_batch_);
                batch_queue_.push(data);
                cv_batch_.notify_one();
            }
        }
    }
    if (!imgs.empty()) {
        BatchData data;
        while (!imgs.empty()) {
            data.images.push_back(imgs.front());
            imgs.pop();
        }
        // 推入Batch队列（线程安全）
        {
            std::lock_guard<std::mutex> lock(mtx_batch_);
            batch_queue_.push(data);
            cv_batch_.notify_one();
        }
    }
    // 生产结束
    std::lock_guard<std::mutex> lock(mtx_batch_);
    stop_flag_ = true;
    cv_batch_.notify_all();
    std::cout << "[视频线程] 读帧完成" << std::endl;
}

// 线程2：预处理（从内存池取显存）
void Pipeline::threadPreprocess() {
    while (true) {
        if (stop_flag_ && batch_queue_.empty()) break;

        std::unique_lock<std::mutex> lock(mtx_batch_);
        cv_batch_.wait(
            lock, [this]() { return !batch_queue_.empty() || stop_flag_; });

        if (batch_queue_.empty() && stop_flag_) break;

        // 取一个Batch
        auto batch_data = batch_queue_.front();
        batch_queue_.pop();
        lock.unlock();
        calculateBatchData(batch_data);
        // std::cout << "处理尺寸为 " << batch_data.src_w << "x"
        //           << batch_data.src_h << " 的图片，共 "
        //           << batch_data.images.size() << " 张" << std::endl;

        // GPU 预处理 → 直接写入推理输入显存
        pre_->batchProcess(batch_data, batch_data.gpu_buf->gpu_input);

        std::lock_guard<std::mutex> lock2(mtx_prep_);
        preprocessed_queue_.push(batch_data);
        cv_prep_.notify_one();
    }

    std::lock_guard<std::mutex> lock2(mtx_prep_);
    stop_flag_ = true;
    cv_prep_.notify_all();
    std::cout << "[线程2] 预处理线程退出" << std::endl;
}

// 线程3：推理 + 后处理（使用内存池 +回收）
void Pipeline::threadInferPost() {
    while (true) {
        if (stop_flag_ && preprocessed_queue_.empty()) break;

        std::unique_lock<std::mutex> lock(mtx_prep_);
        cv_prep_.wait(lock, [this]() {
            return !preprocessed_queue_.empty() || stop_flag_;
        });

        if (preprocessed_queue_.empty() && stop_flag_) break;

        auto batch_data = preprocessed_queue_.front();
        preprocessed_queue_.pop();
        lock.unlock();

        // 推理
        trt_->set_dynamic_batch(batch_data.images.size());
        trt_->infer(batch_data.gpu_buf);

        // 后处理
        auto result =
            post_->batchProcess(batch_data, batch_data.gpu_buf->gpu_output);

        saveResult(batch_data, result);

        // 归还 GPU 显存
        gpu_pool_->free(batch_data.gpu_buf);
    }
    if (video_flag_) {
        cap_.release();
        writer_.release();
        video_flag_ = false;
    }
    std::cout << "[线程3] 推理线程退出" << std::endl;
}

void Pipeline::calculateBatchData(BatchData& data) {
    data.src_h = data.images[0].mat->rows;
    data.src_w = data.images[0].mat->cols;
    data.src_support_max_size = trt_->getImgMaxSupportSize();
    data.dst_w = trt_->getInputWidth();
    data.dst_h = trt_->getInputHeight();
    data.scale = std::min((float)data.dst_w / data.src_w,
                          (float)data.dst_h / data.src_h);
    int new_w = (int)(data.src_w * data.scale);
    int new_h = (int)(data.src_h * data.scale);
    data.pad_w = (data.dst_w - new_w) / 2;
    data.pad_h = (data.dst_h - new_h) / 2;
    data.gpu_buf = gpu_pool_->allocate();
}

void Pipeline::saveResult(BatchData& batch_data,
                          const BatchResult& batch_result) {
    for (int i = 0; i < batch_data.images.size(); i++) {
        vis_->draw(batch_data.images[i].mat, batch_result.imgs_ret[i].boxes);
        if (video_flag_) {
            writer_.write(*(batch_data.images[i].mat));
        } else {
            std::string save_path = output_dir_ + batch_data.images[i].name;
            std::cout << "保存图片: " << save_path << std::endl;
            cv::imwrite(save_path, *(batch_data.images[i].mat));
        }
    }
}

}  // namespace TensorRTYolo