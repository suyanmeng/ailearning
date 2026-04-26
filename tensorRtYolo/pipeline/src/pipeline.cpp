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
    gpu_pool_->init(trt_->getMaxBufferNum(), trt_->getMaxBatch(),
                    trt_->getImgMaxSupportSize(), trt_->getInputMaxSize(),
                    trt_->getOutputMaxSize(), trt_->getMaxBoxesSize(),
                    trt_->createExecutionContexts(trt_->getMaxBufferNum()));
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::run() {
    std::cout << "[Pipeline] 启动 3 线程 GPU 流水线" << std::endl;
    start_time_ = std::chrono::steady_clock::now();
    total_images_=0;
    // 启动 3 个线程
    if (video_flag_) {
        t_producer_ = std::thread(&Pipeline::threadVideoProducer, this);
    } else {
        t_producer_ = std::thread(&Pipeline::threadImageProducer, this);
    }
    t_preprocess_ = std::thread(&Pipeline::threadInfer, this);
    t_infer_post_ = std::thread(&Pipeline::threadVisSave, this);

    // 等待线程结束
    if (t_producer_.joinable()) t_producer_.join();
    if (t_preprocess_.joinable()) t_preprocess_.join();
    if (t_infer_post_.joinable()) t_infer_post_.join();

    std::cout << "[Pipeline] 全部任务完成" << std::endl;
}

void Pipeline::stop() {
    infer_stop_flag_ = true;
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
            total_images_++;
            std::string img_name = entry.path().string().substr(
                entry.path().string().find_last_of("/") + 1);
            w_h_imgs[{img->cols, img->rows}].push({img_name, img});
        }
    }

    for (auto& [w_h, imgs] : w_h_imgs) {
        while (!imgs.empty()) {
            BatchData data;
            for (int i = 0; i < trt_->getMaxBatch() && !imgs.empty(); i++) {
                // std::cout << "读取图片: " << imgs.front().name << " 尺寸: " << w_h.first
                //           << "x" << w_h.second << std::endl;
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
    prod_stop_flag_ = true;
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
        total_images_++;
        imgs.push(
            {std::to_string(cnt), std::make_shared<cv::Mat>(std::move(frame))});
        if (cnt == trt_->getMaxBatch()) {
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
    prod_stop_flag_ = true;
    cv_batch_.notify_all();
    std::cout << "[视频线程] 读帧完成" << std::endl;
}

// 线程2：GPU预处理 + 推理 + 后处理
void Pipeline::threadInfer() {
    while (true) {
        if (prod_stop_flag_ && batch_queue_.empty()) break;

        std::unique_lock<std::mutex> lock(mtx_batch_);
        cv_batch_.wait(
            lock, [this]() { return !batch_queue_.empty() || prod_stop_flag_; });

        if (batch_queue_.empty() && prod_stop_flag_) break;

        // 取一个Batch
        std::queue<TensorRTYolo::BatchData> batch_queue;
        swap(batch_queue, batch_queue_);
        lock.unlock();
        //std::cout<< "当前待处理batch_queue: " << batch_queue.size() << std::endl;
        while (!batch_queue.empty()) {
            std::vector<
                std::pair<TensorRTYolo::BatchData, TensorRTYolo::BatchResult>>
                preprocessed_queue;
            //std::cout << "当前待处理---------------- "  << std::endl;
            for (int i = 0; i < trt_->getMaxBufferNum() && !batch_queue.empty();
                 i++) {
                auto batch_data = batch_queue.front();
                batch_queue.pop();
                calculateBatchData(batch_data);
                // std::cout << "处理尺寸为 " << batch_data.src_w << "x"
                //           << batch_data.src_h << " 的图片，共 "
                //           << batch_data.images.size() << " 张" << std::endl;

                // GPU 预处理 → 直接写入推理输入显存
                pre_->batchProcess(batch_data);
                trt_->infer(batch_data);
                post_->batchProcess(batch_data);
                preprocessed_queue.push_back(std::move(std::make_pair(batch_data, BatchResult())));
            }
            for(int i =0;i<preprocessed_queue.size();i++){
                CUDA_CHECK(cudaStreamSynchronize(preprocessed_queue[i].first.gpu_buf->cuda_stream));
            }
            //获取结果
            for(auto& pair : preprocessed_queue){
                auto& batch_data = pair.first;
                auto& ret = pair.second;
                ret.imgs_ret.reserve(batch_data.images.size());
                for (int i = 0; i < batch_data.images.size(); i++) {
                    ImageResult img_boxes;
                    img_boxes.boxes.reserve(batch_data.gpu_buf->h_last_box_num[i]);
                    for (int j = 0; j < batch_data.gpu_buf->h_last_box_num[i]; j++) {
                        // std::cout<< "Batch " << i << ", Box " << j << ": (" << batch_data.gpu_buf->h_last_boxes[i * trt_->getMaxBoxes() + j].x1
                        //          << ", " << batch_data.gpu_buf->h_last_boxes[i * trt_->getMaxBoxes() + j].y1
                        //          << ") - (" << batch_data.gpu_buf->h_last_boxes[i * trt_->getMaxBoxes() + j].x2
                        //          << ", " << batch_data.gpu_buf->h_last_boxes[i * trt_->getMaxBoxes() + j].y2
                        //          << "), class_id: " << batch_data.gpu_buf->h_last_boxes[i * trt_->getMaxBoxes() + j].class_id
                        //          << ", score: " << batch_data.gpu_buf->h_last_boxes[i * trt_->getMaxBoxes() + j].score
                        //          << std::endl;
                        img_boxes.boxes.push_back(
                            batch_data.gpu_buf->h_last_boxes[i *trt_->getMaxBoxes() + j]);
                    }
                    ret.imgs_ret.push_back(std::move(img_boxes));
                    // std::cout << "图片: " << batch_data.images[i].name << " 检测到 "
                    //         << batch_data.gpu_buf->h_last_box_num[i] << " 个框"
                    //         << std::endl;
                }
            }
            std::lock_guard<std::mutex> lock2(mtx_prep_);
            for (auto& pair : preprocessed_queue) {
                preprocessed_queue_.push(std::move(pair));
            }
            cv_prep_.notify_one();
        }
    }

    std::lock_guard<std::mutex> lock2(mtx_prep_);
    infer_stop_flag_ = true;
    cv_prep_.notify_all();
    std::cout << "[线程2] 推理线程退出" << std::endl;
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    std::cout<<"\n===== 推理线程处理完成 =====\n"<<"总图片数:"<<(int)total_images_<<"总耗时:"<<total_time/1000.0<<" 秒,最终 FPS : "<<total_images_ * 1000.0 / total_time<<std::endl;
}

// 线程3：可视化 + 保存结果
void Pipeline::threadVisSave() {
    while (true) {
        if (infer_stop_flag_ && preprocessed_queue_.empty()) break;

        std::unique_lock<std::mutex> lock(mtx_prep_);
        cv_prep_.wait(lock, [this]() {
            return !preprocessed_queue_.empty() || infer_stop_flag_;
        });

        if (preprocessed_queue_.empty() && infer_stop_flag_) break;

        auto batch = preprocessed_queue_.front();
        preprocessed_queue_.pop();
        lock.unlock();

        saveResult(batch.first, batch.second);

        // 归还 GPU 显存
        gpu_pool_->free(batch.first.gpu_buf);
    }
    if (video_flag_) {
        cap_.release();
        writer_.release();
        video_flag_ = false;
    }
    std::cout << "[线程3] 可视化线程退出" << std::endl;
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    std::cout<<"\n===== 可视化线程处理完成 =====\n"<<"总图片数:"<<(int)total_images_<<"总耗时:"<<total_time/1000.0<<" 秒,最终 FPS : "<<total_images_ * 1000.0 / total_time<<std::endl;
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
        // std::cout << "正在保存图片: " << batch_data.images[i].name <<" 检测到 " << batch_result.imgs_ret[i].boxes.size() << " 个框"
        //           << std::endl;
        vis_->draw(batch_data.images[i].mat, batch_result.imgs_ret[i].boxes);
        if (video_flag_) {
            writer_.write(*(batch_data.images[i].mat));
        } else {
            std::string save_path = output_dir_ + batch_data.images[i].name;
            // std::cout << "保存图片: " << save_path << std::endl;
            cv::imwrite(save_path, *(batch_data.images[i].mat));
        }
    }
}

}  // namespace TensorRTYolo