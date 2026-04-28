#include <iostream>

#include "pipeline.h"
//#define MULTI_IMAGE
#ifdef MULTI_IMAGE
int main(int argc, char** argv) {
    // 1. 传入三个参数：engine路径 + 图片目录 + 保存目录
    std::string engine_path;
    std::string img_dir;
    std::string save_dir;
    if (argc == 4) {
        engine_path = argv[1];
        img_dir = argv[2];
        save_dir = argv[3];
    } else {
        engine_path = "/work/cuda/tensorRtYolo/models/engine/yolov8n.engine";
        img_dir = "/work/cuda/tensorRtYolo/pipeline/pic2/";
        save_dir = "/work/cuda/tensorRtYolo/pipeline/output/";
    }
    TensorRTYolo::Pipeline pipeline(engine_path);
    pipeline.setImageDir(img_dir, save_dir);
    pipeline.run();
    return 0;
}
#endif

#define VEDIO_0
#ifdef VEDIO_0
int main() {
    const std::string engine_path =
        "/work/cuda/tensorRtYolo/models/engine/yolov8n_b16_fp32.engine";
    const std::string mv_path = "/work/cuda/yolo8Test/resource/demo0.mp4";
    const std::string save_mv_path =
        "/work/cuda/tensorRtYolo/pipeline/output/demo0.mp4";
    TensorRTYolo::Pipeline pipeline(engine_path);
    pipeline.setVideoPath(mv_path, save_mv_path);
    pipeline.run();

    return 0;
}
#endif