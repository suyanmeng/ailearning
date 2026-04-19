#include <iostream>

#include "pipeline.h"
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
        engine_path = "/work/cuda/yolo8Test/resource/yolov8ndynamic_cpp.engine";
        img_dir = "/work/cuda/yolo8Test/picture/";
        save_dir = "/work/cuda/tensorRtYolo/pipeline/output/";
    }
    TensorRTYolo::Pipeline pipeline(engine_path);
    pipeline.setInputDir(img_dir);
    pipeline.setOutputDir(save_dir);
    pipeline.run();
    return 0;
}