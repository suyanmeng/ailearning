#include <iostream>

#include "pipeline.h"

void parseArguments(int argc, char** argv, std::string& engine_path,
                    std::string& imgs_dir, std::string& video_path,
                    std::string& output_dir) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " -e <engine> (-i <imgs_dir> | -v <video_path>) [-o <output_dir>] "
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-e" || arg == "--engine") {
            engine_path = argv[++i];
        } else if (arg == "-i" || arg == "--images") {
            imgs_dir = argv[++i];
        } else if (arg == "-v" || arg == "--video") {
            video_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            output_dir = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    std::string engine_path, imgs_dir, video_path, output_dir;
    // ./tensorRtYolo -e /work/cuda/tensorRtYolo/models/engine/yolov8n_b16_fp16.engine -v /work/cuda/yolo8Test/resource/demo0.mp4 -o /work/cuda/tensorRtYolo/pipeline/output/

    parseArguments(argc, argv, engine_path, imgs_dir, video_path, output_dir);
    TensorRTYolo::Pipeline pipeline(engine_path);
    if (imgs_dir.empty() && !video_path.empty()) {
        pipeline.setVideoPath(video_path, output_dir);
    } else if (!imgs_dir.empty() && video_path.empty()) {
        pipeline.setImageDir(imgs_dir, output_dir);
    } else {
        std::cerr
            << "Please specify either a picture directory or a video file."
            << std::endl;
        return EXIT_FAILURE;
    }
    pipeline.run();
    return 0;
}

// #define MULTI_IMAGE
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

// #define VEDIO_0
#ifdef VEDIO_0
int main() {
    const std::string engine_path =
        "/work/cuda/tensorRtYolo/models/engine/yolov8n_b16_fp16.engine";
    const std::string mv_path = "/work/cuda/yolo8Test/resource/demo0.mp4";
    const std::string save_mv_path =
        "/work/cuda/tensorRtYolo/pipeline/output/demo0.mp4";
    TensorRTYolo::Pipeline pipeline(engine_path);
    pipeline.setVideoPath(mv_path, save_mv_path);
    pipeline.run();

    return 0;
}
#endif