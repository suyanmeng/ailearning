#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cuda_runtime.h>

struct Box {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

void launch_postprocess_kernel(
    const float* d_model_output,  // 模型输出 GPU 地址
    int num_anchors,              // 8400
    int num_classes,              // 80
    float conf_thresh,            // 0.25
    float iou_thresh,             // 0.45
    float scale, int pad_w, int pad_h, int img_w,
    int img_h,           // 原图宽高
    Box* d_final_boxes,  // 输出：最终框 GPU 地址
    int* d_num_boxes     // 输出：框数量 GPU 地址
);

#endif