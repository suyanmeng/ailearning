#pragma once

#include "common.h"

void launch_postprocess_kernel(
    const float* d_model_output,  // 模型输出 GPU 地址
    int num_anchors,              // 8400
    int num_classes,              // 80
    float conf_thresh,            // 0.25
    float iou_thresh,             // 0.45
    float scale, int pad_w, int pad_h, int img_w,
    int img_h,                               // 原图宽高
    TensorRTYolo::BoxResult* d_candidates,   // 输出：最终框 GPU 地址
    int* d_num_candidates,                   // 输出：框数量 GPU 地址
    TensorRTYolo::BoxResult* d_final_boxes,  // 输出：最终框 GPU 地址
    int* d_num_boxes,                        // 输出：框数量 GPU 地址
    int batch_size,                          // 批量大小，默认为1
    cudaStream_t stream);
