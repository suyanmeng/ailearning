#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include <vector>

#include "postprocess.h"
using namespace std;

__global__ void decode_kernel(const float* output, int num_anchors,
                              int num_classes, float conf_thresh, float scale,
                              int pad_w, int pad_h, int img_w, int img_h,
                              Box* d_candidates, int* d_num_candidates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors) return;

    // 你自己的输出格式：[4, 80, 8400]
    float cx = output[i];
    float cy = output[num_anchors + i];
    float w = output[2 * num_anchors + i];
    float h = output[3 * num_anchors + i];

    // 最大类别置信度
    float max_conf = 0.0f;
    int class_id = 0;
    for (int j = 0; j < num_classes; j++) {
        float conf = output[(4 + j) * num_anchors + i];
        if (conf > max_conf) {
            max_conf = conf;
            class_id = j;
        }
    }

    // 低置信度过滤
    if (max_conf < conf_thresh) return;

    // 坐标还原
    float x1 = (cx - w * 0.5f - pad_w) / scale;
    float y1 = (cy - h * 0.5f - pad_h) / scale;
    float x2 = (cx + w * 0.5f - pad_w) / scale;
    float y2 = (cy + h * 0.5f - pad_h) / scale;

    // 边界裁剪
    x1 = max(0.0f, x1);
    y1 = max(0.0f, y1);
    x2 = min((float)img_w - 1, x2);
    y2 = min((float)img_h - 1, y2);

    // 原子计数写入候选框
    int pos = atomicAdd(d_num_candidates, 1);
    d_candidates[pos] = {x1, y1, x2, y2, max_conf, class_id};
}

__device__ float iou(float x1, float y1, float x2, float y2, float xx1,
                     float yy1, float xx2, float yy2) {
    float area = (x2 - x1) * (y2 - y1);
    float area2 = (xx2 - xx1) * (yy2 - yy1);

    float inter_x1 = max(x1, xx1);
    float inter_y1 = max(y1, yy1);
    float inter_x2 = min(x2, xx2);
    float inter_y2 = min(y2, yy2);

    float w = max(0.0f, inter_x2 - inter_x1);
    float h = max(0.0f, inter_y2 - inter_y1);
    float inter = w * h;
    return inter / (area + area2 - inter + 1e-6f);
}

__global__ void nms_kernel(Box* d_candidates, int num_candidates, Box* d_output,
                           int* d_num_output, float iou_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_candidates) return;

    Box a = d_candidates[i];
    bool keep = true;

    // 只和分数更高的框做 IOU（标准 NMS 逻辑）
    for (int j = 0; j < i; j++) {
        Box b = d_candidates[j];
        if (a.class_id != b.class_id) continue;
        if (iou(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) > iou_thresh) {
            keep = false;
            break;
        }
    }

    if (keep) {
        int pos = atomicAdd(d_num_output, 1);
        d_output[pos] = a;
    }
}

void launch_postprocess_kernel(
    const float* d_model_output,  // 模型输出 GPU 地址
    int num_anchors,              // 8400
    int num_classes,              // 80
    float conf_thresh,            // 0.25
    float iou_thresh,             // 0.45
    float scale, int pad_w, int pad_h, int img_w, int img_h,  // 原图宽高
    Box* d_final_boxes,  // 输出：最终框 GPU 地址
    int* d_num_boxes     // 输出：框数量 GPU 地址
) {
    const int MAX_CANDIDATES = 1024;
    Box* d_candidates;
    int* d_num_candidates;

    // 分配临时 GPU 内存
    cudaMalloc(&d_candidates, MAX_CANDIDATES * sizeof(Box));
    cudaMalloc(&d_num_candidates, sizeof(int));
    cudaMemset(d_num_candidates, 0, sizeof(int));
    cudaMemset(d_num_boxes, 0, sizeof(int));

    // ========== 第一步：Decode ==========
    int block = 256;
    int grid = (num_anchors + block - 1) / block;
    decode_kernel<<<grid, block>>>(d_model_output, num_anchors, num_classes,
                                   conf_thresh, scale, pad_w, pad_h, img_w,
                                   img_h, d_candidates, d_num_candidates);
    cudaDeviceSynchronize();

    // 获取候选框数量
    int num_candidates = 0;
    cudaMemcpy(&num_candidates, d_num_candidates, sizeof(int),
               cudaMemcpyDeviceToHost);

    cout << "候选框数量（GPU Decode后）：" << num_candidates << endl;
    // ========== 第二步：NMS ==========
    if (num_candidates > 0) {
        int grid_nms = (num_candidates + block - 1) / block;
        nms_kernel<<<grid_nms, block>>>(d_candidates, num_candidates,
                                        d_final_boxes, d_num_boxes, iou_thresh);
        cudaDeviceSynchronize();
    }

    // 释放临时内存
    cudaFree(d_candidates);
    cudaFree(d_num_candidates);
}