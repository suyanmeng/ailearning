#include "postprocesskernel.h"

__global__ void decode_kernel(const float* output, int num_anchors,
                              int num_classes, float conf_thresh, float scale,
                              int pad_w, int pad_h, int img_w, int img_h,
                              TensorRTYolo::BoxResult* d_candidates, int* d_num_candidates,
                              int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors * batch_size) return;  // 总anchor = batch × 8400

    // 计算当前属于第几张图
    int b = i / num_anchors;
    int anchor_idx = i % num_anchors;
    const float* feat = output + b * (num_classes + 4) * num_anchors;

    float cx = feat[anchor_idx];
    float cy = feat[num_anchors + anchor_idx];
    float w = feat[2 * num_anchors + anchor_idx];
    float h = feat[3 * num_anchors + anchor_idx];

    // 最大类别置信度
    float max_conf = 0.0f;
    int class_id = 0;
    for (int j = 0; j < num_classes; j++) {
        float conf = feat[(4 + j) * num_anchors + anchor_idx];
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

    // 原子计数写入当前 batch 的候选区
    int pos = atomicAdd(&d_num_candidates[b], 1);
    d_candidates[b * 1024 + pos] = {x1, y1, x2, y2, max_conf, class_id};
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

__global__ void nms_kernel(TensorRTYolo::BoxResult* d_candidates, int* d_num_candidates,
                           TensorRTYolo::BoxResult* d_output, int* d_num_output, float iou_thresh,
                           int batch_size, int max_candidates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_candidates * batch_size;
    if (i >= total) return;

    int b = i / max_candidates;
    int idx = i % max_candidates;
    if (idx >= d_num_candidates[b]) return;

    TensorRTYolo::BoxResult a = d_candidates[b * max_candidates + idx];
    bool keep = true;

    for (int j = 0; j < idx; j++) {
        TensorRTYolo::BoxResult b2 = d_candidates[b * max_candidates + j];
        if (a.class_id != b2.class_id) continue;
        if (iou(a.x1, a.y1, a.x2, a.y2, b2.x1, b2.y1, b2.x2, b2.y2) >
            iou_thresh) {
            keep = false;
            break;
        }
    }

    if (keep) {
        int pos = atomicAdd(&d_num_output[b], 1);
        d_output[b * 1024 + pos] = a;
    }
}

void launch_postprocess_kernel(
    const float* d_model_output,  // 模型输出 GPU 地址
    int num_anchors,              // 8400
    int num_classes,              // 80
    float conf_thresh,            // 0.25
    float iou_thresh,             // 0.45
    float scale, int pad_w, int pad_h, int img_w, int img_h,  // 原图宽高
    TensorRTYolo::BoxResult* d_final_boxes,  // 输出：最终框 GPU 地址[batch,boxs]
    int* d_num_boxes,    // 输出：框数量 GPU 地址[batch,num]
    int batch_size       // 动态batch
) {
    const int MAX_CAND = 1024;  // 每张图最多候选框
    int total_cand = MAX_CAND * batch_size;

    TensorRTYolo::BoxResult* d_candidates;
    int* d_num_candidates;

    // 分配batch式内存
    cudaMalloc(&d_candidates, total_cand * sizeof(TensorRTYolo::BoxResult));
    cudaMalloc(&d_num_candidates, batch_size * sizeof(int));
    cudaMemset(d_num_candidates, 0, batch_size * sizeof(int));
    cudaMemset(d_num_boxes, 0, batch_size * sizeof(int));
    /// ========== 第一步：Decode ==========
    int block = 256;
    int grid_decode = (num_anchors * batch_size + block - 1) / block;

    decode_kernel<<<grid_decode, block>>>(
        d_model_output, num_anchors, num_classes, conf_thresh, scale, pad_w,
        pad_h, img_w, img_h, d_candidates, d_num_candidates, batch_size);

    // int* num_candidates = new int[batch_size]{0};
    // cudaMemcpy(num_candidates, d_num_candidates, batch_size * sizeof(int),
    //            cudaMemcpyDeviceToHost);
    // for (int i = 0; i < batch_size; i++) {
    //     std::cout << "候选框数量（GPU Decode后）：Batch " << i << ": "
    //          << num_candidates[i] << " candidates after decode." << std::endl;
    // }
    // ========== 第二步：NMS ==========
    int grid_nms = (total_cand + block - 1) / block;
    nms_kernel<<<grid_nms, block>>>(d_candidates, d_num_candidates,
                                    d_final_boxes, d_num_boxes, iou_thresh,
                                    batch_size, MAX_CAND);

    cudaFree(d_candidates);
    cudaFree(d_num_candidates);
}