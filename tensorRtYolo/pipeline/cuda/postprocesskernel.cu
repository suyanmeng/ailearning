#include "postprocesskernel.h"

template <typename T>
__global__ void decode_kernel(const T* output, int num_anchors, int num_classes,
                              float conf_thresh, float scale, int pad_w,
                              int pad_h, int img_w, int img_h,
                              TensorRTYolo::BoxResult* d_candidates,
                              int* d_num_candidates, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors * batch_size) return;  // 总anchor = batch × 8400

    // 计算当前属于第几张图
    int b = i / num_anchors;
    int anchor_idx = i % num_anchors;
    const T* feat = output + b * (num_classes + 4) * num_anchors;

    auto to_float = [](const T& val) {
        if constexpr (std::is_same_v<T, __half>) {
            return __half2float(val);
        } else {
            return (float)val;
        }
    };

    float cx = to_float(feat[anchor_idx]);
    float cy = to_float(feat[num_anchors + anchor_idx]);
    float w = to_float(feat[2 * num_anchors + anchor_idx]);
    float h = to_float(feat[3 * num_anchors + anchor_idx]);

    // 最大类别置信度
    float max_conf = 0.0f;
    int class_id = 0;
    for (int j = 0; j < num_classes; j++) {
        float conf = to_float(feat[(4 + j) * num_anchors + anchor_idx]);
        if (conf > max_conf) {
            max_conf = conf;
            class_id = j;
        }
    }

    if (max_conf < conf_thresh) return;

    // 坐标还原
    float x1 = (cx - w * 0.5f - pad_w) / scale;
    float y1 = (cy - h * 0.5f - pad_h) / scale;
    float x2 = (cx + w * 0.5f - pad_w) / scale;
    float y2 = (cy + h * 0.5f - pad_h) / scale;

    // 边界裁剪
    x1 = fmaxf(0.0f, x1);
    y1 = fmaxf(0.0f, y1);
    x2 = fminf((float)img_w - 1.0f, x2);
    y2 = fminf((float)img_h - 1.0f, y2);

    // 原子计数
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

__global__ void nms_kernel(TensorRTYolo::BoxResult* d_candidates,
                           int* d_num_candidates,
                           TensorRTYolo::BoxResult* d_output, int* d_num_output,
                           float iou_thresh, int batch_size,
                           int max_candidates) {
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
        // printf("NMS Kernel: Processing pos %d/%d\n", pos, b);  // 调试输出
        // printf("NMS Kernel: Batch %d, Box %d: (%f,%f) - (%f,%f) at pos %d
        // with class_id %d and score %.4f\n",
        //        b, pos, a.x1, a.y1,a.x2,a.y2, pos, a.class_id, a.score);  //
        //        调试输出
    }
}

void launch_postprocess_kernel(
    const void* d_model_output,  // 模型输出 GPU 地址
    int num_anchors,             // 8400
    int num_classes,             // 80
    float conf_thresh,           // 0.25
    float iou_thresh,            // 0.45
    float scale, int pad_w, int pad_h, int img_w, int img_h,  // 原图宽高
    TensorRTYolo::BoxResult* d_candidates,  // 中间候选框 GPU 地址[batch,1024]
    int* d_num_candidates,  // 中间候选框数量 GPU 地址[batch,num]
    TensorRTYolo::BoxResult*
        d_final_boxes,  // 输出：最终框 GPU 地址[batch,boxs]
    int* d_num_boxes,   // 输出：框数量 GPU 地址[batch,num]
    int batch_size,     // 动态batch
    cudaStream_t stream, nvinfer1::DataType d_model_output_type) {
    const int MAX_CAND = 1024;  // 每张图最多候选框
    int total_cand = MAX_CAND * batch_size;

    // 分配batch式内存
    cudaMemsetAsync(d_num_candidates, 0, batch_size * sizeof(int), stream);
    cudaMemsetAsync(d_num_boxes, 0, batch_size * sizeof(int), stream);
    /// ========== 第一步：Decode ==========
    int block = 256;
    int grid_decode = (num_anchors * batch_size + block - 1) / block;

    switch (d_model_output_type) {
        case nvinfer1::DataType::kFLOAT:
            decode_kernel<float><<<grid_decode, block, 0, stream>>>(
                static_cast<const float*>(d_model_output), num_anchors,
                num_classes, conf_thresh, scale, pad_w, pad_h, img_w, img_h,
                d_candidates, d_num_candidates, batch_size);
            break;
        case nvinfer1::DataType::kHALF:
            decode_kernel<__half><<<grid_decode, block, 0, stream>>>(
                static_cast<const __half*>(d_model_output), num_anchors,
                num_classes, conf_thresh, scale, pad_w, pad_h, img_w, img_h,
                d_candidates, d_num_candidates, batch_size);
            break;
        case nvinfer1::DataType::kINT8:
            decode_kernel<int8_t><<<grid_decode, block, 0, stream>>>(
                static_cast<const int8_t*>(d_model_output), num_anchors,
                num_classes, conf_thresh, scale, pad_w, pad_h, img_w, img_h,
                d_candidates, d_num_candidates, batch_size);
            break;
        default:
            break;
    }

    // decode_kernel<<<grid_decode, block, 0, stream>>>(
    //     d_model_output, num_anchors, num_classes, conf_thresh, scale, pad_w,
    //     pad_h, img_w, img_h, d_candidates, d_num_candidates, batch_size);

    // int* num_candidates = new int[batch_size]{0};
    // cudaMemcpyAsync(num_candidates, d_num_candidates, batch_size *
    // sizeof(int),
    //            cudaMemcpyDeviceToHost, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // for (int i = 0; i < batch_size; i++) {
    //     std::cout << "候选框数量（GPU Decode后）：Batch " << i << ": "
    //          << num_candidates[i] << " candidates after decode." <<
    //          std::endl;
    // }
    // ========== 第二步：NMS ==========
    int grid_nms = (total_cand + block - 1) / block;
    nms_kernel<<<grid_nms, block, 0, stream>>>(
        d_candidates, d_num_candidates, d_final_boxes, d_num_boxes, iou_thresh,
        batch_size, MAX_CAND);
}