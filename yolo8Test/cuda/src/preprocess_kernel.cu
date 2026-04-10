#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include "preprocess.h"

// 双线性插值 + BGR->RGB + 归一化 + HWC->CHW 融合Kernel
__global__ void bgr_to_rgb_norm_resize_kernel(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    bool keep_aspect_ratio) {

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;  // 0:R,1:G,2:B

    if (w >= dst_w || h >= dst_h || c >= 3) return;

    // 计算缩放因子和填充偏移（letterbox）
    float scale_x, scale_y, pad_x = 0, pad_y = 0;
    if (keep_aspect_ratio) {
        float scale_w = (float)dst_w / src_w;
        float scale_h = (float)dst_h / src_h;
        float scale = fminf(scale_w, scale_h);
        int new_w = (int)(src_w * scale);
        int new_h = (int)(src_h * scale);
        pad_x = (dst_w - new_w) / 2.0f;
        pad_y = (dst_h - new_h) / 2.0f;
        scale_x = (float)src_w / new_w;
        scale_y = (float)src_h / new_h;
    } else {
        scale_x = (float)src_w / dst_w;
        scale_y = (float)src_h / dst_h;
    }

    float src_x, src_y;
    if (keep_aspect_ratio) {
        src_x = (w - pad_x) * scale_x;
        src_y = (h - pad_y) * scale_y;
        if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
            dst[c * dst_w * dst_h + h * dst_w + w] = 114.0f / 255.0f;
            return;
        }
    } else {
        src_x = (w + 0.5f) * scale_x - 0.5f;
        src_y = (h + 0.5f) * scale_y - 0.5f;
        src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
        src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));
    }

    int x0 = (int)floorf(src_x);
    int x1 = x0 + 1;
    int y0 = (int)floorf(src_y);
    int y1 = y0 + 1;
    float dx = src_x - x0;
    float dy = src_y - y0;
    float dx1 = 1.0f - dx;
    float dy1 = 1.0f - dy;

    x1 = (x1 < src_w) ? x1 : x0;
    y1 = (y1 < src_h) ? y1 : y0;

    // 读取四个像素的BGR值（每个像素3字节）
    int idx00 = (y0 * src_w + x0) * 3;
    int idx01 = (y0 * src_w + x1) * 3;
    int idx10 = (y1 * src_w + x0) * 3;
    int idx11 = (y1 * src_w + x1) * 3;

    uint8_t v00_b = src[idx00 + 0], v00_g = src[idx00 + 1], v00_r = src[idx00 + 2];
    uint8_t v01_b = src[idx01 + 0], v01_g = src[idx01 + 1], v01_r = src[idx01 + 2];
    uint8_t v10_b = src[idx10 + 0], v10_g = src[idx10 + 1], v10_r = src[idx10 + 2];
    uint8_t v11_b = src[idx11 + 0], v11_g = src[idx11 + 1], v11_r = src[idx11 + 2];

    float v_b = dy1 * dx1 * v00_b + dy1 * dx * v01_b + dy * dx1 * v10_b + dy * dx * v11_b;
    float v_g = dy1 * dx1 * v00_g + dy1 * dx * v01_g + dy * dx1 * v10_g + dy * dx * v11_g;
    float v_r = dy1 * dx1 * v00_r + dy1 * dx * v01_r + dy * dx1 * v10_r + dy * dx * v11_r;

    v_b = v_b / 255.0f;
    v_g = v_g / 255.0f;
    v_r = v_r / 255.0f;

    float out_val;
    if (c == 0) out_val = v_r;      // 输出通道0: R
    else if (c == 1) out_val = v_g; // 输出通道1: G
    else out_val = v_b;              // 输出通道2: B

    dst[c * dst_w * dst_h + h * dst_w + w] = out_val;
}

// 封装函数：启动 Kernel
void launch_preprocess_kernel(const uint8_t* src, int src_w, int src_h,
                              float* dst, int dst_w, int dst_h,
                              bool keep_aspect_ratio,
                              cudaStream_t stream) {
    dim3 threads(16, 16, 1);
    dim3 blocks((dst_w + 15) / 16, (dst_h + 15) / 16, 3);

    bgr_to_rgb_norm_resize_kernel<<<blocks, threads, 0, stream>>>(
        src, src_w, src_h, dst, dst_w, dst_h, keep_aspect_ratio);
}