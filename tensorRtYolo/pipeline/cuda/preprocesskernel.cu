#include "preprocesskernel.h"

template <typename T>
__global__ void bgr_to_rgb_norm_resize_kernel(const uint8_t* src, int src_w,
                                              int src_h, T* dst, int dst_w,
                                              int dst_h, float scale, int pad_w,
                                              int pad_h, bool keep_aspect_ratio,
                                              int batch) {
    int total_w = dst_w * batch;
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 总宽度（含batch）
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 高度
    int c = blockIdx.z;                             // // 0:R,1:G,2:B

    if (x >= total_w || y >= dst_h || c >= 3) return;

    // 计算当前属于第几张图
    int b = x / dst_w;
    int w = x % dst_w;
    int h = y;
    const uint8_t* src_img = src + b * src_w * src_h * 3;
    float scale_x, scale_y;
    if (keep_aspect_ratio) {
        int new_w = (int)(src_w * scale);
        int new_h = (int)(src_h * scale);
        scale_x = (float)src_w / new_w;
        scale_y = (float)src_h / new_h;
    } else {
        scale_x = (float)src_w / dst_w;
        scale_y = (float)src_h / dst_h;
    }

    float src_x, src_y;
    if (keep_aspect_ratio) {
        src_x = (w - pad_w) * scale_x;
        src_y = (h - pad_h) * scale_y;
        if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
            dst[b * 3 * dst_w * dst_h + c * dst_w * dst_h + h * dst_w + w] =
                114.0f / 255.0f;
            return;
        }
    } else {
        src_x = (w + 0.5f) * scale_x - 0.5f;
        src_y = (h + 0.5f) * scale_y - 0.5f;
        src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
        src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));
    }

    // ======================== 双线性插值 ========================
    int x0 = floorf(src_x);
    int x1 = min(x0 + 1, src_w - 1);
    int y0 = floorf(src_y);
    int y1 = min(y0 + 1, src_h - 1);
    float dx = src_x - x0;
    float dy = src_y - y0;

    int i00 = (y0 * src_w + x0) * 3;
    int i01 = (y0 * src_w + x1) * 3;
    int i10 = (y1 * src_w + x0) * 3;
    int i11 = (y1 * src_w + x1) * 3;

    float v_r =
        (src_img[i00 + 2] * (1 - dx) * (1 - dy) +
         src_img[i01 + 2] * dx * (1 - dy) + src_img[i10 + 2] * (1 - dx) * dy +
         src_img[i11 + 2] * dx * dy) /
        255.0f;
    float v_g =
        (src_img[i00 + 1] * (1 - dx) * (1 - dy) +
         src_img[i01 + 1] * dx * (1 - dy) + src_img[i10 + 1] * (1 - dx) * dy +
         src_img[i11 + 1] * dx * dy) /
        255.0f;
    float v_b =
        (src_img[i00 + 0] * (1 - dx) * (1 - dy) +
         src_img[i01 + 0] * dx * (1 - dy) + src_img[i10 + 0] * (1 - dx) * dy +
         src_img[i11 + 0] * dx * dy) /
        255.0f;

    float out_val;
    if (c == 0)
        out_val = v_r;  // 输出通道0: R
    else if (c == 1)
        out_val = v_g;  // 输出通道1: G
    else
        out_val = v_b;  // 输出通道2: B

    dst[b * 3 * dst_w * dst_h + c * dst_w * dst_h + h * dst_w + w] = out_val;
}

void launch_preprocess_kernel(const uint8_t* src, int src_w, int src_h,
                              void* dst, int dst_w, int dst_h, float scale,
                              int pad_w, int pad_h, bool keep_aspect_ratio,
                              int batch, cudaStream_t stream,
                              nvinfer1::DataType dst_data_type) {
    dim3 threads(16, 16, 1);
    dim3 blocks((dst_w * batch + 15) / 16, (dst_h + 15) / 16, 3);
    switch (dst_data_type) {
        case nvinfer1::DataType::kFLOAT:
            bgr_to_rgb_norm_resize_kernel<float>
                <<<blocks, threads, 0, stream>>>(
                    src, src_w, src_h, static_cast<float*>(dst), dst_w, dst_h,
                    scale, pad_w, pad_h, keep_aspect_ratio, batch);
            break;
        case nvinfer1::DataType::kHALF:
            bgr_to_rgb_norm_resize_kernel<__half>
                <<<blocks, threads, 0, stream>>>(
                    src, src_w, src_h, static_cast<__half*>(dst), dst_w, dst_h,
                    scale, pad_w, pad_h, keep_aspect_ratio, batch);
            break;
        case nvinfer1::DataType::kINT8:
            bgr_to_rgb_norm_resize_kernel<uint8_t>
                <<<blocks, threads, 0, stream>>>(
                    src, src_w, src_h, static_cast<uint8_t*>(dst), dst_w, dst_h,
                    scale, pad_w, pad_h, keep_aspect_ratio, batch);
            break;
        default:
            break;
    }
}