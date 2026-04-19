#pragma once
#include "common.h"

// 启动预处理 Kernel 的封装函数
// 参数说明：
//   src        : 输入图像 (uint8_t, HWC布局, BGR顺序)
//   src_w, src_h: 输入图像宽高
//   dst        : 输出张量 (float, CHW布局, RGB顺序, 归一化0~1)
//   dst_w, dst_h: 目标尺寸
//   keep_aspect_ratio : 是否保持宽高比（letterbox）
void launch_preprocess_kernel(const uint8_t* src, int src_w, int src_h,
                              float* dst, int dst_w, int dst_h,float scale, int pad_w, int pad_h,
                              bool keep_aspect_ratio, int batch);
