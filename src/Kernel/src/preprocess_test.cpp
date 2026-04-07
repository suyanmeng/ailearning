#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "preprocess.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(int argc, char** argv) {
    // if (argc < 2) {
    //     printf("Usage: %s <image_path> [dst_w] [dst_h]\n", argv[0]);
    //     return -1;
    // }

    const char* img_path = "/work/cuda/src/test.jpg";
    int dst_w = (argc >= 3) ? atoi(argv[2]) : 640;
    int dst_h = (argc >= 4) ? atoi(argv[3]) : 640;

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        printf("Failed to load image: %s\n", img_path);
        return -1;
    }
    int src_w = img.cols, src_h = img.rows;
    printf("Source: %dx%d, Target: %dx%d\n", src_w, src_h, dst_w, dst_h);

    // 分配GPU内存
    uint8_t* d_src = nullptr;
    float* d_dst = nullptr;
    size_t src_bytes = src_w * src_h * 3 * sizeof(uint8_t);
    size_t dst_bytes = 3 * dst_w * dst_h * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_bytes));
    CUDA_CHECK(cudaMemcpy(d_src, img.data, src_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 预热
    launch_preprocess_kernel(d_src, src_w, src_h, d_dst, dst_w, dst_h, true, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    launch_preprocess_kernel(d_src, src_w, src_h, d_dst, dst_w, dst_h, true, stream);

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    printf("Kernel time: %.3f ms\n", kernel_ms);

    // 拷贝回CPU并保存为图像（验证用）
    float* h_dst = (float*)malloc(dst_bytes);
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));

    cv::Mat output_img(dst_h, dst_w, CV_32FC3);//32位浮点型3通道
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < dst_h; ++h) {
            for (int w = 0; w < dst_w; ++w) {
                float val = h_dst[c * dst_w * dst_h + h * dst_w + w];
                if (c == 0) output_img.at<cv::Vec3f>(h, w)[2] = val; // R
                else if (c == 1) output_img.at<cv::Vec3f>(h, w)[1] = val; // G
                else output_img.at<cv::Vec3f>(h, w)[0] = val; // B
            }
        }
    }
    cv::Mat output_vis;
    output_img.convertTo(output_vis, CV_8UC3, 255.0);//8位无符号型3通道，计算要用浮点，写图片要用整形
    cv::imwrite("output_preprocessed.jpg", output_vis);
    printf("Saved output_preprocessed.jpg\n");

    free(h_dst);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}