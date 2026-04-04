#include <cuda_runtime.h>
#include <iostream>
#include <string>

// 简单核函数：模拟推理计算
__global__ void inference_kernel(float *input, float *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        // 模拟卷积/计算
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

int main() {
    // 配置
    const int BATCH_SIZE = 4;      // 4批数据
    const int DATA_SIZE = 1024 * 256;
    const int BYTES = DATA_SIZE * sizeof(float);

    // 1. 创建 4 个 CUDA 流（并行处理4批次）
    cudaStream_t streams[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 2. 分配 pinned 主机内存（高速异步拷贝必备）
    float *h_input, *h_output;
    cudaHostAlloc(&h_input, BATCH_SIZE * BYTES, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, BATCH_SIZE * BYTES, cudaHostAllocDefault);

    // 初始化输入
    for (int i = 0; i < BATCH_SIZE * DATA_SIZE; i++) {
        h_input[i] = 1.0f;
    }

    // 3. 分配 GPU 全局显存
    float *d_input, *d_output;
    cudaMalloc(&d_input, BATCH_SIZE * BYTES);
    cudaMalloc(&d_output, BATCH_SIZE * BYTES);

    // =========================
    // 核心：多流并行推理
    // =========================
    for (int i = 0; i < BATCH_SIZE; i++) {
        // 每个流独立处理一个批次
        int offset = i * DATA_SIZE;

        // 步骤1：异步拷贝 H2D（不阻塞）
        cudaMemcpyAsync(
            d_input + offset,
            h_input + offset,
            BYTES,
            cudaMemcpyHostToDevice,
            streams[i]
        );

        // 步骤2：流内异步执行核函数（拷贝完自动计算）
        inference_kernel<<<(DATA_SIZE + 255) / 256, 256, 0, streams[i]>>>(
            d_input + offset,
            d_output + offset,
            DATA_SIZE
        );

        // 步骤3：异步拷贝 D2H
        cudaMemcpyAsync(
            h_output + offset,
            d_output + offset,
            BYTES,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // 等待所有流完成
    for (int i = 0; i < BATCH_SIZE; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 验证结果
    printf("output[0] = %.2f\n", h_output[0]);

    // 释放
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    for (int i = 0; i < BATCH_SIZE; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}