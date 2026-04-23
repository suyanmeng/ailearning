#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 计算密集型 Kernel：每个线程对数据进行多次迭代运算
__global__ void compute_kernel(const float* d_data, float* d_out, int N, int iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float val = d_data[idx];
        // 内部循环，增加计算量
        for (int i = 0; i < iter; ++i) {
            val = val * val + 0.5f;
        }
        d_out[idx] = val;
    }
}

// 单流顺序执行
void run_single_stream(float** h_inputs, float** h_outputs, int num_tasks, int N, int iter) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int t = 0; t < num_tasks; ++t) {
        // 异步拷贝 H->D
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_inputs[t], N * sizeof(float), cudaMemcpyHostToDevice, stream));
        // Kernel 启动（每个任务只启动一次，但内部计算量大）
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        compute_kernel<<<grid, block, 0, stream>>>(d_input, d_output, N, iter);
        // 异步拷贝 D->H
        CUDA_CHECK(cudaMemcpyAsync(h_outputs[t], d_output, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Single stream total time: %.3f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// 多流并发执行（每个任务一个独立流）
void run_multi_stream(float** h_inputs, float** h_outputs, int num_tasks, int N, int iter) {
    cudaStream_t* streams = (cudaStream_t*)malloc(num_tasks * sizeof(cudaStream_t));
    float** d_inputs = (float**)malloc(num_tasks * sizeof(float*));
    float** d_outputs = (float**)malloc(num_tasks * sizeof(float*));

    for (int t = 0; t < num_tasks; ++t) {
        CUDA_CHECK(cudaMalloc(&d_inputs[t], N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs[t], N * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[t]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));   // 使用默认流记录开始

    // 同时启动所有流的操作
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    for (int t = 0; t < num_tasks; ++t) {
        if (t > 0) {
            cudaEvent_t delay_event;
            cudaEventCreate(&delay_event);
            cudaEventRecord(delay_event, streams[t-1]);  // 在前一个流中记录事件
            cudaStreamWaitEvent(streams[t], delay_event, 0);
            cudaEventDestroy(delay_event);
        }
        CUDA_CHECK(cudaMemcpyAsync(d_inputs[t], h_inputs[t], N * sizeof(float), cudaMemcpyHostToDevice, streams[t]));
        compute_kernel<<<grid, block, 0, streams[t]>>>(d_inputs[t], d_outputs[t], N, iter);
        CUDA_CHECK(cudaMemcpyAsync(h_outputs[t], d_outputs[t], N * sizeof(float), cudaMemcpyDeviceToHost, streams[t]));
    }

    // 等待所有流完成
    for (int t = 0; t < num_tasks; ++t) {
        CUDA_CHECK(cudaStreamSynchronize(streams[t]));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Multi stream total time: %.3f ms\n", ms);

    // 清理
    for (int t = 0; t < num_tasks; ++t) {
        CUDA_CHECK(cudaStreamDestroy(streams[t]));
        CUDA_CHECK(cudaFree(d_inputs[t]));
        CUDA_CHECK(cudaFree(d_outputs[t]));
    }
    free(streams);
    free(d_inputs);
    free(d_outputs);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    // 可调参数
    int num_tasks = 4;                 // 任务数量（模拟4张图）
    int N = 10 << 20;                 // 每个任务数据量：10M 浮点数 ≈ 40 MB
    int iter = 5000;                  // Kernel 内部循环次数，增加计算量

    if (argc >= 2) N = atoi(argv[1]) << 20;
    if (argc >= 3) iter = atoi(argv[2]);

    size_t bytes = N * sizeof(float);
    printf("Running with %d tasks, each size = %.2f MB, iter = %d\n", num_tasks, (float)bytes / (1024*1024), iter);

    // 分配固定内存（Pinned Memory）用于异步传输
    float** h_inputs = (float**)malloc(num_tasks * sizeof(float*));
    float** h_outputs = (float**)malloc(num_tasks * sizeof(float*));
    for (int t = 0; t < num_tasks; ++t) {
        CUDA_CHECK(cudaHostAlloc(&h_inputs[t], bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_outputs[t], bytes, cudaHostAllocDefault));
        // 初始化输入数据
        for (int i = 0; i < N; ++i) {
            h_inputs[t][i] = (float)(i % 100) / 100.0f;
        }
    }

    // 运行测试
    run_single_stream(h_inputs, h_outputs, num_tasks, N, iter);
    run_multi_stream(h_inputs, h_outputs, num_tasks, N, iter);

    // 释放固定内存
    for (int t = 0; t < num_tasks; ++t) {
        CUDA_CHECK(cudaFreeHost(h_inputs[t]));
        CUDA_CHECK(cudaFreeHost(h_outputs[t]));
    }
    free(h_inputs);
    free(h_outputs);

    return 0;
}