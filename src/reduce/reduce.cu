#include <iostream>
#include <cuda_runtime.h>
using namespace std;
__global__ void reduce_naive(float* in, float* out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float s[256];

    // 1. 加载到共享内存
    s[threadIdx.x] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    // 2. 暴力规约 每次都有一半线程计算，一半偷懒，一个wrap内线程尽量保持同进退，减少分化(就是有的计算有的偷懒)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 3. 块0写结果
    if (threadIdx.x == 0)
        out[blockIdx.x] = s[0];
}

__global__ void reduce_interleaved(float* in, float* out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float s[256];

    s[threadIdx.x] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    // 交错规约（无分支分化）以blockDim为初始，减半往前加，最开始256个线程计算，然后128/64/32...最后1个
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = s[0];
}

__global__ void reduce_optimized(float* in, float* out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float s[256];

    // 合并访存加载
    float val = (idx < N) ? in[idx] : 0.0f;
    s[threadIdx.x] = val;
    __syncthreads();

    // 全块规约
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }

    // warp 内指令级快速规约（超快）
    if (threadIdx.x < 32) {
        warp_shfl_down_sync(0xffffffff, s[threadIdx.x], 16);//最后的一个wrap里面的线程数是32，把后面16个线程的寄存器值加到前面的16个线程的寄存器里
        warp_shfl_down_sync(0xffffffff, s[threadIdx.x], 8);
        warp_shfl_down_sync(0xffffffff, s[threadIdx.x], 4);
        warp_shfl_down_sync(0xffffffff, s[threadIdx.x], 2);
        warp_shfl_down_sync(0xffffffff, s[threadIdx.x], 1);
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = s[0];
}


#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 1.归一化
__global__ void normalize_kernel(
    float* img, float mean, float std, int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        img[idx] = (img[idx] - mean) / std;
    }
}

// 2.ReLU
__global__ void relu_kernel(float* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = fmaxf(x[idx], 0.0f);
}

// 3.Sigmoid
__global__ void sigmoid_kernel(float* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

int main(){
    // ========== 1. 模拟实际场景：8个图像像素(0~255) ==========
    const int N = 8;
    float h_data[N] = {10.f, 50.f, 128.f, 200.f, 255.f, 0.f, -20.f, 80.f};

    // GPU显存
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 线程配置：256线程/block 常规写法
    int block = 256;
    int grid  = (N + block - 1) / block;

    // ========== 2. 分步调用核函数 ==========
    // 模拟YOLO常用归一化：mean=128, std=128
    normalize_kernel<<<grid, block>>>(d_data, 128.f, 128.f, N);

    relu_kernel<<<grid, block>>>(d_data, N);

    sigmoid_kernel<<<grid, block>>>(d_data, N);

    // ========== 3. 拿回CPU打印结果 ==========
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("最终输出(归一化+ReLU+Sigmoid)：\n");
    for(int i=0;i<N;i++){
        printf("%.4f  ", h_data[i]);
    }
    printf("\n");

    cudaFree(d_data);
    return 0;
}