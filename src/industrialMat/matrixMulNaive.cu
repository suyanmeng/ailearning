#include <iostream>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;

// 核函数：Naive矩阵乘法 C=A*B
// 每个线程计算C[row][col]
__global__ void matrixMulNaive(
    float* C, 
    const float* A, 
    const float* B, 
    int M, int N, int K
) {
    // 计算当前线程负责的C矩阵坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界保护
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    // 暴力累加 K 次
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// 初始化矩阵
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

int main() {
    // 矩阵尺寸（方形矩阵，方便测试）
    const int M = 1024, N = 1024, K = 1024;

    // 主机内存分配
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    // 数据拷贝主机->设备
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数：16x16线程块（CUDA通用最优块大小）
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    auto t1 = chrono::high_resolution_clock::now();
    matrixMulNaive<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();
    auto t2 = chrono::high_resolution_clock::now();
    double gpu_time = chrono::duration<double>(t2-t1).count()*1000;
    cout << "gpu耗时: " << gpu_time << " ms" << endl;
    // 结果拷贝回主机
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // 释放内存
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    cout << "Naive矩阵乘法执行完成" << endl;
    return 0;
}