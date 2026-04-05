#include <iostream>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>
using namespace std;

// 分块大小（16x16，CUDA最佳实践）
#define TILE_SIZE 16

__global__ void matrixMulShared(
    float *C,
    const float *A,
    const float *B,
    int M, int N, int K)
{
    // 共享显存：存储A、B的分块
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有分块
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // 加载A的分块到共享显存（合并访存）
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // 加载B的分块到共享显存（合并访存）
        if (col < N && t * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // 等待所有线程加载完成
        __syncthreads();

        // 分块内计算
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 等待计算完成，再加载下一个分块
        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ------------------------------------------------------------
// 矩阵初始化：填充随机浮点数（范围 0~1）
// ------------------------------------------------------------
void initMatrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

// 主函数与Naive版完全一致，仅替换核函数
int main()
{
    const int M = 1024, N = 1024, K = 1024;
    float *h_A = new float[M * K], *h_B = new float[K * N], *h_C = new float[M * N];
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    auto t1 = chrono::high_resolution_clock::now();
    matrixMulShared<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();
    auto t2 = chrono::high_resolution_clock::now();
    double gpu_time = chrono::duration<double>(t2-t1).count()*1000;
    cout << "gpu耗时: " << gpu_time << " ms" << endl;

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cout << "共享显存分块矩阵乘法执行完成" << endl;
    return 0;
}