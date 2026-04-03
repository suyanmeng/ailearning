#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;

__global__ void vecAddGPU(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 10) {
        printf("thread idx = %d | A=%f, B=%f\n", idx, A[idx], B[idx]);
    }
    if (idx < N)
        //printf("curid=%d",idx);
        C[idx] = A[idx] * 0.5f + B[idx] * 2.0f; // 稍微复杂一点，更贴近AI计算
}

void vecAddCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * 0.5f + B[i] * 2.0f;
    }
}

int main()
{
    int N = 1024 * 1024 * 100; // 1亿个元素
    int size = N * sizeof(float);

    cout << "数据量: " << N << " 个元素\n" << endl;

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N]{};

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ==================== CPU 测试 ====================
    auto t1 = chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_C, N);
    auto t2 = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double>(t2 - t1).count() * 1000;
    cout << "CPU 单核耗时: " << cpu_time << " ms" << endl;

    // ==================== GPU 准备 ====================
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;//向上取整公式

    // 🔥 关键：GPU 热身（消除第一次启动开销）
    vecAddGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // ==================== GPU 正式测试 ====================
    auto t3 = chrono::high_resolution_clock::now();
    // 多跑几次，让GPU满载
    for(int i=0;i<10;i++){
        vecAddGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    auto t4 = chrono::high_resolution_clock::now();
    double gpu_time = chrono::duration<double>(t4 - t3).count() * 1000 / 10.0;

    cout << "GPU 平均耗时: " << gpu_time << " ms" << endl;

    // ==================== 结果 ====================
    cout << "\n🚀 GPU 比 CPU 快: " << cpu_time / gpu_time << " 倍" << endl;

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}