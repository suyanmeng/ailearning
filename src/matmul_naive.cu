#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// 定义矩阵索引宏：行优先存储，ld 是 leading dimension（矩阵宽度）
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 辅助函数：检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ------------------------------------------------------------
// 朴素矩阵乘法 Kernel
// C = A * B, 矩阵维度: A: M x K, B: K x N, C: M x N
// 每个线程计算 C 的一个元素
// ------------------------------------------------------------
__global__ void naive_matmul_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    // 获取当前线程在 Grid 中的全局坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保不超出矩阵范围
    if (row < M && col < N) {
        float sum = 0.0f;
        // 内积计算：遍历 K 维度
        for (int k = 0; k < K; ++k) {
            sum += A[OFFSET(row, k, K)] * B[OFFSET(k, col, N)];
        }
        C[OFFSET(row, col, N)] = sum;
    }
}

// ------------------------------------------------------------
// CPU 参考实现：用于验证正确性和性能对比
// ------------------------------------------------------------
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[OFFSET(i, k, K)] * B[OFFSET(k, j, N)];
            }
            C[OFFSET(i, j, N)] = sum;
        }
    }
}

// ------------------------------------------------------------
// 矩阵初始化：填充随机浮点数（范围 0~1）
// ------------------------------------------------------------
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

// ------------------------------------------------------------
// 比较两个矩阵是否在误差范围内相等
// ------------------------------------------------------------
bool compare_matrices(const float* C1, const float* C2, int M, int N, float eps = 1e-2f) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C1[i] - C2[i]) > eps) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------
// 计算 GFLOPS（每秒十亿次浮点运算）
// 矩阵乘法总浮点操作数 = 2 * M * N * K
// ------------------------------------------------------------
double compute_gflops(double elapsed_ms, int M, int N, int K) {
    double total_ops = 2.0 * M * N * K;
    double elapsed_sec = elapsed_ms / 1000.0;
    return total_ops / elapsed_sec / 1e9;
}

// ------------------------------------------------------------
// 主函数：演示正确性验证与性能对比
// ------------------------------------------------------------
int main(int argc, char** argv) {
    // 默认矩阵维度
    int M = 1024, N = 1024, K = 1024;
    
    // 支持命令行参数：./matmul_naive M N K
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        printf("Custom matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    } else {
        printf("Using default matrix size: M=%d, N=%d, K=%d\n", M, N, K);
        printf("Usage: ./matmul_naive [M N K]\n\n");
    }
    
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    // 1. 分配主机内存
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_cpu = (float*)malloc(bytes_C);
    float *h_C_gpu = (float*)malloc(bytes_C);
    
    // 2. 初始化矩阵（随机数据）
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    memset(h_C_cpu, 0, bytes_C);
    memset(h_C_gpu, 0, bytes_C);
    
    // 3. CPU 计算（作为参考基准）
    printf("\n=== CPU Computation ===\n");
    clock_t cpu_start = clock();
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) * 1000.0 / CLOCKS_PER_SEC;
    double cpu_gflops = compute_gflops(cpu_ms, M, N, K);
    printf("CPU time: %.3f ms, GFLOPS: %.2f\n", cpu_ms, cpu_gflops);
    
    // 4. 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    // 5. 将数据从主机拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    // 6. 配置 Kernel 启动参数
    // 每个 block 使用 16x16 = 256 个线程，这是一种常见的平衡配置
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    
    printf("\n=== GPU Computation (Naive) ===\n");
    printf("Kernel configuration: Grid(%d, %d), Block(16, 16)\n",
           blocks_per_grid.x, blocks_per_grid.y);
    
    // 创建 CUDA 事件用于精确计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 预热：执行一次 Kernel，避免冷启动影响
    naive_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));
    naive_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    double gpu_gflops = compute_gflops(gpu_ms, M, N, K);
    
    printf("GPU Kernel time: %.3f ms, GFLOPS: %.2f\n", gpu_ms, gpu_gflops);
    printf("Speedup (CPU/GPU): %.2fx\n", cpu_ms / gpu_ms);
    
    // 7. 将结果拷贝回主机
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    // 8. 正确性验证
    printf("\n=== Verification ===\n");
    if (compare_matrices(h_C_cpu, h_C_gpu, M, N)) {
        printf("Result: PASSED (matrices match within tolerance)\n");
    } else {
        printf("Result: FAILED (matrices do not match)\n");
    }
    
    // 9. 清理资源
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}