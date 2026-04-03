#include <cuda_runtime.h>
#include <stdio.h>
#include <unordered_map>
#include <mutex>

// ======================
// 四级显存架构 总代码
// ======================

// ==========================================
// L4 显存：主机端 Pinned 内存（最慢，CPU 侧）
// 作用：给 GPU 高速拷贝用，不使用普通 malloc
// ==========================================
class L4_HostMemory {
public:
    float* h_input;  // Pinned 输入
    float* h_output; // Pinned 输出
    int size;

    L4_HostMemory(int buf_size) : size(buf_size) {
        // 分配 pinned 主机内存（比 malloc 快 3~5 倍拷贝）
        cudaHostAlloc(&h_input, size * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h_output, size * sizeof(float), cudaHostAllocDefault);
    }

    ~L4_HostMemory() {
        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
    }
};

// ==========================================
// L3 显存：GPU 全局显存 + 显存池（核心层级）
// 作用：推理全程复用，绝不反复 malloc/free
// ==========================================
class L3_GlobalMemPool {
private:
    std::unordered_map<size_t, void*> pool;
    std::mutex mtx;

public:
    void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (pool.find(size) != pool.end()) {
            void* ptr = pool[size];
            pool.erase(size);
            return ptr;
        }
        void* ptr;
        cudaMalloc(&ptr, size); // 真正分配
        return ptr;
    }

    void free(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        pool[size] = ptr; // 放回池子，不释放
    }

    ~L3_GlobalMemPool() {
        for (auto& p : pool) cudaFree(p.second);
    }
};

// ==========================================
// L2 显存：Shared Memory（块内共享，二级快）
// 作用：卷积/矩阵乘法 必备，减少全局显存访问
// ==========================================
__global__ void l2_shared_memory_kernel(
    float* __restrict__ d_input,   // L3 全局显存
    float* __restrict__ d_output,
    int size
) {
    // 声明共享内存（每个 block 一份）
    __shared__ float s_mem[256];  // L2 显存

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    // 1. 从 L3 全局显存 → L2 共享内存
    s_mem[threadIdx.x] = d_input[idx];
    __syncthreads();

    // 2. 在 L2 里计算（极快）
    s_mem[threadIdx.x] *= 2.0f;

    // 3. 写回 L3
    d_output[idx] = s_mem[threadIdx.x];
}

// ==========================================
// L1 显存：寄存器（核函数内自动使用，最快）
// 作用：线程私有，0 延时，编译器自动管理
// ==========================================
__global__ void l1_register_kernel(
    float* __restrict__ d_input,
    float* __restrict__ d_output,
    int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    // 👇 这些变量全部存在 L1 寄存器里
    float reg_in = d_input[idx];   // L1 加载
    float reg_out = reg_in * 3.0f; // L1 计算
    d_output[idx] = reg_out;       // 写回 L3
}

// ======================
// 四级显存 统一推理入口
// ======================
class FourLevelMemoryEngine {
public:
    L4_HostMemory*    l4_host;
    L3_GlobalMemPool* l3_pool;
    float* d_input;   // L3
    float* d_output;  // L3
    int buf_size;
    cudaStream_t stream;

    FourLevelMemoryEngine(int size) : buf_size(size) {
        // 1. 初始化 L4：Pinned 主机内存 页锁定内存
        l4_host = new L4_HostMemory(size);

        // 2. 初始化 L3：显存池 + 预分配
        l3_pool = new L3_GlobalMemPool();
        d_input = (float*)l3_pool->alloc(buf_size * sizeof(float));
        d_output = (float*)l3_pool->alloc(buf_size * sizeof(float));

        // 3. 创建流
        cudaStreamCreate(&stream);
    }

    // 推理：L4 → L3 → L2 → L1 → L3 → L4
    void inference(float* cpu_data, float* cpu_result) {
        size_t bytes = buf_size * sizeof(float);

        // ========== L4 → L3（异步拷贝）==========
        memcpy(l4_host->h_input, cpu_data, bytes);
        cudaMemcpyAsync(d_input, l4_host->h_input, bytes, cudaMemcpyHostToDevice, stream);

        // ========== L3 → L2 → L1（计算）==========
        l2_shared_memory_kernel<<<(buf_size+255)/256, 256, 0, stream>>>(
            d_input, d_output, buf_size
        );

        // ========== L3 → L4（结果拷回）==========
        cudaMemcpyAsync(l4_host->h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        memcpy(cpu_result, l4_host->h_output, bytes);
    }

    ~FourLevelMemoryEngine() {
        l3_pool->free(d_input, buf_size * sizeof(float));
        l3_pool->free(d_output, buf_size * sizeof(float));
        delete l3_pool;
        delete l4_host;
        cudaStreamDestroy(stream);
    }
};

// ======================
// 测试主函数
// ======================
int main() {
    const int SIZE = 1024 * 1024;
    FourLevelMemoryEngine engine(SIZE);

    float* cpu_in = new float[SIZE];
    float* cpu_out = new float[SIZE];
    for (int i = 0; i < SIZE; i++) cpu_in[i] = 1.0f;

    // 运行推理（完整四级显存链路）
    engine.inference(cpu_in, cpu_out);

    printf("Done! output[0] = %.2f\n", cpu_out[0]);

    delete[] cpu_in;
    delete[] cpu_out;
    return 0;
}