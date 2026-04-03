//优化 1：固定显存预分配
// ❌ 严重错误：循环里反复申请释放显存
// for (int i = 0; i < 1000; i++) {
//     float* d_tmp;
//     cudaMalloc(&d_tmp, 1024 * sizeof(float));  // 循环内malloc
//     kernel<<<>>>(d_tmp);
//     cudaFree(d_tmp);  // 循环内free
// }
//正确做法
#include <cuda_runtime.h>

// 1. 定义全局/类成员缓冲区（只分配一次）
float* d_input;    // GPU输入
float* d_output;   // GPU输出
float* d_buffer;   // 中间临时缓冲区
const int BUF_SIZE = 1024 * 1024; // 提前确定大小

// 2. 初始化函数：程序启动时调用一次
void initMemory() {
    // 一次性分配所有需要的显存
    cudaMalloc(&d_input, BUF_SIZE * sizeof(float));
    cudaMalloc(&d_output, BUF_SIZE * sizeof(float));
    cudaMalloc(&d_buffer, BUF_SIZE * sizeof(float));
    
    // 检查分配错误
    cudaCheckErrors("cudaMalloc failed");
}

// 3. 推理循环：只计算，不申请不释放
void inferenceLoop(float* h_input, float* h_output) {
    // 拷贝数据到GPU
    cudaMemcpy(d_input, h_input, BUF_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    // 运行核函数（复用显存）
    your_kernel<<<256, 256>>>(d_input, d_output, d_buffer, BUF_SIZE);
    
    // 结果拷回CPU
    cudaMemcpy(h_output, d_output, BUF_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
}

// 4. 程序退出时统一释放
void freeMemory() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_buffer);
}

//优化 2：杜绝 CPU↔GPU 频繁小拷贝
// 权重数组（很大）
float* d_weights;  
const int WEIGHT_SIZE = 1024 * 1024 * 4;

// --------------------------
// 初始化时：权重一次性拷入GPU
// --------------------------
void initWeights(float* h_weights) {
    cudaMalloc(&d_weights, WEIGHT_SIZE * sizeof(float));
    // 只拷贝1次！常驻GPU
    cudaMemcpy(d_weights, h_weights, WEIGHT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
}

// --------------------------
// 推理循环：绝不拷贝权重！
// --------------------------
void inference() {
    // 只拷贝输入（必须拷贝）
    cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);
    
    // 推理计算（复用GPU上的权重）
    kernel<<<>>>(d_input, d_weights, d_output);
    
    // 只拷回输出
    cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, stream);
}

//优化 3：显存对齐 + 连续访存
// 普通分配（不对齐）
cudaMalloc(&d_data, width * height * sizeof(float));

// --------------------------
// ✅ 对齐分配（推荐）
// --------------------------
size_t pitch;
cudaMallocPitch(
    &d_data,     // 输出指针
    &pitch,      // 输出实际行宽（对齐后）
    width * sizeof(float),  // 期望行宽
    height                // 行数
);
//优化 4：中间 Buffer 复用（显存占用减少 50%）
// 预分配一大块通用缓冲区
float* d_shared_buf;  
const int SHARED_BUF_SIZE = 1024 * 1024 * 8;

void initSharedBuffer() {
    cudaMalloc(&d_shared_buf, SHARED_BUF_SIZE * sizeof(float));
}

// 推理时：不同层复用同一块显存
void inference() {
    // 第一层用
    first_kernel<<<>>>(d_input, d_shared_buf);
    
    // 第二层直接覆盖复用
    second_kernel<<<>>>(d_shared_buf, d_output);
}
//优化 5：显存池（彻底解决显存泄漏 & 长期卡顿）

#include <unordered_map>
#include <mutex>

class SimpleCudaMemPool {
private:
    std::unordered_map<size_t, void*> pool;
    std::mutex mtx;

public:
    void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (pool.count(size)) {
            void* ptr = pool[size];
            pool.erase(size);
            return ptr;
        }
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    void free(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        pool[size] = ptr; // 放回池子，不真正释放
    }

    ~SimpleCudaMemPool() {
        for (auto& p : pool) cudaFree(p.second);
    }
};

// 全局使用
SimpleCudaMemPool g_mem_pool;

// 使用
float* d_buf = (float*)g_mem_pool.alloc(1024*sizeof(float));
g_mem_pool.free(d_buf, 1024*sizeof(float));