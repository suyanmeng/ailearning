# 1.关键函数
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMemcpy(
    void* dst,        // 目标地址
    const void* src,  // 源地址
    size_t count,     // 拷贝字节数
    cudaMemcpyKind    // 拷贝方向（最重要！） cudaMemcpyHostToHost       // CPU -> CPU（几乎不用）
                                            cudaMemcpyHostToDevice     // CPU -> GPU（最常用：上传数据）
                                            cudaMemcpyDeviceToHost     // GPU -> CPU（最常用：下载结果）
                                            cudaMemcpyDeviceToDevice   // GPU -> GPU（卡内拷贝）
);

# 2.注意错误：
float* d_A;
cudaMalloc(&d_A, size);
d_A[0] = 1.0f;  // ❌ 崩溃！CPU 不能碰 GPU 内存