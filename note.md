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

# 3.概念
Kernel（核函数） → Grid（网格） → Block（线程块） → Warp（线程束） → Thread（线程）
SM（流式多处理器）
Block (Thread Block, 线程块) 通常 32/64/128/256/512/1024 线程
Warp (线程束 / 线程 warp) 1 Warp = 32 个连续线程
1 个 GPU 有多个 SM（流式多处理器）
1 个 SM 能塞很多 Block
1 个 Block 里最多 1024 个 Thread（硬限制）
Thread 最小调度单位：Warp（32 个 Thread 一组）
分支分化会让 Warp 变慢，所以并行逻辑要尽量统一

# 4.生成性能报告
nsys profile --trace=cuda --cuda-memory-usage=true --stats=true -o naive_matmul_detailed ./matmul_naive
使用Nsightsystem查看占用时间比例，优先减小拷贝时间

# 5.5大核心显存优化
优化 1：固定显存预分配 
优化 2：杜绝 CPU↔GPU 频繁小拷贝（头号性能杀手）
优化 3：显存对齐 + 连续访存
优化 4：中间 Buffer 复用
优化 5：显存池（彻底解决显存泄漏 & 长期卡顿）

# 6.四级显存架构
1.线程内临时变量
2.block内的共享变量，为block内线程共享，__shared__修饰
3.gpu内存，可以搭个内存池避免频繁申请释放
4.cpu专门给gpu用的内存，无分页连续，使用cudaHostAlloc申请，cudaFreeHost释放

# 7.cudaStream
多个流并行处理任务，异步
同一个流内的任务是顺序的