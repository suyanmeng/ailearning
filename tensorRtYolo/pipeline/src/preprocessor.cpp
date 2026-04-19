#include "preprocessor.h"

#include "preprocesskernel.h"
namespace TensorRTYolo {
void PreProcessor::batchProcess(const BatchData& batch_data, float* gpu_input) {
    uint8_t* d_src = nullptr;
    const int batch_size = batch_data.images.size();
    const size_t src_bytes = batch_size * batch_data.origin_w *
                             batch_data.origin_h * 3 * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaMemcpy(
            d_src + i * batch_data.origin_w * batch_data.origin_h * 3,
            batch_data.images[i].mat->data,
            batch_data.origin_w * batch_data.origin_h * 3 * sizeof(uint8_t),
            cudaMemcpyHostToDevice));
    }
    launch_preprocess_kernel(d_src, batch_data.origin_w, batch_data.origin_h,
                             gpu_input, batch_data.dst_w, batch_data.dst_h,
                             batch_data.scale, batch_data.pad_w,
                             batch_data.pad_h, true, batch_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace TensorRTYolo