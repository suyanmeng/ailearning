#include "preprocessor.h"

#include "preprocesskernel.h"
namespace TensorRTYolo {
void PreProcessor::batchProcess(const BatchData& batch_data, float* gpu_input) {
    const int batch_size = batch_data.images.size();
    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaMemcpy(
            batch_data.gpu_buf->gpu_img_ + i * batch_data.src_w * batch_data.src_h * 3,
            batch_data.images[i].mat->data,
            batch_data.src_w * batch_data.src_h * 3 * sizeof(uint8_t),
            cudaMemcpyHostToDevice));
    }
    launch_preprocess_kernel(batch_data.gpu_buf->gpu_img_, batch_data.src_w, batch_data.src_h,
                             gpu_input, batch_data.dst_w, batch_data.dst_h,
                             batch_data.scale, batch_data.pad_w,
                             batch_data.pad_h, true, batch_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace TensorRTYolo