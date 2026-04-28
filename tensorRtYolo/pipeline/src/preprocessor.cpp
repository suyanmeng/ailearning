#include "preprocessor.h"

#include "preprocesskernel.h"
namespace TensorRTYolo {
void PreProcessor::batchProcess(const std::shared_ptr<const BatchData>& batch_data) {
    const int batch_size = batch_data->images.size();
    for (int i = 0; i < batch_size; i++) {
        int src_dist = i * batch_data->src_w * batch_data->src_h * 3;
        memcpy(batch_data->gpu_buf->cpu_img + src_dist,
               batch_data->images[i].mat->data,
               batch_data->src_w * batch_data->src_h * 3 * sizeof(uint8_t));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        batch_data->gpu_buf->gpu_img, batch_data->gpu_buf->cpu_img,
        batch_size * batch_data->src_w * batch_data->src_h * 3 * sizeof(uint8_t),
        cudaMemcpyHostToDevice, batch_data->gpu_buf->cuda_stream));

    launch_preprocess_kernel(
        batch_data->gpu_buf->gpu_img, batch_data->src_w, batch_data->src_h,
        batch_data->gpu_buf->gpu_input, batch_data->dst_w, batch_data->dst_h,
        batch_data->scale, batch_data->pad_w, batch_data->pad_h, true, batch_size,
        batch_data->gpu_buf->cuda_stream, batch_data->gpu_buf->d_data_type);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace TensorRTYolo