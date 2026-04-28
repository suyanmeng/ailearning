#include "postprocessor.h"

#include <thread>

#include "postprocesskernel.h"
namespace TensorRTYolo {
void TensorRTYolo::PostProcessor::batchProcess(const std::shared_ptr<const BatchData>& batch_data) {
    const int MAX_BOXES = 1024;
    launch_postprocess_kernel(
        batch_data->gpu_buf->gpu_output, batch_data->out_width, batch_data->out_height - 4, 0.25f,
        0.45f, batch_data->scale, batch_data->pad_w, batch_data->pad_h,
        batch_data->src_w, batch_data->src_h, batch_data->gpu_buf->d_boxes,
        batch_data->gpu_buf->d_box_num, batch_data->gpu_buf->d_last_boxes,
        batch_data->gpu_buf->d_last_box_num, batch_data->images.size(),
        batch_data->gpu_buf->cuda_stream, batch_data->gpu_buf->d_data_type);

    CUDA_CHECK(cudaMemcpyAsync(
        batch_data->gpu_buf->h_last_box_num, batch_data->gpu_buf->d_last_box_num,
        batch_data->images.size() * sizeof(int), cudaMemcpyDeviceToHost,
        batch_data->gpu_buf->cuda_stream));

    CUDA_CHECK(cudaMemcpyAsync(
        batch_data->gpu_buf->h_last_boxes, batch_data->gpu_buf->d_last_boxes,
        batch_data->images.size() * MAX_BOXES * sizeof(BoxResult),
        cudaMemcpyDeviceToHost, batch_data->gpu_buf->cuda_stream));
}
}  // namespace TensorRTYolo