#include "postprocessor.h"

#include "postprocesskernel.h"
namespace TensorRTYolo {
BatchResult TensorRTYolo::PostProcessor::batchProcess(
    const BatchData& batch_data, float* gpu_output) {
    const int MAX_BOXES = 1024;
    launch_postprocess_kernel(
        gpu_output, out_width_, out_height_ - 4, 0.25f, 0.45f, batch_data.scale,
        batch_data.pad_w, batch_data.pad_h, batch_data.src_w, batch_data.src_h,
        batch_data.gpu_buf->d_boxes, batch_data.gpu_buf->d_box_num,
        batch_data.gpu_buf->d_last_boxes, batch_data.gpu_buf->d_last_box_num,
        batch_data.images.size());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA 错误: %d %s\n", err, cudaGetErrorString(err));
    }

    int* box_num = new int[batch_data.images.size()];
    cudaMemcpy(box_num, batch_data.gpu_buf->d_last_box_num,
               batch_data.images.size() * sizeof(int), cudaMemcpyDeviceToHost);
    BoxResult* detections = new BoxResult[batch_data.images.size() * MAX_BOXES];
    cudaMemcpy(detections, batch_data.gpu_buf->d_last_boxes,
               batch_data.images.size() * MAX_BOXES * sizeof(BoxResult),
               cudaMemcpyDeviceToHost);

    BatchResult ret;
    ret.imgs_ret.reserve(batch_data.images.size());
    for (int i = 0; i < batch_data.images.size(); i++) {
        ImageResult img_boxes;
        img_boxes.boxes.reserve(box_num[i]);
        for (int j = 0; j < box_num[i]; j++) {
            img_boxes.boxes.push_back(detections[i * MAX_BOXES + j]);
        }
        ret.imgs_ret.push_back(std::move(img_boxes));
        // std::cout << "第 " << i << " 张图片检测到 " << box_num[i] << " 个框"
        //           << std::endl;
    }
    delete[] box_num;
    delete[] detections;
    return ret;
}
}  // namespace TensorRTYolo