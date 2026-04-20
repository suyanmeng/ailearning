#include "postprocessor.h"

#include "postprocesskernel.h"
namespace TensorRTYolo {
BatchResult TensorRTYolo::PostProcessor::batchProcess(
    const BatchData& batch_data, float* gpu_output) {
    const int MAX_BOXES = 1024;
    BoxResult* d_boxes;
    int* d_box_num;
    CUDA_CHECK(cudaMalloc(
        &d_boxes, batch_data.images.size() * MAX_BOXES * sizeof(BoxResult)));
    CUDA_CHECK(cudaMalloc(&d_box_num, batch_data.images.size() * sizeof(int)));
    launch_postprocess_kernel(
        gpu_output, out_width_, out_height_ - 4, 0.25f, 0.45f, batch_data.scale,
        batch_data.pad_w, batch_data.pad_h, batch_data.src_w,
        batch_data.src_h, d_boxes, d_box_num, batch_data.images.size());
    cudaDeviceSynchronize();
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("CUDA 错误: %d %s\n", err1, cudaGetErrorString(err1));
    }

    int* box_num = new int[batch_data.images.size()];
    cudaMemcpy(box_num, d_box_num, batch_data.images.size() * sizeof(int),
               cudaMemcpyDeviceToHost);
    BoxResult* detections =
        new BoxResult[batch_data.images.size() * MAX_BOXES * sizeof(BoxResult)];
    cudaMemcpy(detections, d_boxes,
               batch_data.images.size() * MAX_BOXES * sizeof(BoxResult),
               cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_box_num));

    BatchResult ret;
    for (int i = 0; i < batch_data.images.size(); i++) {
        ImageResult img_boxes;
        for (int j = 0; j < box_num[i]; j++) {
            img_boxes.boxes.push_back(detections[i * MAX_BOXES + j]);
        }
        ret.imgs_ret.push_back(img_boxes);
        std::cout << "第 " << i << " 张图片检测到 " << box_num[i] << " 个框"
                  << std::endl;
    }

    return ret;
}
}  // namespace TensorRTYolo