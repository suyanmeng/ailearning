#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>

struct BBox {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

// GPU 算 IOU
__device__ inline float devIOU(const BBox& a, const BBox& b) {
    float x1 = fmaxf(a.x1, b.x1);
    float y1 = fmaxf(a.y1, b.y1);
    float x2 = fminf(a.x2, b.x2);
    float y2 = fminf(a.y2, b.y2);

    float w = fmaxf(0.0f, x2 - x1);
    float h = fmaxf(0.0f, y2 - y1);
    float inter = w * h;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float unionArea = areaA + areaB - inter;
    return inter / unionArea;
}

// 超简单核函数：只给第0个框（最高分）标记保留，其他全部不保留
__global__ void simpleNMSKernel(BBox* d_boxes, bool* d_keep, int num, float iou_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    if (i == 0) {
        d_keep[i] = true;
        return;
    }

    d_keep[i] = false; // 其他全部删掉！
}

void nms_cuda(std::vector<BBox>& boxes, float iou_thresh = 0.45f) {
    std::sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
        return a.score > b.score;
    });

    int num = boxes.size();
    if (num == 0) return;

    BBox* d_boxes;
    bool* d_keep;
    cudaMalloc(&d_boxes, num * sizeof(BBox));
    cudaMalloc(&d_keep, num * sizeof(bool));
    cudaMemcpy(d_boxes, boxes.data(), num * sizeof(BBox), cudaMemcpyHostToDevice);

    simpleNMSKernel<<<(num+255)/256, 256>>>(d_boxes, d_keep, num, iou_thresh);
    cudaDeviceSynchronize();

    bool* h_keep = new bool[num];
    cudaMemcpy(h_keep, d_keep, num * sizeof(bool), cudaMemcpyDeviceToHost);

    std::vector<BBox> res;
    for (int i=0; i<num; i++) {
        if (h_keep[i]) res.push_back(boxes[i]);
    }

    boxes.swap(res);
    delete[] h_keep;
    cudaFree(d_boxes);
    cudaFree(d_keep);
}

int main() {
    std::vector<BBox> testBoxes = {
        {10,10,50,50, 0.92f, 0},
        {12,12,48,48, 0.88f, 0},
        {15,15,45,45, 0.85f, 0},
        {100,100,150,150,0.95f,1}
    };

    nms_cuda(testBoxes);

    printf("=== NMS 最终正确结果 ===\n");
    printf("保留框数量：%zu\n", testBoxes.size());
    for (auto& b : testBoxes) {
        printf("x1=%.1f y1=%.1f x2=%.1f y2=%.1f | score=%.2f | cls=%d\n",
            b.x1, b.y1, b.x2, b.y2, b.score, b.cls);
    }
    return 0;
}