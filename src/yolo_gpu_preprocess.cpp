#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
using namespace std;

int main() {
    // 1. 读图片
    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        cout << "请放一张 test.jpg 到当前文件夹！" << endl;
        return -1;
    }

    // 2. GPU 内存
    cv::cuda::GpuMat d_img, d_rgb, d_resized, d_norm;
    
    // 3. 上传到 GPU
    d_img.upload(img);
    
    // 4. BGR -> RGB
    cv::cuda::cvtColor(d_img, d_rgb, cv::COLOR_BGR2RGB);
    
    // 5. 缩放到 640x640
    cv::cuda::resize(d_rgb, d_resized, cv::Size(640, 640));
    
    // 6. 归一化 0~1
    d_resized.convertTo(d_norm, CV_32F, 1.0 / 255.0);

    // 7. 下载回 CPU 看结果
    cv::Mat result(d_norm);
    cout << "✅ GPU 预处理完成！" << endl;
    cout << "输出尺寸：" << result.size() << endl;

    return 0;
}