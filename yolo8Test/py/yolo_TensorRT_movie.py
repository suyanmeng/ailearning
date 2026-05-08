from ultralytics import YOLO
import torch
import urllib.request
import time
import os
import cv2

# ==================== 基础信息打印 ====================
print("="*50)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 型号:", torch.cuda.get_device_name(0))
print("="*50)

# ==================== 自动下载测试图 ====================
if not os.path.exists("bus.jpg"):
    print("正在下载测试图片 bus.jpg...")
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "bus.jpg")

# ==================== 自动下载模型 ====================
if not os.path.exists("yolov8n.pt"):
    print("正在下载 yolov8n.pt...")
    model = YOLO("yolov8n.pt")

# ==================== 自动生成 TensorRT 引擎 ====================
if not os.path.exists("yolov8n.engine"):
    print("🔧 首次运行：正在生成 TensorRT 引擎，请稍等（仅一次）...")
    model = YOLO("yolov8n.pt")
    model.export(
        format="engine",
        device=0,
        imgsz=640,
        dynamic=True,
        half=True,
    )
    print("✅ 引擎生成完成！")

# ==================== 加载 TensorRT 引擎 ====================
engine_model = YOLO("yolov8n.engine", task="detect")

# ==================== 模型预热 ====================
print("🔍 预热模型...")
for _ in range(5):
    engine_model.predict("bus.jpg", device=0, verbose=False)

# ==================== 视频推理配置 ====================
video_path = "demo0.mp4"  # 你的视频路径
output_path = "output_video.mp4"  # 输出带检测框的视频

# 打开视频
cap = cv2.VideoCapture(video_path)
fps_video = cap.get(cv2.CAP_PROP_FPS)  # 原视频帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

print(f"📹 视频信息：{width}x{height} | 原帧率：{fps_video:.1f} | 总帧数：{total_frames}")
print("🚀 开始视频推理...\n")

# ==================== 开始推理 + 计时 ====================
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # TensorRT 推理
    results = engine_model.predict(
        frame,
        device=0,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    # 绘制检测结果
    annotated_frame = results[0].plot()

    # 写入输出视频
    out.write(annotated_frame)

    frame_count += 1

    # 进度打印
    if frame_count % 50 == 0:
        print(f"已处理：{frame_count}/{total_frames} 帧")

# ==================== 结束统计 ====================
end_time = time.time()
total_cost_time = end_time - start_time
real_fps = frame_count / total_cost_time  # 真实推理帧率

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

# ==================== 最终结果 ====================
print("\n" + "="*60)
print(f"✅ 视频处理完成！")
print(f"📊 总耗时：{total_cost_time:.2f} 秒")
print(f"📊 处理总帧数：{frame_count}")
print(f"🚀 真实推理帧率：**{real_fps:.2f} FPS**")
print(f"📁 输出视频：{output_path}")
print("="*60)