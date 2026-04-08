from ultralytics import YOLO
import torch
import urllib.request
import time
import os

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
    model = YOLO("yolov8n.pt")  # 会自动下载

# ==================== 自动生成 TensorRT 引擎 ====================
if not os.path.exists("yolov8n.engine"):
    print("🔧 首次运行：正在生成 TensorRT 引擎，请稍等（仅一次）...")
    model = YOLO("yolov8n.pt")
    model.export(
        format="engine",
        device=0,
        imgsz=640,
        dynamic=False,
        half=True,
    )
    print("✅ 引擎生成完成！")

# ==================== 加载引擎（秒加载） ====================
engine_model = YOLO("yolov8n.engine", task="detect")

# ==================== GPU 预热 ====================
print("🔍 预热模型...")
for _ in range(5):
    engine_model.predict("bus.jpg", device=0, verbose=False)

# ==================== 精准测速 ====================
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    results = engine_model.predict(
        "bus.jpg",
        device=0,
        imgsz=640,
        conf=0.25,
        save=False,
        verbose=False
    )

torch.cuda.synchronize()
infer_time = (time.time() - start) * 1000

# ==================== 输出结果 ====================
print(f"\n🚀 TensorRT 引擎推理耗时: **{infer_time:.2f} ms**")
print("="*50)