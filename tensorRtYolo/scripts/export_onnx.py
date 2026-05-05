from ultralytics import YOLO
import torch
import urllib.request
import time
import os

MODEL_PATH = "../models/pt/yolov8n.pt"
ONNX_PATH = "../models/onnx/yolov8n.onnx"


# ==================== 基础信息打印 ====================
print("="*50)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 型号:", torch.cuda.get_device_name(0))
print("="*50)

# ==================== 自动下载测试图 ====================
# if not os.path.exists("bus.jpg"):
#     print("正在下载测试图片 bus.jpg...")
#     urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "bus.jpg")


# ==================== 自动下载模型 ====================
if not os.path.exists("yolov8n.pt"):
    print("正在下载 yolov8n.pt...")
    model = YOLO("yolov8n.pt")  # 会自动下载

# ==================== 自动导出ONNX ====================
if not os.path.exists("yolov8n_fp16.onnx"):
    print("🔧 正在导出 **FP16 精度 ONNX 模型** ...")
    
    # 加载模型并放到 GPU
    model = YOLO("yolov8n.pt").to('cuda:0') # 导出 FP16 ONNX 输入是fp16 这个是在gpu计算
    #model = YOLO("yolov8n.pt")  # 导出 FP32 ONNX 输入是fp32
    
    # 导出 FP16 ONNX
    model.export(
        format="onnx",
        device=0,
        imgsz=640,
        dynamic=True,
        half=True,#tensorrt自身使用fp16推理，onnx输入可以是fp16也可以是fp32
        simplify=True,
    )
    print("✅ onnx生成完成！")