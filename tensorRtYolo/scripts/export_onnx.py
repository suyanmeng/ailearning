from ultralytics import YOLO
import torch
import urllib.request
import time
import os

# ==================== 统一路径配置 ====================
MODEL_PATH = "../models/pt/yolov8n.pt"    # .pt 模型路径
ONNX_PATH = "../models/onnx/yolov8n.onnx" # 导出的 ONNX 路径

# ==================== 自动创建目录 ====================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

# ==================== 基础信息打印 ====================
print("="*50)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 型号:", torch.cuda.get_device_name(0))
print("="*50)

# ==================== 自动下载模型 ====================
if not os.path.exists(MODEL_PATH):
    print(f"正在下载模型到: {MODEL_PATH}")
    model = YOLO("yolov8n.pt")  # 自动下载
    model.save(MODEL_PATH)      # 保存到指定路径
else:
    print(f"加载本地模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

# ==================== 导出 ONNX ====================
if not os.path.exists(ONNX_PATH):
    print("🔧 正在导出 **FP16 精度 ONNX 模型** ...")
    
    # 放到 GPU 上导出 FP16 ONNX
    model = YOLO(MODEL_PATH).to('cuda:0')  # 导出 FP16 ONNX 输入是fp16 这个是在gpu计算
    #model = YOLO("yolov8n.pt")  # 导出 FP32 ONNX 输入是fp32
    
    # 导出 FP16 ONNX
    model.export(
        format="onnx",
        device=0,
        imgsz=640,
        dynamic=True,
        half=True,    # tensorrt自身使用fp16推理，onnx输入可以是fp16也可以是fp32
        simplify=True,
    )
    exported_onnx_file = MODEL_PATH.replace('.pt', '.onnx')
    if os.path.exists(exported_onnx_file):
        os.rename(exported_onnx_file, ONNX_PATH)
        print(f"✅ ONNX 已生成并保存到: {ONNX_PATH}")
    
else:
    print(f"✅ ONNX 已存在: {ONNX_PATH}")