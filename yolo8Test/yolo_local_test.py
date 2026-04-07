from ultralytics import YOLO
import torch
import urllib.request
import time

# 自动下载测试图片
try:
    with open("bus.jpg", "rb") as f:
        pass
except:
    print("正在下载测试图片 bus.jpg...")
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "bus.jpg")

# =============== 设备检查 ===============
print("="*50)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 型号:", torch.cuda.get_device_name(0))
print("="*50)

# 加载本地模型
model = YOLO("yolov8n.pt")

# =============== 推理 + 计时 ===============
print("\n开始推理...")

# 预热模型（GPU 必须预热，时间才准）
for _ in range(5):
    model.predict("bus.jpg", device=0, verbose=False)

torch.cuda.synchronize()
start_time = time.time()

# 正式推理
for _ in range(100):
    results = model.predict(
        source="bus.jpg",
        device=0,
        imgsz=640,
        conf=0.25,
        save=False,      # 保存带框图片
        verbose=False   # 关闭冗余日志
    )

# results = model.predict(
#         source="girl.png",
#         device=0,
#         imgsz=640,
#         conf=0.25,
#         save=True,      # 保存带框图片
#         verbose=False   # 关闭冗余日志
#     )
torch.cuda.synchronize()
infer_time = (time.time() - start_time) * 1000  # 转毫秒

# =============== 输出结果 ===============
print(f"\n🚀 推理耗时: {infer_time:.2f} ms")
# print("="*50)
# print("检测到的目标：")

# for box in results[0].boxes:
#     cls_name = model.names[int(box.cls)]
#     conf = float(box.conf)
#     print(f"✅ {cls_name}  置信度: {conf:.2f}")