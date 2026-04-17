import onnx

# 换成你的 onnx 路径
onnx_path = "/work/cuda/yolo8Test/resource/yolov8n.onnx"

# 加载模型
model = onnx.load(onnx_path)

# 打印核心信息
print("="*50)
print(" ONNX 模型基本信息")
print("="*50)
print(f"模型版本: {model.opset_import[0].version}")
print(f"生产者: {model.producer_name}")

# 输入信息
print("\n" + "="*50)
print(" 输入信息")
print("="*50)
for input in model.graph.input:
    print(f"输入名: {input.name}")
    shape = [dim.dim_value if dim.dim_value != 0 else dim.dim_param for dim in input.type.tensor_type.shape.dim]
    print(f"输入形状: {shape}")
    print(f"输入类型: {input.type.tensor_type.elem_type} -> {onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)}")

# 输出信息
print("\n" + "="*50)
print(" 输出信息")
print("="*50)
for output in model.graph.output:
    print(f"输出名: {output.name}")
    shape = [dim.dim_value if dim.dim_value != 0 else dim.dim_param for dim in output.type.tensor_type.shape.dim]
    print(f"输出形状: {shape}")
    print(f"输出类型: {output.type.tensor_type.elem_type} -> {onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)}")

# 模型检查
print("\n检查模型是否有效...")
onnx.checker.check_model(model)
print("✅ ONNX 模型有效！")