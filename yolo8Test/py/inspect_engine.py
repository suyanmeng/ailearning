import tensorrt as trt
import sys
import os

# 设置 TensorRT 日志记录器
class Logger(trt.ILogger):
    def log(self, severity, msg):
        print(f"[{severity}]: {msg}")

def inspect_engine(engine_path):
    if not os.path.exists(engine_path):
        print(f"错误: 文件 {engine_path} 不存在")
        return

    logger = Logger()
    # 初始化 TensorRT 插件（YOLOv8 可能依赖某些插件）
    trt.init_libnvinfer_plugins(logger, namespace="")

    print(f"--- 正在加载引擎: {engine_path} ---")
    
    try:
        with open(engine_path, "rb") as f:
            # 反序列化引擎
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
            
        if engine is None:
            print("无法反序列化引擎，请检查文件是否损坏或版本不匹配。")
            return

        # 1. 打印基本信息
        print(f"\n✅ 引擎加载成功!")
        print(f"📦 名称: {engine.name}")
        print(f"🔢 绑定数量 (输入+输出): {engine.num_bindings}")
        
        # 2. 打印输入输出信息 (Bindings)
        print(f"\n--- 输入/输出张量信息 ---")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            
            type_str = "INPUT" if is_input else "OUTPUT"
            # 获取数据类型名称
            dtype_str = trt.nptype(dtype).__name__
            
            print(f"[{type_str}] 索引 {i}: 名称='{name}', 形状={shape}, 数据类型={dtype_str}")

        # 3. 打印网络层级信息 (Layers)
        # 注意：序列化后的引擎可能不包含完整的层名称，取决于导出时的配置
        print(f"\n--- 网络层级结构 (前5层 & 后5层) ---")
        num_layers = engine.num_layers
        
        if num_layers > 0:
            print(f"总层数: {num_layers}")
            
            # 为了防止打印过多，只打印头部和尾部
            indices_to_print = []
            if num_layers <= 10:
                indices_to_print = list(range(num_layers))
            else:
                indices_to_print = list(range(5)) + [-1] + list(range(num_layers-5, num_layers))

            for i in indices_to_print:
                if i == -1:
                    print("... (省略中间层) ...")
                    continue
                
                layer = engine.get_layer(i)
                layer_name = layer.name if layer.name else "Unnamed"
                layer_type = layer.type
                print(f"层 {i}: 类型={layer_type}, 名称={layer_name}")
        else:
            print("引擎未包含层信息（可能是纯函数式或已高度优化）。")

        # 4. 打印优化配置 (如果有)
        print(f"\n--- 优化配置 ---")
        num_profiles = engine.num_optimization_profiles
        print(f"优化配置文件数量: {num_profiles}")
        
        if num_profiles > 0:
            # 获取第一个配置文件的形状信息
            profile_shape = engine.get_profile_shape(0, 0) # profile_index=0, binding_index=0
            # 这里的逻辑比较通用，YOLOv8 动态形状通常会有 Min/Opt/Max
            print(f"输入张量形状范围 (Min/Opt/Max):")
            # 简单展示第一个绑定的形状范围
            try:
                min_shape = engine.get_profile_shape(0, 0)
                opt_shape = engine.get_profile_shape(1, 0) # 这里的索引逻辑视具体导出而定，仅作演示
                # 更稳健的打印方式是遍历
                print(f" (具体形状需根据 binding index 查询)")
            except:
                pass
                
    except Exception as e:
        print(f"发生错误: {e}")
        print("提示: 请确保 TensorRT 版本与生成该 engine 的版本兼容。")

if __name__ == "__main__":
    # 替换为你的 engine 文件路径
    engine_file = "/work/cuda/yolo8Test/resource/yolov8n_cpp.engine" 
    
    if len(sys.argv) > 1:
        engine_file = sys.argv[1]
        
    inspect_engine(engine_file)