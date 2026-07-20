import os
import copy
import torch

# Swift 4.0 物理目录结构导入
from swift.template import TEMPLATE_MAPPING, register_template, TemplateMeta
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
from swift.utils import seed_everything

from model_patcher import patch_qwen_spatial_residual

# ================= 配置区 =================
# 🌟 必须分别指定“原生底座”和“LoRA 目录”
BASE_MODEL_PATH = "/home/hk-project-pai00054/wm9835/qwen35/qwen_training/checkpoint-2000"  
LORA_PATH = "/home/hk-project-pai00054/wm9835/qwen35/qwen35_2b_expert_stage2_lora_LLM_only/v3-20260429-010852/checkpoint-3700"
IMAGE_DIR = "/home/hk-project-pai00054/wm9835/qwen35/test1"

SYSTEM_PROMPT = "You are a document analysis expert. You can read text, analyze layouts, and locate elements using precise bounding boxes [ymin, xmin, ymax, xmax]. Answer the user's question directly based on the provided image."
USER_PROMPT = "Analyze the page layout: provide bounding boxes coordinate, labels, OCR content, and alt text for figures or tables in this page."
# ==========================================

def apply_monkey_patch():
    print("🛠️ 正在执行内存补丁，强制关闭 qwen3_5 的思考模式...")
    if 'qwen3_5' in TEMPLATE_MAPPING:
        template_info = TEMPLATE_MAPPING['qwen3_5']
        if isinstance(template_info, dict):
            meta = template_info.get('template_meta', template_info)
        else:
            meta = template_info
        try:
            meta.is_thinking = False
            meta.thinking_prefix = ''
            meta.non_thinking_prefix = ''
            print(f"✅ 内存补丁已生效：已在 {type(meta).__name__} 中禁用思考模式。")
        except AttributeError as e:
            print(f"❌ 修改属性失败: {e}")
    else:
        print("⚠️ 警告：在 TEMPLATE_MAPPING 中未找到 qwen3_5。")

def run_batch_inference():
    seed_everything(42)

    # 1. 第一步：先打补丁！
    apply_monkey_patch()

    print(f"正在加载基础模型: {BASE_MODEL_PATH}")
    print(f"准备挂载 LoRA 权重: {LORA_PATH}")
    
    # 2. 初始化引擎 (双重加载！)
    try:
        engine = TransformersEngine(
            model=BASE_MODEL_PATH, 
            adapters=[LORA_PATH],   # Swift 4.0 常用写法
            dtype=torch.bfloat16, 
            template_type='qwen3_5'
        )
    except TypeError:
        # 如果上面的参数报错，自动降级为这种写法
        engine = TransformersEngine(
            model=BASE_MODEL_PATH, 
            adapter_id_or_path=LORA_PATH, 
            dtype=torch.bfloat16, 
            template_type='qwen3_5'
        )
    
    # 3. 空间增强补丁 (搭建架构)
    print("🔧 正在打入空间增强补丁 (初始化架构)...")
    patch_qwen_spatial_residual(engine.model, weight_path="/home/hk-project-pai00054/wm9835/qwen35/spatial_enhancer_final.pth")
    
    # ---------------------------------------------------------
    # 🌟 4. 核心修改：直接加载独立的 spatial_enhancer.pth
    # ---------------------------------------------------------
    enhancer_path = os.path.join( "/home/hk-project-pai00054/wm9835/qwen35/spatial_enhancer_final.pth")
    print(f"📥 正在寻找独立的增强器权重: {enhancer_path}")
    
    if os.path.exists(enhancer_path):
        try:
            # 1. 自动映射设备，并开启 weights_only=True 消除安全警告
            raw_state_dict = torch.load(enhancer_path, map_location=engine.model.device, weights_only=True)
            
            # 2. 清洗字典：削掉 "spatial_enhancer." 前缀
            cleaned_state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith("spatial_enhancer."):
                    new_key = k.replace("spatial_enhancer.", "", 1)
                    cleaned_state_dict[new_key] = v
                else:
                    cleaned_state_dict[k] = v
            
            # 3. 将清洗后的字典加载给子模块
            engine.model.spatial_enhancer.load_state_dict(cleaned_state_dict)
            
            current_gate = engine.model.spatial_enhancer.gate.data.item()
            print(f"✅ 完美恢复增强器权重！")
            print(f"🔥 当前测绘仪 Gate 值为: {current_gate:.4f}")
        except Exception as e:
            print(f"❌ 读取 {enhancer_path} 失败: {e}")
    # ---------------------------------------------------------

    # 5. 推理循环
    request_config = RequestConfig(max_tokens=8192, temperature=0.0)

    print(f"\n🚀 开始推理目录: {IMAGE_DIR}")
    for root, dirs, files in os.walk(IMAGE_DIR):
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files: continue
            
        for img_name in sorted(img_files):
            img_path = os.path.join(root, img_name)
            output_path = os.path.join(root, os.path.splitext(img_name)[0] + ".mmd")

            if os.path.exists(output_path): continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<img>{img_path}</img>{USER_PROMPT}"}
            ]
            infer_request = InferRequest(messages=messages)

            print(f" 👀 正在处理: {img_name}...")
            try:
                resp_list = engine.infer([infer_request], request_config)
                llm_response = resp_list[0].choices[0].message.content
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
                print(f" 📝 写入完成。")
                
            except Exception as e:
                print(f" ❌ 处理错误: {e}")

if __name__ == "__main__":
    run_batch_inference()