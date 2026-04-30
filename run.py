import os
import glob
import copy
import torch

# Swift 4.0 物理目录结构导入
from swift.template import TEMPLATE_MAPPING, register_template, TemplateMeta
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
from swift.utils import seed_everything
from safetensors.torch import load_file

# ================= 配置区 =================
MODEL_PATH = "/home/hk-project-pai00054/wm9835/qwen35/Qwen3.5-2B"
IMAGE_DIR = "/home/hk-project-pai00054/wm9835/qwen35/qwen_benchmark"

SYSTEM_PROMPT = "You are a document expert. For each page, provide the correct bounding boxes coordinate [ymin, xmin, ymax, xmax] and corresponding labels, ocr content and alt text for figure- or table-element in this page."
USER_PROMPT = """
You are analyzing one document page image.
Original image size: width={W}, height={H}
Task: Extract visible layout blocks and OCR text. for figure and table elements give the alt text if available.
Return ONLY a valid JSON array. No markdown. No explanation.

Each item must have this schema:
{{
  "label": "header|title|author_information|paragraph|caption|section_header|footer|list|equation|table|figure|footnote",
  "bbox": [xmin, ymin, xmax, ymax],
  "text": "OCR text of the block",
  "reading_order": 1,
  "alt_text": "description for figure/table, or null if not applicable"
}}

Rules:
- bbox must use pixel coordinates in the ORIGINAL image coordinate system.
- IMPORTANT: Escape all backslashes in text (e.g., use \\\\pi instead of \\pi).
- Do not invent text. Keep bbox tight.
"""
# ==========================================

def apply_monkey_patch():
    """
    🌟 修正后的 Monkey Patch：适配 Swift 4.0 的对象结构
    """
    print("🛠️ 正在执行内存补丁，强制关闭 qwen3_5 的思考模式...")
    
    if 'qwen3_5' in TEMPLATE_MAPPING:
        # 获取映射内容
        template_info = TEMPLATE_MAPPING['qwen3_5']
        
        # 4.0 兼容性逻辑：判断它是字典还是对象
        if isinstance(template_info, dict):
            # 如果是旧版字典结构
            meta = template_info.get('template_meta', template_info)
        else:
            # 如果是新版对象结构（对应你现在的报错情况）
            meta = template_info
            
        # 强制修改属性
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

    print(f"正在加载基础模型: {MODEL_PATH}")
    
    # 2. 初始化引擎（直接使用原始的 qwen3_5 名称即可，因为内存里它已经被改了）
    engine = TransformersEngine(
        model=MODEL_PATH, 
        dtype=torch.bfloat16, 
        template_type='qwen3_5'
    )
    

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