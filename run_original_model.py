import os
import torch
# 推理核心组件从顶级包导入
from swift import TransformersEngine, RequestConfig, InferRequest
# 工具函数从 utils 导入
from swift.utils import seed_everything

# ================= 配置区 =================
MODEL_PATH = "/home/hk-project-pai00054/wm9835/qwen35/output_qwen35_expert/v6-20260309-170907_thinking_version/checkpoint-2086"
IMAGE_DIR = "/home/hk-project-pai00054/wm9835/qwen35/3290605-3300231"

SYSTEM_PROMPT = "You are a document analysis expert. For each page, output the correct containing bounding boxes [ymin, xmin, ymax, xmax] and corresponding labels, ocr text and alt text for figures or tables in this page."
USER_PROMPT = "Analyze the page layout: provide bounding boxes, labels, OCR text, and alt text for figures or tables in this page."

def run_batch_inference():
    seed_everything(42)

    # 1. 初始化引擎 (Swift 4.0 TransformersEngine 不依赖 vllm)
    print(f"正在加载模型: {MODEL_PATH}")
    engine = TransformersEngine(model=MODEL_PATH, torch_dtype=torch.bfloat16)
    
    # 推理配置 (根据需要调整 max_tokens)
    request_config = RequestConfig(max_tokens=8192, temperature=0)

    # 2. 递归遍历子文件夹
    print(f"开始扫描目录: {IMAGE_DIR}")
    
    # 使用 os.walk 遍历所有层级的子目录
    for root, dirs, files in os.walk(IMAGE_DIR):
        # 筛选当前目录下的图片文件
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not img_files:
            continue
            
        print(f"\n进入目录: {root} (发现 {len(img_files)} 张图片)")

        for img_name in sorted(img_files):
            img_path = os.path.join(root, img_name)

            output_filename = os.path.splitext(img_name)[0] + ".mmd"
            output_path = os.path.join(root, output_filename)

            if os.path.exists(output_path):
                print(f"  跳过: {img_name} (已存在 {output_filename})")
                continue
            # 3. 构建推理请求 (使用 <img> 标签包裹路径)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<img>{img_path}</img>{USER_PROMPT}"}
            ]
            infer_request = InferRequest(messages=messages)

            # 4. 执行推理
            print(f"  正在处理: {img_name}...")
            try:
                # TransformersEngine 推理
                resp_list = engine.infer([infer_request], request_config)
                response = resp_list[0].choices[0].message.content
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                print(f"  已保存至: {output_path}")
                
            except Exception as e:
                print(f"  处理 {img_name} 时发生错误: {e}")

if __name__ == "__main__":
    run_batch_inference()