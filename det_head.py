import os
import torch
from swift import TransformersEngine, RequestConfig, InferRequest
from swift.utils import seed_everything
# 导入极简版补丁
from model_patcher import patch_qwen_spatial_residual

# ================= 配置区 =================
MODEL_PATH = "/home/hk-project-pai00054/wm9835/qwen35/output_qwen35_expert/v5-20260305-144841_nothinking_version/checkpoint-2086"
ENHANCER_WEIGHT_PATH = os.path.join(MODEL_PATH, "/home/hk-project-pai00054/wm9835/qwen35/spatial_enhancer_final.pth") # 你以后训练的权重
IMAGE_DIR = "/home/hk-project-pai00054/wm9835/qwen35/test"

SYSTEM_PROMPT = "You are a document analysis expert. For each page, output the correct containing bounding boxes [ymin, xmin, ymax, xmax] and corresponding labels, ocr text and alt text for figures or tables in this page."
USER_PROMPT = "Analyze the page layout: provide bounding boxes, labels, OCR text, and alt text for figures or tables in this page."

def run_batch_inference():
    seed_everything(42)

    print(f"正在加载模型: {MODEL_PATH}")
    engine = TransformersEngine(model=MODEL_PATH, torch_dtype=torch.bfloat16)
    
    # 核心注入步骤：打上空间残差增强补丁
    patch_qwen_spatial_residual(engine.model, weight_path=ENHANCER_WEIGHT_PATH)
    
    request_config = RequestConfig(max_tokens=8192, temperature=0)

    print(f"开始扫描目录: {IMAGE_DIR}")
    for root, dirs, files in os.walk(IMAGE_DIR):
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files: continue
            
        for img_name in sorted(img_files):
            img_path = os.path.join(root, img_name)
            output_filename = os.path.splitext(img_name)[0] + ".mmd"
            output_path = os.path.join(root, output_filename)

            if os.path.exists(output_path): continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<img>{img_path}</img>{USER_PROMPT}"}
            ]
            infer_request = InferRequest(messages=messages)

            print(f"  正在处理: {img_name}...")
            try:
                # 引擎会按照打好补丁的流程运行，生成的文本直接具备强空间感知
                resp_list = engine.infer([infer_request], request_config)
                llm_response = resp_list[0].choices[0].message.content
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
                
            except Exception as e:
                print(f"  处理 {img_name} 时发生错误: {e}")

if __name__ == "__main__":
    run_batch_inference()