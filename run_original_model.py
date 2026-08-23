import os
import torch
from swift import TransformersEngine, RequestConfig, InferRequest
from swift.utils import seed_everything

MODEL_PATH = ""
IMAGE_DIR = ""

SYSTEM_PROMPT = "You are a document analysis expert. For each page, output the correct containing bounding boxes [ymin, xmin, ymax, xmax] and corresponding labels, ocr text and alt text for figures or tables in this page."
USER_PROMPT = "Analyze the page layout: provide bounding boxes, labels, OCR text, and alt text for figures or tables in this page."

def run_batch_inference():
    seed_everything(42)
    print(f"loading model: {MODEL_PATH}")
    engine = TransformersEngine(model=MODEL_PATH, torch_dtype=torch.bfloat16)
    
    request_config = RequestConfig(max_tokens=8192, temperature=0)
    print(f"开始扫描目录: {IMAGE_DIR}")
    
    for root, dirs, files in os.walk(IMAGE_DIR):
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            continue
            
        print(f"\ninto: {root} (find {len(img_files)} images)")
        for img_name in sorted(img_files):
            img_path = os.path.join(root, img_name)

            output_filename = os.path.splitext(img_name)[0] + ".mmd"
            output_path = os.path.join(root, output_filename)

            if os.path.exists(output_path):
                print(f"  skip: {img_name} (already exists {output_filename})")
                continue
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<img>{img_path}</img>{USER_PROMPT}"}
            ]
            infer_request = InferRequest(messages=messages)

            print(f"  processing: {img_name}...")
            try:
                resp_list = engine.infer([infer_request], request_config)
                response = resp_list[0].choices[0].message.content
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                print(f"  save to: {output_path}")
                
            except Exception as e:
                print(f"  processing {img_name} failed: {e}")

if __name__ == "__main__":
    run_batch_inference()
