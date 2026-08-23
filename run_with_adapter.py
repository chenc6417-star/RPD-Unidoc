import os
import copy
import torch

from swift.template import TEMPLATE_MAPPING, register_template, TemplateMeta
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
from swift.utils import seed_everything

from model_patcher import patch_qwen_spatial_residual

BASE_MODEL_PATH = ""  
LORA_PATH = ""
IMAGE_DIR = ""

SYSTEM_PROMPT = "You are a document analysis expert. You can read text, analyze layouts, and locate elements using precise bounding boxes [ymin, xmin, ymax, xmax]. Answer the user's question directly based on the provided image."
USER_PROMPT = "Analyze the page layout: provide bounding boxes coordinate, labels, OCR content, and alt text for figures or tables in this page."
# ==========================================

def apply_monkey_patch():
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
        except AttributeError as e:
    else:
        print("No TEMPLATE_MAPPING for qwen3_5")

def run_batch_inference():
    seed_everything(42)
    apply_monkey_patch()
    
    try:
        engine = TransformersEngine(
            model=BASE_MODEL_PATH, 
            adapters=[LORA_PATH],   
            dtype=torch.bfloat16, 
            template_type='qwen3_5'
        )
    except TypeError:
        engine = TransformersEngine(
            model=BASE_MODEL_PATH, 
            adapter_id_or_path=LORA_PATH, 
            dtype=torch.bfloat16, 
            template_type='qwen3_5'
        )
    
    print(" initializing the structural set")
    patch_qwen_spatial_residual(engine.model, weight_path="")
    
    enhancer_path = os.path.join( "")
    print(f"finding the enhancement weight: {enhancer_path}")
    
    if os.path.exists(enhancer_path):
        try:
            raw_state_dict = torch.load(enhancer_path, map_location=engine.model.device, weights_only=True)
            cleaned_state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith("spatial_enhancer."):
                    new_key = k.replace("spatial_enhancer.", "", 1)
                    cleaned_state_dict[new_key] = v
                else:
                    cleaned_state_dict[k] = v
            
            engine.model.spatial_enhancer.load_state_dict(cleaned_state_dict)
            
            current_gate = engine.model.spatial_enhancer.gate.data.item()
            print(f"the current gate value: {current_gate:.4f}")
        except Exception as e:
            print(f"fall to find the path: {enhancer_path} : {e}")

    request_config = RequestConfig(max_tokens=8192, temperature=0.0)

    print(f"start to process: {IMAGE_DIR}")
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

            print(f" processing: {img_name}...")
            try:
                resp_list = engine.infer([infer_request], request_config)
                llm_response = resp_list[0].choices[0].message.content
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
                print(f" final file is finished")
                
            except Exception as e:
                print(f" error: {e}")

if __name__ == "__main__":
    run_batch_inference()
