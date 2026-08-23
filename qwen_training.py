import json
import os
from PIL import Image
from tqdm import tqdm

ROOT_DIR = ""
OUTPUT_JSONL = "qwen35_unified_full_v3.jsonl"
UNIFIED_PROMPT = "Analyze the page layout: provide all layout labels with bounding boxes coordinates OCR content, and generate alt text for figures or tables in this page."

def normalize_bbox(bbox, width, height):

    x1, y1, x2, y2 = bbox
    ymin = max(0, min(1000, int(round(y1 / height * 1000))))
    xmin = max(0, min(1000, int(round(x1 / width * 1000))))
    ymax = max(0, min(1000, int(round(y2 / height * 1000))))
    xmax = max(0, min(1000, int(round(x2 / width * 1000)))) 

    return f"[{ymin},{xmin},{ymax},{xmax}]"

def clean_content(text):
    if not text:
        return ""
    return str(text).replace("'", "\\'").strip()

def main():
    all_samples = []
    json_files = []

    for root, dirs, files in os.walk(ROOT_DIR):
        if "markdown" in root: continue 
        for file in files:
            if file.endswith(".json") and not file.endswith("_alt.json"):
                json_files.append(os.path.join(root, file))


    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                elements = json.load(f)
            if not elements: continue
            results_dir = os.path.dirname(os.path.dirname(json_path))
            subdir_root = os.path.dirname(results_dir)
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            
            full_img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                p = os.path.join(subdir_root, base_name + ext)
                if os.path.exists(p):
                    full_img_path = os.path.abspath(p)
                    break
            
            if not full_img_path: continue
            with Image.open(full_img_path) as img:
                w, h = img.size
            sorted_elems = sorted(elements, key=lambda x: x.get("reading_order", 0))
            page_elements_list = []
            
            for e in sorted_elems:
                if "bbox" not in e or not e["bbox"]: continue
                box = normalize_bbox(e["bbox"], w, h)
                label = e.get('label', 'text')
                ocr_text = clean_content(e.get('text', ''))[:4000]
                
                if label in ["fig", "tab"] and e.get("alt_text"):
                    alt = clean_content(e["alt_text"])
                    page_elements_list.append(f"[ '{label}', {box}, '{ocr_text}', '{alt}']")
                else:
                    page_elements_list.append(f"['{label}', {box},  '{ocr_text}']")

            if page_elements_list:
                assistant_json_str = "[" + ", ".join(page_elements_list) + "]"
                all_samples.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": full_img_path},
                                {"type": "text", "text": UNIFIED_PROMPT}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": assistant_json_str}]
                        }
                    ]
                })

        except Exception as err:
            continue

    print(f"\nfinished, final images: {len(all_samples)}")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
