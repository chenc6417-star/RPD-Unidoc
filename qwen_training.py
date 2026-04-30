import json
import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
# 原始数据根目录
ROOT_DIR = "/home/hk-project-pai00054/wm9835/DeepSeek-OCR/DeepSeek-OCR-master/output_new"
# 输出文件名
OUTPUT_JSONL = "qwen35_unified_full_v3.jsonl"
# 统一的任务指令
UNIFIED_PROMPT = "Analyze the page layout: provide all layout labels with bounding boxes coordinates OCR content, and generate alt text for figures or tables in this page."

def normalize_bbox(bbox, width, height):
    """
    将 [x1, y1, x2, y2] 转换为 Qwen 标准的 [ymin, xmin, ymax, xmax] (0-1000)
    并添加特殊的 bbox 标签
    """
    x1, y1, x2, y2 = bbox
    # 严格限缩在 0-1000 范围内
    ymin = max(0, min(1000, int(round(y1 / height * 1000))))
    xmin = max(0, min(1000, int(round(x1 / width * 1000))))
    ymax = max(0, min(1000, int(round(y2 / height * 1000))))
    xmax = max(0, min(1000, int(round(x2 / width * 1000)))) # 修正: 应基于原图比例
    

    # 返回带标签的字符串格式，例如: <|box_start|>(ymin,xmin,ymax,xmax)<|box_end|>
    return f"[{ymin},{xmin},{ymax},{xmax}]"

def clean_content(text):
    """清洗文本，保留换行符并进行转义，防止破坏 JSON 结构"""
    if not text:
        return ""
    # 不再替换为 ' '，而是保留并在后续写入 jsonl 时通过 json.dumps 自动处理转义
    # 如果你想手动处理更安全，可以写成：
    return str(text).replace("'", "\\'").strip()

def main():
    all_samples = []
    json_files = []

    # 1. 递归扫描所有原始标注 JSON
    for root, dirs, files in os.walk(ROOT_DIR):
        # 避开 markdown 文件夹，只找根目录下的标注
        if "markdown" in root: continue 
        for file in files:
            if file.endswith(".json") and not file.endswith("_alt.json"):
                json_files.append(os.path.join(root, file))

    print(f"找到 {len(json_files)} 个标注文件，开始构建全量整合数据集...")

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                elements = json.load(f)
            if not elements: continue

            # 2. 自动定位对应的全页图片路径
            # 逻辑：从 results/xxx.json 向上跳转找原图
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

            # 3. 获取图片尺寸进行坐标归一化
            with Image.open(full_img_path) as img:
                w, h = img.size

            # 4. 按照阅读顺序构建 Assistant 回复
            sorted_elems = sorted(elements, key=lambda x: x.get("reading_order", 0))
            page_elements_list = []
            
            for e in sorted_elems:
                if "bbox" not in e or not e["bbox"]: continue
                
                # 归一化坐标
                box = normalize_bbox(e["bbox"], w, h)
                label = e.get('label', 'text')
                ocr_text = clean_content(e.get('text', ''))[:4000] # 限制长度防止 Token 过载
                
                # 如果是图表且有 alt_text，则整合进同一个条目
                if label in ["fig", "tab"] and e.get("alt_text"):
                    alt = clean_content(e["alt_text"])
                    # 格式：[坐标, 标签, OCR, 描述]
                    page_elements_list.append(f"[ '{label}', {box}, '{ocr_text}', '{alt}']")
                else:
                    # 普通格式：[坐标, 标签, OCR]
                    page_elements_list.append(f"['{label}', {box},  '{ocr_text}']")

            # 5. 封装为 SWIFT 4.0 消息格式
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

    # 6. 保存最终训练集
    print(f"\n数据集构建完成！总页面数: {len(all_samples)}")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()