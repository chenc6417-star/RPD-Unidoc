import os
import json
import re

# ================= 配置区 =================
BASE_DIR = "/home/hk-project-pai00054/wm9835/qwen35/test"
# ==========================================

def extract_items_final(content):
    items = []
    # 匹配 [ [bbox], 'label', 'text' ... ]
    item_pattern = r'\[\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])\s*,\s*[\'"](.*?)[\'"]\s*,\s*[\'"](.*?)[\'"](?:\s*,\s*[\'"](.*?)[\'"])?\s*\]'
    
    matches = re.finditer(item_pattern, content, re.DOTALL)
    
    for i, match in enumerate(matches):
        bbox_str = match.group(1)
        label = match.group(2)
        text = match.group(3)
        alt_text = match.group(4)
        
        # 1. 解析原始坐标数字
        raw_bbox = [int(x.strip()) for x in bbox_str.strip("[]").split(",")]
        
        # 2. 执行 x 和 y 对调 (索引 0<->1, 2<->3)
        # 假设原格式: [y0, x0, y1, x1] -> 新格式: [x0, y0, x1, y1]
        if len(raw_bbox) == 4:
            swapped_bbox = [raw_bbox[1], raw_bbox[0], raw_bbox[3], raw_bbox[2]]
        else:
            swapped_bbox = raw_bbox # 防止意外情况
        
        entry = {
            "label": label,
            "bbox": swapped_bbox, # 使用对调后的坐标
            "text": text.replace('\\n', '\n'),
            "reading_order": i
        }
        
        if alt_text is not None:
            entry["alt_text"] = alt_text.replace('\\n', '\n')
            
        items.append(entry)
            
    return items

def main():
    if not os.path.exists(BASE_DIR):
        print(f"路径不存在: {BASE_DIR}")
        return

    count = 0
    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            if filename.endswith('.mmd'):
                file_path = os.path.join(root, filename)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(root, f"{base_name}.json")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip(): continue
                        
                        json_data = extract_items_final(content)
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        # 使用 indent=4 使输出格式与你图片中的漂亮格式一致
                        json.dump(json_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"已处理 (坐标已对调): {output_file}")
                    count += 1
                except Exception as e:
                    print(f"出错 {file_path}: {e}")

    print(f"\n全部完成，共处理 {count} 个文件。")

if __name__ == "__main__":
    main()