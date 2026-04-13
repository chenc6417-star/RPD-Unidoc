import os
import json
import cv2
import random

# ================= 配置区 =================
DATA_DIR = "/home/hk-project-pai00054/wm9835/qwen35/benchmark_before/3290605-3300231" 
SAVE_DIR = "vis_results_auto_scale" 
TARGET_S = 1000  # 模型内部坐标的归一化基准（1000x1000格式）
# =========================================

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def visualize_auto_scale():
    # 扫描所有 JSON 文件
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    matched_pairs = []
    for j_name in json_files:
        img_name = j_name.replace('.json', '.png')
        if os.path.exists(os.path.join(DATA_DIR, img_name)):
            matched_pairs.append((j_name, img_name))

    if not matched_pairs:
        print("未找到匹配的 JSON-图片 对，请检查路径。")
        return

    # 随机采样 5 个进行验证
    selected_pairs = random.sample(matched_pairs, min(5, len(matched_pairs)))
    print(f"--- 正在根据图片实际尺寸进行自适应缩放绘制 ({len(selected_pairs)}个) ---")

    for j_name, i_name in selected_pairs:
        # 1. 读取 JSON 数据
        with open(os.path.join(DATA_DIR, j_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 兼容不同层级的 JSON 结构
        elements = data.get("elements", data) if isinstance(data, dict) else data
        
        # 2. 读取图片并获取实际像素尺寸
        img = cv2.imread(os.path.join(DATA_DIR, i_name))
        if img is None:
            print(f"无法读取图片: {i_name}")
            continue
        
        real_h, real_w, _ = img.shape 
        print(f"处理文件: {i_name} | 实际分辨率: {real_w}x{real_h}")

        for idx, el in enumerate(elements):
            bbox_raw = el.get("bbox", [])
            if not bbox_raw:
                continue
            
            # 处理嵌套格式，如 [[x1, y1, x2, y2]] 或 [x1, y1, x2, y2]
            bbox = bbox_raw[0] if isinstance(bbox_raw[0], list) else bbox_raw
            
            # ---------------- 核心缩放对齐逻辑 ----------------
            # 将 1000*1000 格式的坐标映射到实际图片分辨率
            # 公式: 实际像素 = (坐标 / 1000) * 实际尺寸
            
            x1 = int(bbox[0] * real_w / TARGET_S)
            y1 = int(bbox[1] * real_h / TARGET_S)
            x2 = int(bbox[2] * real_w / TARGET_S)
            y2 = int(bbox[3] * real_h / TARGET_S)
            
            # 边界保护：确保坐标不超出图片像素范围
            x1, x2 = max(0, min(x1, real_w)), max(0, min(x2, real_w))
            y1, y2 = max(0, min(y1, real_h)), max(0, min(y2, real_h))
            # ------------------------------------------------

            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签文本
            label_text = f"{idx}:{el.get('label', 'obj')}"
            cv2.putText(img, label_text, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 保存结果
        save_path = os.path.join(SAVE_DIR, f"scaled_{i_name}")
        cv2.imwrite(save_path, img)
        print(f"  [SUCCESS] 结果已保存: {save_path}")

if __name__ == "__main__":
    visualize_auto_scale()