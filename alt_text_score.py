import os
import json
import numpy as np
import Levenshtein
import re
import torch  # 新增：用于检测 GPU
from evaluate import load

# ================= 配置区 =================
PRED_DIR = "/home/hk-project-pai00054/wm9835/qwen35/gpt_alt"
GT_DIR = "/home/hk-project-pai00054/wm9835/Dolphin/ground_truth"

LABEL_MAP = {
    "table": "tab",
    "image": "fig",
    "figure": "fig"
}

# 检测 GPU 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current Device: {DEVICE}")
# =========================================

# 加载指标
bleu = load("bleu")
meteor = load("meteor")
bertscore = load("bertscore")

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
    unionArea = float(box1Area + box2Area - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def get_page_scale(preds_raw, gts_raw):
    def get_max_coords(elements):
        all_coords = []
        for e in elements:
            if not isinstance(e, dict): continue
            b = e.get("bbox", [[0,0,0,0]])
            box = b[0] if isinstance(b[0], list) else b
            if len(box) >= 4: all_coords.append(box)
        if not all_coords: return 0, 0
        arr = np.array(all_coords)
        return np.max(arr[:, [0, 2]]), np.max(arr[:, [1, 3]])
    p_mx, p_my = get_max_coords(preds_raw)
    g_mx, g_my = get_max_coords(gts_raw)
    sx = g_mx / p_mx if p_mx > 0 else 1.0
    sy = g_my / p_my if p_my > 0 else 1.0
    return sx, sy

def evaluate_json(predict_json, gt_json, iou_threshold=0.5):
    preds_raw = predict_json.get("elements", []) if isinstance(predict_json, dict) else predict_json
    gts_raw = gt_json if isinstance(gt_json, list) else gt_json.get("elements", [])

    if not isinstance(preds_raw, list) or not isinstance(gts_raw, list):
        return None

    sx, sy = get_page_scale(preds_raw, gts_raw)

    preds = []
    for p in preds_raw:
        if not isinstance(p, dict): continue
        b = p.get("bbox", [[0,0,0,0]])
        raw_box = b[0] if isinstance(b[0], list) else b
        raw_label = str(p.get("label") or p.get("category") or "").lower()
        mapped_label = LABEL_MAP.get(raw_label, raw_label)
        
        preds.append({
            "bbox": [raw_box[0]*sx, raw_box[1]*sy, raw_box[2]*sx, raw_box[3]*sy],
            "alt_text": p.get("alt_text", ""),
            "label": mapped_label
        })

    gts = []
    for g in gts_raw:
        if not isinstance(g, dict): continue
        b = g.get("bbox", [0,0,0,0])
        g_bbox = b[0] if isinstance(b[0], list) else b
        gts.append({
            "bbox": g_bbox,
            "alt_text": g.get("alt_text", ""),
            "label": str(g.get("label") or g.get("category") or "").lower()
        })

    paired_results = []
    used_gts = set()

    for p in preds:
        best_iou, best_gt_idx = 0, -1
        for g_idx, g in enumerate(gts):
            if g_idx in used_gts: continue
            iou = calculate_iou(p["bbox"], g["bbox"])
            if iou >= iou_threshold and p["label"] == g["label"]:
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, g_idx
        
        if best_gt_idx >= 0:
            used_gts.add(best_gt_idx)
            paired_results.append({
                "pred_alt": p["alt_text"].strip(),
                "gt_alt": gts[best_gt_idx]["alt_text"].strip()
            })
    return paired_results

def batch_evaluate(predict_root, gt_root):
    def scan(root_dir):
        m = {}
        for r, _, fs in os.walk(root_dir):
            for f in fs:
                if f.endswith(".json") and "page_" in f:
                    m[f] = os.path.join(r, f)
        return m

    pred_map = scan(predict_root)
    gt_map = scan(gt_root)
    
    all_pred_alts = []
    all_gt_alts = []
    total_pages = 0

    for fname in pred_map:
        if fname in gt_map:
            try:
                with open(pred_map[fname], 'r', encoding='utf-8') as f: p_d = json.load(f)
                with open(gt_map[fname], 'r', encoding='utf-8') as f: g_d = json.load(f)
                
                pairs = evaluate_json(p_d, g_d)
                if pairs:
                    for pair in pairs:
                        if pair["gt_alt"]:
                            all_pred_alts.append(pair["pred_alt"])
                            all_gt_alts.append(pair["gt_alt"])
                    total_pages += 1
            except Exception:
                continue

    if not all_pred_alts:
        return "No matching objects found."

    print(f"Computing metrics for {len(all_pred_alts)} objects...")

    # BLEU 和 METEOR 主要是 CPU 运算
    b4 = bleu.compute(predictions=all_pred_alts, references=[[g] for g in all_gt_alts])
    mt = meteor.compute(predictions=all_pred_alts, references=all_gt_alts)
    
    # BERTScore：使用 GPU 加速
    # batch_size 可以根据显存大小调整，通常 16 或 32 比较稳妥
    bs = bertscore.compute(
        predictions=all_pred_alts, 
        references=all_gt_alts, 
        lang="en", 
        device=DEVICE,
        batch_size=16  
    )

    return {
        "Alt_Text_Evaluation": {
            "Total_Matched_Pages": total_pages,
            "Total_Matched_Objects": len(all_pred_alts),
            "BLEU-4": round(b4['bleu'], 4),
            "METEOR": round(mt['meteor'], 4),
            "BERTScore_F1": round(np.mean(bs['f1']), 4)
        }
    }

if __name__ == "__main__":
    report = batch_evaluate(PRED_DIR, GT_DIR)
    import pprint
    pprint.pprint(report)