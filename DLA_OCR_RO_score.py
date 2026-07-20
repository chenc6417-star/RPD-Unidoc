import os
import json
import numpy as np
import Levenshtein
import re

# ================= 配置区 =================
PRED_DIR = "/home/hk-project-pai00054/wm9835/qwen35/test1"
GT_DIR = "/home/hk-project-pai00054/wm9835/Dolphin/ground_truth"
# =========================================

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

def clean_text(text):
    if not text: return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.lower()

def evaluate_json(predict_json, gt_json, iou_threshold=0.5):
    """
    TP 只有在框 IoU >= iou_threshold 且 label 完全匹配才算
    OCR 分数单独计算 text
    """
    preds_raw = predict_json.get("elements", []) if isinstance(predict_json, dict) else predict_json
    gts_raw = gt_json if isinstance(gt_json, list) else gt_json.get("elements", [])

    if not isinstance(preds_raw, list) or not isinstance(gts_raw, list):
        return {"f1":0.0,"ro":0.0,"ocr":0.0,"tp":0,"bbox_mIoU":0.0}

    sx, sy = get_page_scale(preds_raw, gts_raw)

    preds = []
    for p in preds_raw:
        if not isinstance(p, dict): continue
        b = p.get("bbox", [[0,0,0,0]])
        raw_box = b[0] if isinstance(b[0], list) else b
        preds.append({
            "bbox": [raw_box[0]*sx, raw_box[1]*sy, raw_box[2]*sx, raw_box[3]*sy],
            "order": p.get("reading_order", 0),
            "text": clean_text(p.get("text", "")),
            "label": p.get("label", "")
        })

    gts = []
    for g in gts_raw:
        if not isinstance(g, dict): continue
        b = g.get("bbox", [0,0,0,0])
        g_bbox = b[0] if isinstance(b[0], list) else b
        gts.append({
            "bbox": g_bbox,
            "order": g.get("reading_order", 0),
            "text": clean_text(g.get("text", "")),
            "label": g.get("label", "")
        })

    tp = 0
    matched_gt_indices = {}
    used_gts = set()
    matched_ious = []

    # TP 匹配：框 + label
    for p_idx, p in enumerate(preds):
        best_iou, best_gt_idx = 0, -1
        for g_idx, g in enumerate(gts):
            if g_idx in used_gts: continue
            iou = calculate_iou(p["bbox"], g["bbox"])
            if iou >= iou_threshold and p["label"] == g["label"]:
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, g_idx
        if best_gt_idx >= 0:
            tp += 1
            used_gts.add(best_gt_idx)
            matched_gt_indices[p_idx] = best_gt_idx
            matched_ious.append(best_iou)

    FP = len(preds) - tp
    FN = len(gts) - tp
    precision = tp / (tp + FP) if (tp + FP) > 0 else 0
    recall = tp / (tp + FN) if (tp + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # RO 惩罚 FP/FN
    max_char = 1114111
    dummy_char = 0
    p_seq = []
    for i, p in enumerate(preds):
        if i in matched_gt_indices:
            g_idx = matched_gt_indices[i]
            p_seq.append(chr(min(g_idx + 33, max_char)))
        else:
            p_seq.append(chr(min(dummy_char + 33, max_char)))
    g_seq = []
    matched_pred_for_gt = {v:k for k,v in matched_gt_indices.items()}
    for g_idx in range(len(gts)):
        if g_idx in matched_pred_for_gt:
            p_idx = matched_pred_for_gt[g_idx]
            g_seq.append(chr(min(g_idx + 33, max_char)))
        else:
            g_seq.append(chr(min(dummy_char + 33, max_char)))
    ro_score = round(1 - (Levenshtein.distance("".join(p_seq), "".join(g_seq)) / max(len(p_seq), len(g_seq), 1)), 4)

    # OCR 分数只针对 TP 框计算 text
    box_ocr_scores = []
    for p_idx, g_idx in matched_gt_indices.items():
        p_t, g_t = preds[p_idx]["text"], gts[g_idx]["text"]
        if not p_t and not g_t:
            box_ocr_scores.append(1.0)
            continue
        dist = Levenshtein.distance(p_t, g_t)
        max_l = max(len(p_t), len(g_t), 1)
        box_ocr_scores.append(1 - dist / max_l)
    avg_ocr_acc = round(np.mean(box_ocr_scores), 4) if box_ocr_scores else 0.0

    bbox_miou = round(np.mean(matched_ious), 4) if matched_ious else 0.0

    return {"f1":f1,"ro":ro_score,"ocr":avg_ocr_acc,"tp":tp,"bbox_mIoU":bbox_miou}



# ===================== 批量评估 =====================
def batch_evaluate(predict_root, gt_root):
    def scan(root_dir):
        m = {}
        for r, _, fs in os.walk(root_dir):
            for f in fs:
                if "page_" in f and f.endswith(".json") and "_image_" not in f:
                    m[f] = os.path.join(r, f)
        return m

    pred_map = scan(predict_root)
    gt_map = scan(gt_root)
    
    print(f"DEBUG: 扫描到 Predict 文件 {len(pred_map)} 个")
    print(f"DEBUG: 扫描到 GT 文件 {len(gt_map)} 个")

    results = []
    for fname in pred_map:
        if fname in gt_map:
            try:
                with open(pred_map[fname], 'r', encoding='utf-8') as f: p_d = json.load(f)
                with open(gt_map[fname], 'r', encoding='utf-8') as f: g_d = json.load(f)
                res = evaluate_json(p_d, g_d)
                results.append(res)
            except Exception as e:
                print(f"WARNING: {fname} 解析失败: {e}")
                continue

    if not results: return "No match found."

    return {
        "Evaluation_Summary": {
            "Total_Matched_Pages": len(results),
            "Avg_DLA_F1": round(np.mean([r["f1"] for r in results]), 4),
            "Avg_RO_Score_NED": round(np.mean([r["ro"] for r in results]), 4),
            "Avg_OCR_Acc_BoxWise": round(np.mean([r["ocr"] for r in results]), 4),
            "Avg_BBox_mIoU": round(np.mean([r["bbox_mIoU"] for r in results]), 4),
        },
        "Total_Correct_Boxes": sum([r["tp"] for r in results])
    }

# ===================== 主函数 =====================
if __name__ == "__main__":
    report = batch_evaluate(PRED_DIR, GT_DIR)
    print(json.dumps(report, indent=4))
