"""
Run inference on the val set and bucket errors into:
  - False Positives
  - False Negatives  
  - Low-confidence correct detections

Usage:
    python src/error_analysis.py \
        --weights runs/detect/grouped/weights/best.pt \
        --data-dir data/gtsdb_yolo_grouped \
        --split val \
        --conf-threshold 0.5
"""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

RESULTS_DIR = Path("results")


def load_ground_truth(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        class_id, xc, yc, w, h = map(float, line.split())
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((int(class_id), x1, y1, x2, y2))
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def analyze_image(model, img_path, label_path, iou_threshold, low_conf_threshold):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gt_boxes = load_ground_truth(label_path, w, h)

    results = model.predict(str(img_path), verbose=False)[0]
    pred_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf     = float(box.conf[0])
        class_id = int(box.cls[0])
        pred_boxes.append((class_id, x1, y1, x2, y2, conf))

    matched_gt, matched_pred = set(), set()
    low_conf_correct = []

    for pi, (p_cls, px1, py1, px2, py2, conf) in enumerate(pred_boxes):
        best_iou, best_gi = 0.0, None
        for gi, (g_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gi in matched_gt or g_cls != p_cls:
                continue
            cur_iou = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if cur_iou > best_iou:
                best_iou, best_gi = cur_iou, gi
        if best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            if conf < low_conf_threshold:
                low_conf_correct.append((p_cls, conf))

    false_positives = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
    false_negatives = [gt_boxes[i]   for i in range(len(gt_boxes))   if i not in matched_gt]

    return {
        "image":                img_path.name,
        "num_gt":               len(gt_boxes),
        "num_pred":             len(pred_boxes),
        "false_positives":      len(false_positives),
        "false_negatives":      len(false_negatives),
        "low_confidence_correct": len(low_conf_correct),
    }


def run_error_analysis(weights, data_dir, split, conf_threshold, iou_threshold):
    model = YOLO(weights)
    model.overrides["conf"] = conf_threshold

    data_dir = Path(data_dir)
    img_dir   = data_dir / "images" / split
    label_dir = data_dir / "labels" / split
    img_paths = sorted(img_dir.glob("*.jpg"))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    rows = []
    for img_path in img_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        row = analyze_image(model, img_path, label_path, iou_threshold, conf_threshold)
        rows.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "error_analysis.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_fp      = sum(r["false_positives"]        for r in rows)
    total_fn      = sum(r["false_negatives"]         for r in rows)
    total_lowconf = sum(r["low_confidence_correct"]  for r in rows)

    print(f"Analyzed {len(rows)} images  (conf threshold: {conf_threshold})")
    print(f"  Total false positives:              {total_fp}")
    print(f"  Total false negatives:              {total_fn}")
    print(f"  Total low-confidence-correct:       {total_lowconf}")
    print(f"Saved per-image breakdown to {out_csv}")

    worst = sorted(rows, key=lambda r: r["false_positives"] + r["false_negatives"], reverse=True)[:10]
    print("\nTop 10 worst images (most FP+FN):")
    for r in worst:
        print(f"  {r['image']}: FP={r['false_positives']}, FN={r['false_negatives']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",        type=str, required=True)
    parser.add_argument("--data-dir",       type=str, default="data/gtsdb_yolo_grouped")
    parser.add_argument("--split",          type=str, default="val", choices=["train","val","test"])
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold",  type=float, default=0.5)
    args = parser.parse_args()

    run_error_analysis(
        args.weights, args.data_dir, args.split,
        args.conf_threshold, args.iou_threshold
    )


if __name__ == "__main__":
    main()