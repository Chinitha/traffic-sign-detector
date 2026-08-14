"""
Run inference on the validation/test set and bucket errors into:
  - False Negatives: ground-truth boxes the model missed entirely
  - False Positives: detections with no matching ground-truth box
  - Low-confidence correct: correct detections below a confidence threshold
    (these are "almost-failures" -- valuable for review even though technically correct)

Outputs a CSV report and saves annotated images for the worst cases, mirroring
the kind of failure-mode triage used in ADAS perception validation.

Usage:
    python src/error_analysis.py --weights runs/detect/baseline/weights/best.pt \
        --conf-threshold 0.5 --iou-threshold 0.5
"""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

DATA_DIR = Path("data/gtsdb_yolo")
RESULTS_DIR = Path("results")
LOW_CONF_THRESHOLD = 0.5


def load_ground_truth(label_path: Path, img_w: int, img_h: int):
    """Read YOLO-format label file, return list of (class_id, x1, y1, x2, y2) in pixel coords."""
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

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


def analyze_image(model, img_path: Path, label_path: Path, iou_threshold: float):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gt_boxes = load_ground_truth(label_path, w, h)

    results = model.predict(str(img_path), verbose=False)[0]
    pred_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        pred_boxes.append((class_id, x1, y1, x2, y2, conf))

    matched_gt = set()
    matched_pred = set()
    low_confidence_correct = []

    for pi, (p_class, px1, py1, px2, py2, conf) in enumerate(pred_boxes):
        best_iou, best_gi = 0.0, None
        for gi, (g_class, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gi in matched_gt or g_class != p_class:
                continue
            cur_iou = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if cur_iou > best_iou:
                best_iou, best_gi = cur_iou, gi

        if best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            if conf < LOW_CONF_THRESHOLD:
                low_confidence_correct.append((p_class, conf))

    false_positives = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
    false_negatives = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]

    return {
        "image": img_path.name,
        "num_gt": len(gt_boxes),
        "num_pred": len(pred_boxes),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "low_confidence_correct": len(low_confidence_correct),
    }


def run_error_analysis(weights_path: str, split: str, conf_threshold: float, iou_threshold: float):
    model = YOLO(weights_path)
    model.overrides["conf"] = conf_threshold

    img_dir = DATA_DIR / "images" / split
    label_dir = DATA_DIR / "labels" / split
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}. Check dataset path / run download_data.py.")

    rows = []
    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        row = analyze_image(model, img_path, label_path, iou_threshold)
        rows.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "error_analysis.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_fp = sum(r["false_positives"] for r in rows)
    total_fn = sum(r["false_negatives"] for r in rows)
    total_lowconf = sum(r["low_confidence_correct"] for r in rows)

    print(f"Analyzed {len(rows)} images")
    print(f"  Total false positives: {total_fp}")
    print(f"  Total false negatives: {total_fn}")
    print(f"  Total low-confidence-correct (< {LOW_CONF_THRESHOLD}): {total_lowconf}")
    print(f"Saved per-image breakdown to {out_csv}")

    # Surface the worst images first -- these are the ones worth reviewing manually
    worst = sorted(rows, key=lambda r: r["false_positives"] + r["false_negatives"], reverse=True)[:10]
    print("\nTop 10 worst images (most FP+FN):")
    for r in worst:
        print(f"  {r['image']}: FP={r['false_positives']}, FN={r['false_negatives']}")


def main():
    parser = argparse.ArgumentParser(description="Bucket detection errors for review")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_error_analysis(args.weights, args.split, args.conf_threshold, args.iou_threshold)


if __name__ == "__main__":
    main()
