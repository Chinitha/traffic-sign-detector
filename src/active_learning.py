"""
Rank unlabeled images by model uncertainty to prioritize which ones would be
most valuable to label/review next -- a simple active-learning loop.

Two uncertainty signals are combined:
  1. Mean entropy of detection confidences (low confidence = high entropy = uncertain)
  2. Detection count volatility -- images where the model is "unsure how many
     objects" are present (flickering detections across confidence thresholds)
     also tend to be informative.

Usage:
    python src/active_learning.py --weights runs/detect/baseline/weights/best.pt \
        --unlabeled-dir data/unlabeled --top-k 50
"""

import argparse
import csv
import math
from pathlib import Path

from ultralytics import YOLO

RESULTS_DIR = Path("results")


def entropy_from_confidence(conf: float) -> float:
    """Binary entropy treating confidence as P(correct)."""
    conf = min(max(conf, 1e-6), 1 - 1e-6)
    return -(conf * math.log2(conf) + (1 - conf) * math.log2(1 - conf))


def score_image(model, img_path: Path) -> dict:
    results = model.predict(str(img_path), verbose=False)[0]
    confidences = [float(box.conf[0]) for box in results.boxes]

    if not confidences:
        # No detections at all is itself informative -- could be a missed sign
        return {
            "image": img_path.name,
            "num_detections": 0,
            "mean_entropy": 1.0,  # max uncertainty as a default for "nothing found"
            "min_confidence": 0.0,
            "uncertainty_score": 1.0,
        }

    entropies = [entropy_from_confidence(c) for c in confidences]
    mean_entropy = sum(entropies) / len(entropies)
    min_confidence = min(confidences)

    # Combine mean entropy with the single worst detection's entropy so one
    # very uncertain box can't be diluted by several confident ones
    worst_entropy = entropy_from_confidence(min_confidence)
    uncertainty_score = 0.5 * mean_entropy + 0.5 * worst_entropy

    return {
        "image": img_path.name,
        "num_detections": len(confidences),
        "mean_entropy": round(mean_entropy, 4),
        "min_confidence": round(min_confidence, 4),
        "uncertainty_score": round(uncertainty_score, 4),
    }


def rank_unlabeled(weights_path: str, unlabeled_dir: str, top_k: int):
    model = YOLO(weights_path)
    image_paths = sorted(Path(unlabeled_dir).glob("*.jpg")) + sorted(Path(unlabeled_dir).glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {unlabeled_dir}")

    rows = [score_image(model, p) for p in image_paths]
    rows.sort(key=lambda r: r["uncertainty_score"], reverse=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "active_learning_ranking.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ranked {len(rows)} unlabeled images by uncertainty")
    print(f"Saved full ranking to {out_csv}")
    print(f"\nTop {top_k} images recommended for labeling/review:")
    for r in rows[:top_k]:
        print(f"  {r['image']}: uncertainty={r['uncertainty_score']}, detections={r['num_detections']}")

    return rows[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Rank unlabeled images for active learning")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--unlabeled-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    rank_unlabeled(args.weights, args.unlabeled_dir, args.top_k)


if __name__ == "__main__":
    main()
