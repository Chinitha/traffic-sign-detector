"""
Evaluate a trained YOLOv8 model: compute mAP/precision/recall, generate
per-class performance plots, and write results to results/metrics.json.

Usage:
    python src/evaluate.py --weights runs/detect/baseline/weights/best.pt --name baseline
    python src/evaluate.py --weights runs/detect/grouped/weights/best.pt --name grouped --data data/gtsdb_yolo_grouped/data.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"


def evaluate(weights_path: str, run_name: str, data_yaml: str):
    model   = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="test")

    summary = {
        "run_name":  run_name,
        "data_yaml": data_yaml,
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    per_class   = {}
    class_names = model.names
    for i, class_id in enumerate(metrics.box.ap_class_index):
        name = class_names[int(class_id)]
        per_class[name] = {
            "AP50":    float(metrics.box.ap50[i]),
            "AP50_95": float(metrics.box.ap[i]),
        }
    summary["per_class"] = per_class

    save_results(summary, run_name)
    plot_per_class_ap(per_class, run_name)
    return summary


def save_results(summary: dict, run_name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"

    all_results = {}
    if metrics_path.exists():
        all_results = json.loads(metrics_path.read_text())

    all_results[run_name] = summary
    metrics_path.write_text(json.dumps(all_results, indent=2))

    print(f"Saved metrics to {metrics_path}")
    print(f"  mAP@0.5:      {summary['mAP50']:.3f}")
    print(f"  mAP@0.5:0.95: {summary['mAP50_95']:.3f}")
    print(f"  Precision:    {summary['precision']:.3f}")
    print(f"  Recall:       {summary['recall']:.3f}")


def save_results(summary: dict, run_name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"

    all_results = {}
    if metrics_path.exists():
        all_results = json.loads(metrics_path.read_text())

    all_results[run_name] = summary
    metrics_path.write_text(json.dumps(all_results, indent=2))

    print(f"Saved metrics to {metrics_path}")
    print(f"  mAP@0.5:      {summary['mAP50']:.3f}")
    print(f"  mAP@0.5:0.95: {summary['mAP50_95']:.3f}")
    print(f"  Precision:    {summary['precision']:.3f}")
    print(f"  Recall:       {summary['recall']:.3f}")


def plot_per_class_ap(per_class: dict, run_name: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not per_class:
        print("No per-class data to plot")
        return

    names  = list(per_class.keys())
    values = [per_class[n]["AP50"] for n in names]

    sorted_pairs      = sorted(zip(names, values), key=lambda x: x[1])
    names_s, values_s = zip(*sorted_pairs)

    plt.figure(figsize=(10, max(4, len(names) * 0.5)))
    plt.barh(names_s, values_s, color="#4C72B0")
    plt.xlabel("AP@0.5")
    plt.title(f"Per-class AP@0.5 — {run_name}")
    plt.xlim(0, 1)
    plt.tight_layout()

    out_path = PLOTS_DIR / f"per_class_ap_{run_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--name",    type=str, default="run",
                        help="Name for this evaluation run")
    parser.add_argument("--data",    type=str, default="data/gtsdb_yolo/data.yaml",
                        help="Path to data.yaml")
    args = parser.parse_args()
    evaluate(args.weights, args.name, args.data)


if __name__ == "__main__":
    main()