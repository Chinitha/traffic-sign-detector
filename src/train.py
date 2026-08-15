"""
Train YOLOv8 on the traffic sign dataset.

Three configs:
  - baseline:  small model, no augmentation -- reference score
  - improved:  augmentation tuned for traffic-sign conditions
  - grouped:   4 superclasses instead of 43, frozen backbone -- best for small datasets

Usage:
    python src/train.py --config baseline
    python src/train.py --config improved
    python src/train.py --config grouped
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML         = "data/gtsdb_yolo/data.yaml"
DATA_YAML_GROUPED = "data/gtsdb_yolo_grouped/data.yaml"

CONFIGS = {
    "baseline": dict(
        model="yolov8n.pt",
        epochs=50,
        imgsz=640,
        batch=16,
        name="baseline",
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        mosaic=0.0, mixup=0.0,
    ),
    "improved": dict(
        model="yolov8s.pt",
        epochs=150,
        imgsz=800,
        batch=8,
        name="improved",
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=5.0, translate=0.1, scale=0.3, shear=2.0,
        mosaic=0.5, mixup=0.1,
        patience=30,
    ),
    "grouped": dict(
        model="yolov8s.pt",
        epochs=100,
        imgsz=800,
        batch=8,
        name="grouped",
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=5.0, translate=0.1, scale=0.4,
        mosaic=1.0, mixup=0.1,
        patience=30,
        freeze=10,          # freeze first 10 backbone layers
        data=DATA_YAML_GROUPED,  # overrides default DATA_YAML
    ),
}


def train(config_name: str):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Choose from {list(CONFIGS)}")

    cfg = CONFIGS[config_name].copy()
    model_name = cfg.pop("model")

    # allow config to override the default data yaml
    data_path = cfg.pop("data", DATA_YAML)

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"{data_path} not found.\n"
            f"  For baseline/improved: run `python data/download_data.py` first.\n"
            f"  For grouped: run the label remapping cell in Colab first."
        )

    print(f"Training config : {config_name}")
    print(f"Base model      : {model_name}")
    print(f"Data yaml       : {data_path}")
    print(f"Epochs          : {cfg.get('epochs')}")
    print(f"Image size      : {cfg.get('imgsz')}")
    print("-" * 40)

    model = YOLO(model_name)
    results = model.train(data=data_path, **cfg)

    weights = f"runs/detect/{cfg['name']}/weights/best.pt"
    print(f"\nTraining complete. Best weights: {weights}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on traffic sign data")
    parser.add_argument(
        "--config", type=str, default="baseline",
        choices=list(CONFIGS),
        help="Training config to use"
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
