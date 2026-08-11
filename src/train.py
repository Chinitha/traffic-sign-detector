"""
Train YOLOv8 on the traffic sign dataset.

Two configs are provided:
  - baseline: small model, minimal augmentation -- establishes a reference score
  - improved: adds augmentation tuned for traffic-sign conditions (motion blur,
    brightness/contrast shifts to simulate weather/lighting, scale jitter for
    distant/close signs) and trains longer

Usage:
    python src/train.py --config baseline
    python src/train.py --config improved
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "data/gtsdb_yolo/data.yaml"

CONFIGS = {
    "baseline": dict(
        model="yolov8n.pt",
        epochs=50,
        imgsz=640,
        batch=16,
        name="baseline",
        augment=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        mosaic=0.0, mixup=0.0,
    ),
    "improved": dict(
        model="yolov8s.pt",
        epochs=100,
        imgsz=640,
        batch=16,
        name="improved",
        augment=True,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,   # lighting/weather variation
        degrees=5.0, translate=0.1, scale=0.3, shear=2.0,  # sign angle/distance variation
        mosaic=0.5, mixup=0.1,
        patience=20,  # early stopping
    ),
}


def train(config_name: str):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Choose from {list(CONFIGS)}")

    cfg = CONFIGS[config_name].copy()
    model_name = cfg.pop("model")
    cfg.pop("augment", None)  # informational only, not a real ultralytics arg

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Run `python data/download_data.py` first."
        )

    print(f"Training config: {config_name} (base model: {model_name})")
    model = YOLO(model_name)
    results = model.train(data=DATA_YAML, **cfg)

    print("\nTraining complete.")
    print(f"Best weights: runs/detect/{cfg['name']}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on traffic sign data")
    parser.add_argument(
        "--config", type=str, default="baseline", choices=list(CONFIGS), help="Training config to use"
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
