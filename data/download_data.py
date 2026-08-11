"""
Download and prepare the GTSDB dataset from Kaggle, converting it into
the YOLO folder structure expected by src/train.py.

Kaggle dataset: icebearogo/german-traffic-sign-detection-gtsdb-dataset
  - 900 images total (600 train, 300 test), already in .jpg format
  - Labels already in YOLO format (.txt, one file per image)
  - Images with no traffic signs have no label file (YOLO handles this fine)

What this script does:
  1. Verifies kaggle.json credentials are in place
  2. Downloads the dataset zip via the Kaggle API
  3. Extracts and inspects the folder structure
  4. Reorganises everything into:
       data/gtsdb_yolo/
         images/{train,val,test}/
         labels/{train,val,test}/
         data.yaml
  5. Splits the 600 training images into train (80%) and val (20%)
     (the original 300 test images become our held-out test split)

Usage (local):
    python data/download_data.py

Usage (Colab) — run the Colab setup cells first, then:
    !python data/download_data.py

Requirements:
    pip install kaggle
"""

import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────
KAGGLE_DATASET   = "icebearogo/german-traffic-sign-detection-gtsdb-dataset"
RAW_ZIP_NAME     = "german-traffic-sign-detection-gtsdb-dataset.zip"
RAW_DIR          = Path("data/raw")
OUTPUT_DIR       = Path("data/gtsdb_yolo")
RANDOM_SEED      = 42
VAL_SPLIT        = 0.20   # fraction of training images used for validation

# 43 GTSDB classes (index = class ID used in label files)
CLASS_NAMES = [
    "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60",
    "speed_limit_70", "speed_limit_80", "end_speed_limit_80", "speed_limit_100",
    "speed_limit_120", "no_overtaking", "no_overtaking_trucks", "priority_road_next",
    "priority_road", "give_way", "stop", "no_vehicles", "no_trucks", "no_entry",
    "danger", "left_curve", "right_curve", "double_curve", "bumpy_road",
    "slippery_road", "road_narrows", "roadworks", "traffic_signals", "pedestrians",
    "children", "cyclists", "ice_snow", "wild_animals", "end_restrictions",
    "turn_right", "turn_left", "ahead_only", "straight_or_right", "straight_or_left",
    "keep_right", "keep_left", "roundabout", "end_no_overtaking",
    "end_no_overtaking_trucks",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def check_kaggle_credentials():
    """
    Support all Kaggle auth methods in priority order:
      1. KAGGLE_API_TOKEN env var (new single-token format → access_token file)
      2. KAGGLE_USERNAME + KAGGLE_KEY env vars
      3. ~/.kaggle/kaggle.json file (classic format)
      4. ~/.kaggle/access_token file (new format, already on disk)
    """
    import json, pathlib, sys

    kaggle_dir   = pathlib.Path.home() / ".kaggle"
    kaggle_json  = kaggle_dir / "kaggle.json"
    access_token = kaggle_dir / "access_token"

    # method 1 — new KAGGLE_API_TOKEN env var → write access_token file
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    if api_token:
        kaggle_dir.mkdir(exist_ok=True)
        access_token.write_text(api_token)
        access_token.chmod(0o600)
        print("✓ Kaggle credentials set from KAGGLE_API_TOKEN")
        return

    # method 2 — KAGGLE_USERNAME + KAGGLE_KEY env vars → write kaggle.json
    username = os.environ.get("KAGGLE_USERNAME", "")
    key      = os.environ.get("KAGGLE_KEY", "")
    if username and key:
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        kaggle_json.chmod(0o600)
        print(f"✓ kaggle.json written from env vars (user: {username})")
        return

    # method 3 — kaggle.json already on disk
    if kaggle_json.exists():
        kaggle_json.chmod(0o600)
        print("✓ kaggle.json found on disk")
        return

    # method 4 — access_token already on disk (new format)
    if access_token.exists():
        access_token.chmod(0o600)
        print("✓ access_token found on disk")
        return

    print("""
ERROR: No Kaggle credentials found. Run this in Colab before the script:

    import os
    from google.colab import userdata
    os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
""")
    sys.exit(1)


def download_from_kaggle():
    """Download dataset zip using the Kaggle CLI."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / RAW_ZIP_NAME

    if zip_path.exists():
        print(f"✓ Dataset zip already present at {zip_path}, skipping download")
        return zip_path

    print(f"Downloading {KAGGLE_DATASET} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Kaggle download failed:\n", result.stderr)
        sys.exit(1)

    print(f"✓ Downloaded to {zip_path}")
    return zip_path


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract the dataset zip and return the root folder inside it."""
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # Show what was extracted so we can adapt if the structure ever changes
    top_level = sorted({str(Path(n).parts[0]) for n in zf.namelist()})
    print(f"  Extracted top-level entries: {top_level}")
    return extract_to


def find_image_label_pairs(img_dir: Path, lbl_dir: Path):
    """
    Given separate image and label directories, pair each jpg with its
    corresponding txt file. Images with no label get an empty txt file
    (valid for YOLO — means image has no signs).
    """
    pairs = []
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / img.with_suffix(".txt").name
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs



def build_yolo_structure(extract_root: Path):
    """
    Handles the exact Kaggle GTSDB structure:
        GTSDB_Train_and_Test/
            Train/
                images/
                labels/
            Test/
                images/
                labels/
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # locate Train and Test folders
    train_img_dir = extract_root / "GTSDB_Train_and_Test" / "Train" / "images"
    train_lbl_dir = extract_root / "GTSDB_Train_and_Test" / "Train" / "labels"
    test_img_dir  = extract_root / "GTSDB_Train_and_Test" / "Test"  / "images"
    test_lbl_dir  = extract_root / "GTSDB_Train_and_Test" / "Test"  / "labels"

    # fallback — search recursively if exact path doesn't match
    if not train_img_dir.exists():
        candidates = [p for p in extract_root.rglob("images") if p.is_dir()
                      and "rain" in str(p.parent).lower()]
        if candidates:
            train_img_dir = candidates[0]
            train_lbl_dir = train_img_dir.parent / "labels"
        else:
            raise FileNotFoundError(
                f"Could not find Train/images inside {extract_root}. "
                f"Run: !find data/raw/extracted -type d  to inspect."
            )

    if not test_img_dir.exists():
        candidates = [p for p in extract_root.rglob("images") if p.is_dir()
                      and "est" in str(p.parent).lower()]
        test_img_dir = candidates[0] if candidates else None
        test_lbl_dir = test_img_dir.parent / "labels" if test_img_dir else None

    # pair images with labels
    all_train_pairs = find_image_label_pairs(train_img_dir, train_lbl_dir)
    all_test_pairs  = find_image_label_pairs(test_img_dir, test_lbl_dir) if test_img_dir else []

    if not all_train_pairs:
        raise FileNotFoundError(f"No .jpg files found in {train_img_dir}")

    # split train → 80% train / 20% val
    random.seed(RANDOM_SEED)
    random.shuffle(all_train_pairs)
    n_val       = max(1, int(len(all_train_pairs) * VAL_SPLIT))
    val_pairs   = all_train_pairs[:n_val]
    train_pairs = all_train_pairs[n_val:]

    splits = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  all_test_pairs,
    }

    # copy into output structure
    for split, pairs in splits.items():
        img_out = OUTPUT_DIR / "images" / split
        lbl_out = OUTPUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        annotated, empty = 0, 0
        for img_path, label_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            dest_lbl = lbl_out / img_path.with_suffix(".txt").name
            if label_path and label_path.exists() and label_path.stat().st_size > 0:
                shutil.copy2(label_path, dest_lbl)
                annotated += 1
            else:
                dest_lbl.touch()  # empty = image with no signs, valid for YOLO
                empty += 1

        print(f"  {split:5s} → {annotated + empty} images "
              f"({annotated} annotated, {empty} no-sign images)")

    write_data_yaml()


def write_data_yaml():
    yaml_lines = [
        f"path: {OUTPUT_DIR.resolve()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {name}")

    out = OUTPUT_DIR / "data.yaml"
    out.write_text("\n".join(yaml_lines) + "\n")
    print(f"✓ Wrote {out}")


def verify_output():
    """Quick sanity check — count files in each split."""
    print("\nVerification:")
    for split in ("train", "val", "test"):
        imgs   = len(list((OUTPUT_DIR / "images" / split).glob("*.jpg")))
        labels = len(list((OUTPUT_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split:5s} → {imgs} images, {labels} label files")
    print(f"\n✓ Dataset ready at {OUTPUT_DIR}/")
    print("  Next step: python src/train.py --config baseline")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" GTSDB Dataset Setup (Kaggle source)")
    print("=" * 55)

    check_kaggle_credentials()
    zip_path    = download_from_kaggle()
    extract_dir = RAW_DIR / "extracted"
    extract_zip(zip_path, extract_dir)
    build_yolo_structure(extract_dir)
    verify_output()


if __name__ == "__main__":
    main()
