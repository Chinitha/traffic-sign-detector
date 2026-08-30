# Traffic Sign Detector — Model Improvement & Error Analysis Tool

A traffic sign detection pipeline built with YOLOv8, focused on diagnosing *why* a model
fails and prioritizing those failures for review — mirroring the confidence-based triage
and validation workflows used in real ADAS perception systems.

## Results

| Run | mAP50 | mAP50-95 | Precision | Recall | Notes |
|---|---|---|---|---|---|
| Baseline | 0.025 | — | 0.017 | 0.245 | YOLOv8n, 43 classes, no augmentation |
| Improved | 0.038 | — | 0.249 | 0.041 | YOLOv8s, 43 classes, augmentation |
| **Grouped** | **0.515** | **0.413** | **0.596** | **0.571** | YOLOv8s, 2 superclasses, frozen backbone |

**20× improvement in mAP50** achieved by diagnosing and correcting a class-imbalance
problem rather than tuning hyperparameters.

### Error Analysis (val set, 120 images, confidence threshold 0.5)
| Metric | Count | Per image |
|---|---|---|
| False Positives | 57 | 0.47 |
| False Negatives | 38 | 0.31 |
| Low-confidence correct | 13 | — |

## Problem Statement

Off-the-shelf object detectors perform unevenly across traffic sign classes due to class
imbalance, occlusion, weather, and scale variation. This project:
1. Fine-tunes YOLOv8 on the GTSDB dataset
2. Diagnoses *why* the model underperforms through EDA and error analysis
3. Corrects the root cause (class imbalance) rather than masking it with more augmentation
4. Implements active-learning-style ranking of uncertain images
5. Wraps everything in a Streamlit dashboard for interactive review

## The Diagnostic Journey (why the numbers look the way they do)

The most important part of this project wasn't running `model.train()` — it was figuring
out why the first two attempts failed and fixing the actual cause.

**Attempt 1 — Baseline (mAP50 = 0.025).** YOLOv8n trained on all 43 GTSDB classes with no
augmentation. The dataset only contains 405 total annotated boxes. Divided across 38–43
classes that's **~10 boxes per class** — far below the 100–300 examples per class YOLO
needs to learn reliably. Symptom: the model over-detected everything (recall 0.245,
precision 0.017) because it couldn't distinguish between sign types.

**Attempt 2 — Improved (mAP50 = 0.038).** Added augmentation (HSV shifts, rotation, mosaic)
on the same 43-class, 10-boxes-per-class dataset. Augmentation cannot fix a data scarcity
problem — it made the model *more conservative* instead, collapsing recall to 0.041 while
barely moving precision. This confirmed the bottleneck was data volume per class, not
augmentation strategy.

**Attempt 3 — Grouped (mAP50 = 0.515).** Consolidated the 43 original classes into 2
semantic superclasses:
- `regulatory` (speed limits, overtaking restrictions, priority signs — original classes 0–15)
- `warning` (danger, hazard, and all other signs — original classes 16–42)

This raised examples per class from ~10 to ~200 — inside YOLO's reliable training range.
Combined with freezing the first 10 backbone layers (necessary with only 480 training
images to prevent overfitting the pretrained ImageNet features), mAP50 rose to 0.515 —
a 20× improvement using the *same* underlying dataset and images.

**What this trades off.** The grouped model detects sign *category* (regulatory vs.
warning) but not sign *value* (e.g. "speed limit 50" vs. "speed limit 80") — that
distinction was merged away along with the classes that caused the data scarcity. Getting
value-level detection back would require either substantially more annotated data per
original class, or a two-stage pipeline: this detector for localization, followed by a
separate classifier trained on cropped sign regions. 

## Repo Structure
```
traffic-sign-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_data.py       # Kaggle download + YOLO format conversion
│   └── prepare_grouped.py     # 43 → 2 superclass remapping
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   └── 01_eda_executed.ipynb   # same notebook, pre-run with plots visible on GitHub
├── src/
│   ├── train.py                # baseline / improved / grouped training configs
│   ├── evaluate.py             # mAP/precision/recall + per-class plots
│   ├── error_analysis.py       # FP/FN bucketing via IoU matching
│   └── active_learning.py      # entropy-based uncertainty ranking
├── app/
│   └── streamlit_app.py        # review dashboard (3 tabs)
└── results/
    ├── metrics.json
    ├── error_analysis.csv
    ├── active_learning_ranking.csv
    └── plots/
        └── per_class_ap_grouped.png
```


## Setup

```bash
python -m venv venv
source venv/bin/activate          # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download and prepare the dataset (see Dataset section for Kaggle credentials setup)
python data/download_data.py

# 2. Consolidate into 2 superclasses (fixes the class-imbalance problem described above)
python data/prepare_grouped.py

# 3. Train
python src/train.py --config baseline    # reference score
python src/train.py --config grouped     # the actual working model

# 4. Evaluate
python src/evaluate.py \
    --weights runs/detect/grouped/weights/best.pt \
    --name grouped \
    --data data/gtsdb_yolo_grouped/data.yaml

# 5. Error analysis
python src/error_analysis.py \
    --weights runs/detect/grouped/weights/best.pt \
    --data-dir data/gtsdb_yolo_grouped \
    --split val \
    --conf-threshold 0.5

# 6. Active learning ranking on unlabeled images
python src/active_learning.py \
    --weights runs/detect/grouped/weights/best.pt \
    --unlabeled-dir data/gtsdb_yolo_grouped/images/test

# 7. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Dataset

[GTSDB](https://benchmark.ini.rub.de/gtsdb.html) (German Traffic Sign Detection Benchmark)
via the [Kaggle mirror](https://www.kaggle.com/datasets/icebearogo/german-traffic-sign-detection-gtsdb-dataset)
— 900 images (600 train, 300 test), pre-converted to YOLO label format.

Requires a free Kaggle account and API token:
```bash
# option 1 — kaggle.json (classic)
# Kaggle → Settings → API → Create New Token → places kaggle.json at ~/.kaggle/

# option 2 — new single-token format
export KAGGLE_API_TOKEN=your_token_here
```
`download_data.py` supports both, plus `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars.

## Tech Stack
`ultralytics` (YOLOv8) · `opencv-python` · `albumentations` · `streamlit` · `scikit-learn`
· `matplotlib` · `pyngrok` (for tunneling the dashboard from Colab)

## Future Improvement — Sign Value Detection

The current model detects sign **category** (`regulatory` vs. `warning`) but not sign
**value** — e.g. it can locate a speed limit sign but not read whether it says 50 or 80.
This was a deliberate tradeoff made when consolidating 43 classes into 2 (see
[Diagnostic Journey](#the-diagnostic-journey-why-the-numbers-look-the-way-they-do) above)
to solve the class-imbalance problem that was blocking training entirely.

Planned approach to recover value-level detection without reintroducing the data-scarcity
problem:

- **Two-stage pipeline**: keep the current YOLOv8 grouped model for sign *localization*,
  and add a separate lightweight classifier trained only on cropped sign regions for sign
  *value*. Decoupling localization from fine-grained classification means the classifier
  only needs to solve a simpler, more sample-efficient problem than full-image detection —
  the same architectural pattern used in production ADAS perception stacks.
- **More annotated data**: supplement GTSDB with additional public datasets (e.g. Mapillary
  Traffic Sign Dataset, GTSRB) to raise per-class example counts for the original 43
  classes above the ~150–300 boxes needed for reliable detection, removing the need to
  merge classes at all.
- **OCR as a narrower interim step**: for speed limit signs specifically, the printed
  number could be read directly off the cropped detection using OCR, without requiring a
  trained classifier — a faster but narrower win limited to numeric signs only.

## What I'd Do With More Time
- Collect or source additional annotated data for the original 43 classes to restore
  sign-value detection (e.g. supplement GTSDB with Mapillary Traffic Sign Dataset)
- Build the two-stage detector → classifier pipeline described above instead of
  permanently merging classes
- Calibrate confidence scores — current YOLO confidences aren't well-calibrated probabilities
- Add temporal consistency checks across video frames, relevant for real ADAS pipelines
- Deploy permanently via Streamlit Community Cloud with weights hosted on Hugging Face,
  rather than tunneling a Colab session

## Background
This project extends real-world experience validating ADAS perception systems — including
traffic sign detection model evaluation and annotation tooling — during an internship at
MAN Truck & Bus SE.
