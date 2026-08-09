# Traffic Sign Detector — Model Improvement & Error Analysis Tool

A traffic sign detection pipeline built with YOLOv8, focused on finding *where the model fails* and
prioritizing those failures for review — the same workflow used in real ADAS validation
(confidence-based triage, hard-example mining, KPI tracking).

## Problem Statement
Off-the-shelf object detectors perform unevenly across traffic sign classes due to class
imbalance, occlusion, weather, and scale variation. This project:
1. Fine-tunes YOLOv8 on a traffic sign dataset
2. Builds an error analysis module that buckets failures (false negatives, false positives,
   low-confidence-but-correct)
3. Implements an active-learning-style ranking of which unlabeled images would most improve
   the model if reviewed next
4. Wraps everything in a Streamlit dashboard for interactive review

## Dataset
Using [GTSDB](https://benchmark.ini.rub.de/gtsdb.html) (German Traffic Sign Detection Benchmark)
— bounding-box-annotated traffic signs, converted to YOLO format. 

## Tech Stack
`ultralytics` (YOLOv8) · `opencv-python` · `albumentations` · `streamlit` · `scikit-learn` · `matplotlib`


