"""
Streamlit dashboard for reviewing traffic sign detections.

Three views:
  1. Live inference -- upload an image, see detections + confidence
  2. Error browser -- step through flagged false positives/negatives
  3. Review queue  -- step through active-learning ranked list

Usage:
    streamlit run app/streamlit_app.py
"""

import csv
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

DATA_DIR     = Path("data/gtsdb_yolo_grouped")   # updated to grouped dataset
RESULTS_DIR  = Path("results")
WEIGHTS_DEFAULT = "runs/detect/grouped/weights/best.pt"  # updated to grouped model

st.set_page_config(page_title="Traffic Sign Detector — Review Tool", layout="wide")


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def run_inference_view(model):
    st.header("Live Inference")
    uploaded      = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        model.overrides["conf"] = conf_threshold
        results = model.predict(img, verbose=False)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            annotated = results.plot()
            st.image(annotated, caption="Detections",
                     use_container_width=True, channels="BGR")

        if len(results.boxes) == 0:
            st.warning("No detections above the confidence threshold.")
        else:
            st.subheader("Detections")
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf     = float(box.conf[0])
                name     = model.names[class_id]
                flag     = "⚠️ low confidence" if conf < 0.5 else ""
                st.write(f"- **{name}** — confidence: {conf:.2f} {flag}")


def error_browser_view():
    st.header("Error Browser")
    csv_path = RESULTS_DIR / "error_analysis.csv"
    if not csv_path.exists():
        st.info("Run `python src/error_analysis.py` first to generate error_analysis.csv.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    rows.sort(
        key=lambda r: int(r["false_positives"]) + int(r["false_negatives"]),
        reverse=True
    )

    st.write(f"Showing {len(rows)} analyzed images, worst errors first.")
    sel = st.selectbox("Select image", [r["image"] for r in rows])
    row = next(r for r in rows if r["image"] == sel)

    img_path = DATA_DIR / "images" / "val" / sel
    if img_path.exists():
        st.image(str(img_path), caption=sel, use_container_width=True)
    else:
        st.warning(f"Image not found at {img_path}")

    c1, c2, c3 = st.columns(3)
    c1.metric("False Positives",       row["false_positives"])
    c2.metric("False Negatives",       row["false_negatives"])
    c3.metric("Low-confidence correct", row["low_confidence_correct"])


def review_queue_view():
    st.header("Active Learning Review Queue")
    csv_path = RESULTS_DIR / "active_learning_ranking.csv"
    if not csv_path.exists():
        st.info("Run `python src/active_learning.py` first to generate active_learning_ranking.csv.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    st.write(f"{len(rows)} unlabeled images ranked by uncertainty (most uncertain first).")
    st.dataframe(rows[:50], use_container_width=True)


def main():
    st.title("🚦 Traffic Sign Detector — Review Tool")
    st.caption(
        "Model improvement & error-analysis dashboard built on YOLOv8. "
        "Mirrors confidence-based triage workflows used in ADAS perception validation."
    )

    weights_path = st.sidebar.text_input("Model weights path", value=WEIGHTS_DEFAULT)
    view = st.sidebar.radio("View", ["Live Inference", "Error Browser", "Review Queue"])

    if view == "Live Inference":
        try:
            model = load_model(weights_path)
            run_inference_view(model)
        except Exception as e:
            st.error(f"Could not load model from '{weights_path}': {e}")
    elif view == "Error Browser":
        error_browser_view()
    else:
        review_queue_view()


if __name__ == "__main__":
    main()