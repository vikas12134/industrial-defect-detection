"""
Streamlit Web Application for Industrial Surface Defect Detection.

A clean, portfolio-ready Deep Learning computer vision application using YOLOv8
to detect, localize, and classify surface defects on industrial steel components.
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image
import streamlit as st

# Setup project root path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import DefectPredictor

# Defect class descriptions for user reference
DEFECT_DESCRIPTIONS = {
    "crazing": "Network of fine surface cracks and micro-fissures",
    "inclusion": "Non-metallic embedded particles or impurities",
    "patches": "Localized surface irregularities and discoloration",
    "pitted_surface": "Small surface cavities or erosion indentations",
    "rolled-in_scale": "Oxidized mill scale pressed into the sheet metal",
    "scratches": "Linear abrasive scratches or mechanical grooves",
}

# --- Page Setup ---
st.set_page_config(
    page_title="Industrial Defect Detection AI",
    page_icon="🔍",
    layout="wide",
)


DEFAULT_MODEL_PATH = ROOT_DIR / "outputs" / "weights" / "best.pt"


@st.cache_resource(show_spinner="Loading trained YOLOv8 model weights...")
def load_model(weights_path: Optional[str] = None) -> Optional[DefectPredictor]:
    """Load and cache the trained YOLO defect detection model."""
    target_path = weights_path if weights_path else str(DEFAULT_MODEL_PATH)
    try:
        predictor = DefectPredictor(model_path=target_path, device="cpu")
        return predictor
    except Exception as e:
        st.error(f"Error loading model weights from {target_path}: {e}")
        return None


def get_sample_images() -> List[Path]:
    """Retrieve sample validation images for quick demonstration."""
    val_dir = ROOT_DIR / "data" / "yolo" / "images" / "val"
    if val_dir.exists():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in val_dir.iterdir() if p.suffix.lower() in valid_exts])
    return []


def main():
    st.title("🔍 Industrial Surface Defect Detection")
    st.write(
        "Upload an industrial surface inspection image to detect, localize, and classify "
        "defects in real-time using a custom-trained **YOLOv8** model."
    )

    # --- Sidebar Controls ---
    st.sidebar.header("Inspection Controls")

    # Model Loading
    predictor = load_model(str(DEFAULT_MODEL_PATH))
    if predictor:
        st.sidebar.success("Model loaded: `best.pt`")
    else:
        st.sidebar.error("Model failed to load. Please verify `outputs/weights/best.pt` exists.")
        return

    # Detection Confidence Threshold Slider
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required for defect detection.",
    )

    # Sample images for easy testing and demonstration
    st.sidebar.subheader("Quick Test Samples")
    sample_images = get_sample_images()
    selected_sample = None
    if sample_images:
        sample_options = ["-- None (Upload your own) --"] + [p.name for p in sample_images]
        chosen = st.sidebar.selectbox("Choose a sample validation image:", sample_options)
        if chosen != "-- None (Upload your own) --":
            selected_sample = next((p for p in sample_images if p.name == chosen), None)

    # Defect Class Catalog in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Defect Classes")
    for name, desc in DEFECT_DESCRIPTIONS.items():
        st.sidebar.markdown(f"- **`{name}`**: {desc}")

    # --- Main Image Upload Area ---
    uploaded_file = st.file_uploader(
        "Upload a component surface image (.jpg, .png, .bmp, .webp)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    # Determine input image
    input_image = None
    image_label = ""
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_label = uploaded_file.name
    elif selected_sample is not None:
        input_image = Image.open(selected_sample).convert("RGB")
        image_label = selected_sample.name

    if input_image is not None:
        st.markdown("---")

        # Run inference using the trained YOLO model
        with st.spinner("Running YOLO defect detection..."):
            result = predictor.predict_image(
                image_input=input_image,
                conf=conf_threshold,
                save_annotated=False,
                imgsz=224,
            )

        detections = result.get("detections", [])
        annotated_image = result.get("annotated_image")

        # 1. Defect Summary Banner
        if detections:
            detected_classes = sorted(list(set(d["class_name"] for d in detections)))
            st.error(
                f"⚠️ **Defects Detected!** Found **{len(detections)} defect instance(s)** "
                f"belonging to class: **{', '.join(detected_classes)}**"
            )
        else:
            st.success("✅ **No defect detected** (Component surface passed inspection)")

        # 2. Side-by-Side Image Display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            st.image(input_image, caption=f"Original: {image_label}", use_container_width=True)

        with col2:
            st.subheader("YOLO Detection Result")
            if annotated_image is not None:
                st.image(
                    annotated_image,
                    caption=f"Predictions: {len(detections)} defect(s) localized with bounding boxes",
                    use_container_width=True,
                )
            else:
                st.image(input_image, caption="No detections to display", use_container_width=True)

        # 3. Detection Details Table & Metrics
        st.markdown("### Detection Details")
        if detections:
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Defect Instances", len(detections))
            with metric_col2:
                st.metric("Unique Defect Classes", len(set(d["class_name"] for d in detections)))
            with metric_col3:
                max_conf = max(d["confidence"] for d in detections)
                st.metric("Highest Confidence", f"{max_conf * 100:.1f}%")

            # Formatted Data Table displaying class, confidence, and bounding box
            table_data = []
            for idx, d in enumerate(detections, start=1):
                table_data.append({
                    "Defect #": idx,
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                    "Confidence Score": round(d["confidence"], 4),
                    "Bounding Box [xmin, ymin, xmax, ymax]": str(d["bbox_xyxy"]),
                    "Width (px)": d.get("width", round(d["bbox_xyxy"][2] - d["bbox_xyxy"][0], 1)),
                    "Height (px)": d.get("height", round(d["bbox_xyxy"][3] - d["bbox_xyxy"][1], 1)),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.info("No defect detected. All quality inspection criteria satisfied at the current confidence threshold.")

    else:
        st.info("👆 Please upload an image above or select a sample image from the sidebar to inspect.")


if __name__ == "__main__":
    main()
