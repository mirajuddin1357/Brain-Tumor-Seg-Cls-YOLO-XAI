"""
Brain Tumor MRI Segmentation and Classification

YOLO-Based Deep Learning with Grad-CAM Explainability

Research prototype Streamlit application for the IEEE paper:
"Automated Brain Tumor Segmentation and Classification in MRI Using
YOLO-Based Deep Learning with Grad-CAM Explainability and Streamlit Deployment"

Run with: streamlit run app.py
"""

import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.inference import BrainTumorSegmenter
from src.visualization import (
    draw_segmentation_overlay,
    draw_detection_result,
    draw_full_result,
)
from src.gradcam import GradCAMEngine
from src.utils import (
    pil_to_cv2,
    cv2_to_rgb,
    image_to_bytes,
    get_missing_model_message,
)


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

MODEL_OPTIONS = {
    "YOLOv11x-Seg": "YOLOv11x_Seg_best.onnx",
    "YOLOv26x-Seg": "YOLOv26x_Seg_best.onnx",
}

IMAGE_SIZE_OPTIONS = {
    "256x256": (256, 256),
}

DEMO_IMAGE_PATH = ASSETS_DIR / "demo_image.png"


# ─── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brain Tumor MRI Segmentation and Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, scientific CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Force Pure White Background across entire app */
    .stApp {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #1E293B !important;
    }

    /* Top App Header */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    .main-header {
        text-align: center;
        padding: 1.25rem 0 0.75rem 0;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 1.5rem;
        background-color: #FFFFFF;
    }
    .main-header h1 {
        color: #0F172A;
        font-weight: 700;
        font-size: 1.85rem;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #64748B;
        font-size: 0.95rem;
        margin-top: 0;
    }

    .section-header {
        color: #0F172A;
        font-weight: 600;
        font-size: 1.15rem;
        margin-top: 1.5rem;
        margin-bottom: 0.85rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #E2E8F0;
    }

    /* Clean Summary Box */
    .summary-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin: 0.85rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        font-size: 0.9rem;
        border-bottom: 1px dashed #F1F5F9;
    }
    .summary-row:last-child {
        border-bottom: none;
    }
    .summary-label {
        color: #64748B;
        font-weight: 500;
    }
    .summary-value {
        color: #0F172A;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid #E2E8F0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Cached Model Loader ─────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path: str) -> BrainTumorSegmenter:
    """Load and cache an ONNX model.

    Args:
        model_path: Absolute path to the ONNX model file.

    Returns:
        BrainTumorSegmenter instance.
    """
    return BrainTumorSegmenter(model_path)


@st.cache_resource
def get_gradcam_engine() -> GradCAMEngine:
    """Get cached Grad-CAM engine instance."""
    return GradCAMEngine()


# ─── Main Application ────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Brain Tumor MRI Segmentation and Classification</h1>
        <p>YOLO-Based Deep Learning with Grad-CAM Explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Validate model files ---
    missing_msg = get_missing_model_message(str(MODELS_DIR), MODEL_OPTIONS)
    if missing_msg:
        st.error(missing_msg)
        st.stop()

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Inference Settings")

        # Model selection
        selected_model_name = st.selectbox(
            "Model",
            list(MODEL_OPTIONS.keys()),
            index=0,
        )

        st.markdown("---")

        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.25,
            step=0.05,
        )

        # IoU threshold
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
        )

        # Image size (default 256x256 matching model training)
        image_size_str = st.selectbox(
            "Image Size",
            list(IMAGE_SIZE_OPTIONS.keys()),
            index=0,
            help="Model input resolution (trained at 256×256).",
        )
        input_size = IMAGE_SIZE_OPTIONS[image_size_str]

        st.markdown("---")

        # Grad-CAM toggle
        enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)

    # ─── MRI Input ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">MRI Image Input</div>',
                unsafe_allow_html=True)

    input_col1, input_col2 = st.columns([3, 1])

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload MRI Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a brain MRI image in PNG, JPG, or JPEG format.",
        )

    with input_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("📷 Use Demo Image", use_container_width=True)

    # Determine input image
    img_bgr = None
    image_source = None

    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
            img_bgr = pil_to_cv2(pil_img)
            image_source = "Uploaded MRI"
        except Exception as e:
            st.error(f"Failed to load the uploaded image: {e}")
            st.stop()

    elif use_demo:
        demo_path = str(DEMO_IMAGE_PATH)
        if os.path.isfile(demo_path):
            img_bgr = cv2.imread(demo_path)
            if img_bgr is None:
                st.error(
                    "Failed to read the demo image. Please ensure "
                    "`assets/demo_image.png` is a valid image file."
                )
                st.stop()
            image_source = "Demo MRI"
        else:
            st.warning(
                "Demo image not found at `assets/demo_image.png`. "
                "Please place a brain tumor MRI image there."
            )
            st.stop()

    # Store in session state for persistence
    if img_bgr is not None:
        st.session_state["img_bgr"] = img_bgr
        st.session_state["image_source"] = image_source
    elif "img_bgr" in st.session_state:
        img_bgr = st.session_state["img_bgr"]
        image_source = st.session_state.get("image_source", "MRI")

    if img_bgr is None:
        st.info(
            "Upload a brain MRI image or click **Use Demo Image** to begin."
        )
        return

    img_rgb = cv2_to_rgb(img_bgr)

    # ─── Run Inference Button ─────────────────────────────────────────────
    run_inference = st.button(
        "🔬 Run Inference", type="primary", use_container_width=True
    )

    if not run_inference and "result" not in st.session_state:
        # Show original MRI before inference
        st.markdown('<div class="section-header">Original MRI</div>',
                    unsafe_allow_html=True)
        preview_col, _ = st.columns([1, 2])
        with preview_col:
            st.image(img_rgb, caption=image_source, use_container_width=True)
        return

    # ─── Inference ────────────────────────────────────────────────────────
    if run_inference:
        model_filename = MODEL_OPTIONS[selected_model_name]
        model_path = str(MODELS_DIR / model_filename)

        try:
            with st.spinner(f"Loading {selected_model_name}..."):
                segmenter = load_model(model_path)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        try:
            with st.spinner(f"Running {selected_model_name} inference..."):
                result = segmenter.predict(
                    img_bgr,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    input_size=input_size,
                )
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

        # Grad-CAM Explainability
        gradcam_result = None
        if enable_gradcam:
            with st.spinner("Computing Grad-CAM Saliency..."):
                engine = get_gradcam_engine()
                gradcam_result = engine.explain(
                    image_rgb=img_rgb,
                    segmentation_result=result,
                    class_names=segmenter.class_names,
                )

        # Store results in session state
        st.session_state["result"] = result
        st.session_state["gradcam_result"] = gradcam_result
        st.session_state["class_names"] = segmenter.class_names
        st.session_state["selected_model"] = selected_model_name
        st.session_state["enable_gradcam"] = enable_gradcam

    # ─── Retrieve results from session state ──────────────────────────────
    result = st.session_state.get("result")
    gradcam_result = st.session_state.get("gradcam_result")
    class_names = st.session_state.get("class_names", [])
    selected_model = st.session_state.get("selected_model", "")
    gradcam_enabled = st.session_state.get("enable_gradcam", False)

    if result is None:
        return

    # ─── Handle no detections ─────────────────────────────────────────────
    if result.n_detections == 0:
        st.warning(
            "No tumor regions detected at the current confidence threshold. "
            "Try lowering the confidence threshold in the sidebar."
        )

        st.markdown('<div class="section-header">Original MRI</div>',
                    unsafe_allow_html=True)
        orig_col, _ = st.columns([1, 2])
        with orig_col:
            st.image(img_rgb, caption=image_source, use_container_width=True)

        # Still show prediction summary
        st.markdown('<div class="section-header">Prediction Summary</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-box">
            <div class="summary-row">
                <span class="summary-label">Selected Model</span>
                <span class="summary-value">{selected_model}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Predicted Class</span>
                <span class="summary-value">—</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Confidence</span>
                <span class="summary-value">—</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Detected Regions</span>
                <span class="summary-value">0</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Inference Time</span>
                <span class="summary-value">{result.inference_time_ms:.1f} ms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Extract top prediction
    top_class_id = int(result.class_ids[0])
    top_class_name = (
        class_names[top_class_id]
        if top_class_id < len(class_names)
        else f"Class_{top_class_id}"
    )
    top_confidence = float(result.scores[0]) * 100

    # ─── Visual Diagnostic Results: 3 Panels on One Row ──────────────────
    st.markdown('<div class="section-header">Visual Diagnostic Results</div>',
                unsafe_allow_html=True)

    # All-in-one Tumor Segmentation + Detection and Classification image
    combined_image = draw_full_result(
        img_rgb, result.boxes, result.scores, result.class_ids, result.masks, class_names
    )

    # Third image: Grad-CAM Saliency
    has_gradcam = gradcam_result is not None and gradcam_result.is_valid
    if has_gradcam:
        third_image = gradcam_result.overlay
        third_title = "Grad-CAM Saliency"
        third_caption = f"(c) Grad-CAM Saliency — {gradcam_result.target_class_name}"
    else:
        third_image = img_rgb.copy()
        third_title = "Grad-CAM Saliency"
        third_caption = f"(c) Grad-CAM Saliency — {top_class_name}"

    # Render all three images side-by-side in one row (compact size)
    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.markdown("**Original MRI**")
        st.image(img_rgb, caption="(a) Input MRI Image", use_container_width=True)

    with img_col2:
        st.markdown("**Tumor Segmentation & Classification**")
        st.image(
            combined_image,
            caption=f"(b) Predicted: {top_class_name} ({top_confidence:.1f}%)",
            use_container_width=True,
        )

    with img_col3:
        st.markdown(f"**{third_title}**")
        st.image(
            third_image,
            caption=third_caption,
            use_container_width=True,
        )

    # ─── Prediction Summary ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Prediction Summary</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-row">
            <span class="summary-label">Selected Model</span>
            <span class="summary-value">{selected_model}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Predicted Class</span>
            <span class="summary-value">{top_class_name}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Confidence</span>
            <span class="summary-value">{top_confidence:.1f}%</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Detected Regions</span>
            <span class="summary-value">{result.n_detections}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Inference Time</span>
            <span class="summary-value">{result.inference_time_ms:.1f} ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Download Results ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Download Results</div>',
                unsafe_allow_html=True)

    download_cols = st.columns(3)

    with download_cols[0]:
        st.download_button(
            "📥 Original MRI",
            image_to_bytes(img_rgb),
            file_name="original_mri.png",
            mime="image/png",
            use_container_width=True,
        )

    with download_cols[1]:
        st.download_button(
            "📥 Segmentation & Classification",
            image_to_bytes(combined_image),
            file_name="tumor_segmentation_classification.png",
            mime="image/png",
            use_container_width=True,
        )

    with download_cols[2]:
        st.download_button(
            "📥 Grad-CAM Result",
            image_to_bytes(third_image),
            file_name="gradcam_result.png",
            mime="image/png",
            use_container_width=True,
        )

    # ─── Footer ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">'
        'Research Prototype | YOLOv11x-Seg and YOLOv26x-Seg | '
        'Not a certified clinical diagnostic tool'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
