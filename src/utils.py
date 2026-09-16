"""
Utility Functions for Brain Tumor MRI Segmentation & Classification

Provides:
- ONNX metadata parsing for class names
- Brain tumor fallback class names
- Color palette for tumor classes
- Image format conversion (PIL ↔ OpenCV)
- Image-to-bytes conversion for download
- Model and asset file validation
"""

import ast
import io
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ─── Brain Tumor Default Class Names ─────────────────────────────────────────

BRAIN_TUMOR_CLASSES: List[str] = ["Glioma", "Meningioma", "Pituitary"]


# ─── ONNX Metadata Parsing ───────────────────────────────────────────────────

def parse_class_names_from_metadata(metadata_str: str) -> List[str]:
    """Parse class names from ONNX model metadata 'names' field.

    The metadata typically stores names as a Python dict string like:
    {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}

    Args:
        metadata_str: The raw string from model metadata.

    Returns:
        Ordered list of class names. Falls back to BRAIN_TUMOR_CLASSES on error.
    """
    try:
        names_dict = ast.literal_eval(metadata_str)
        if isinstance(names_dict, dict):
            max_idx = max(names_dict.keys())
            return [names_dict.get(i, f"class_{i}") for i in range(max_idx + 1)]
    except (ValueError, SyntaxError):
        pass

    return BRAIN_TUMOR_CLASSES


# ─── Color Palette ────────────────────────────────────────────────────────────

# Distinct, accessible colors for brain tumor classes
TUMOR_COLORS: List[Tuple[int, int, int]] = [
    (46, 134, 171),   # Glioma — teal blue
    (231, 111, 81),    # Meningioma — coral
    (42, 157, 143),    # Pituitary — sea green
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Get RGB color for a given class ID.

    Args:
        class_id: Integer class index.

    Returns:
        (R, G, B) color tuple.
    """
    return TUMOR_COLORS[class_id % len(TUMOR_COLORS)]


# ─── Image Conversion ────────────────────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array.

    Args:
        pil_img: PIL Image object.

    Returns:
        BGR numpy array (uint8).
    """
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB numpy array.

    Args:
        img_bgr: BGR numpy array.

    Returns:
        RGB numpy array (uint8).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def image_to_bytes(img_rgb: np.ndarray) -> bytes:
    """Convert RGB numpy array to PNG bytes for download.

    Args:
        img_rgb: RGB numpy array (uint8).

    Returns:
        PNG image as bytes.
    """
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()


# ─── File Validation ─────────────────────────────────────────────────────────

def validate_model_file(model_path: str) -> bool:
    """Check if a model file exists.

    Args:
        model_path: Absolute path to the ONNX model file.

    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(model_path)


def validate_demo_image(image_path: str) -> bool:
    """Check if the demo image file exists.

    Args:
        image_path: Absolute path to the demo image file.

    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(image_path)


def get_missing_model_message(models_dir: str, model_files: Dict[str, str]) -> Optional[str]:
    """Generate a user-friendly error message for missing models.

    Args:
        models_dir: Path to the models directory.
        model_files: Dict mapping display name to filename.

    Returns:
        Error message string, or None if all models are present.
    """
    missing = []
    for display_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if not os.path.isfile(model_path):
            missing.append((display_name, filename))

    if not missing:
        return None

    msg_lines = ["### ⚠️ Missing Model Files\n"]
    msg_lines.append("The following ONNX model files were not found:\n")
    for display_name, filename in missing:
        msg_lines.append(f"- `models/{filename}` ({display_name})")
    msg_lines.append("\n**To fix this:**")
    msg_lines.append("1. Place your trained ONNX model files in the `models/` directory.")
    msg_lines.append("2. Ensure the filenames match exactly:")
    for _, filename in model_files.items():
        msg_lines.append(f"   - `models/{filename}`")
    msg_lines.append("3. Restart the application.")

    return "\n".join(msg_lines)
