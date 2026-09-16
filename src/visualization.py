"""
Visualization Module for Brain Tumor MRI Segmentation & Classification

Provides functions to draw:
- Segmentation mask overlays on MRI images
- Bounding boxes with class labels and confidence percentages
- Combined segmentation + detection all-in-one image
- Composite mask images for download
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import get_color_for_class


# ─── Segmentation Overlay ────────────────────────────────────────────────────

def draw_segmentation_overlay(
    image_rgb: np.ndarray,
    masks: Optional[np.ndarray],
    class_ids: np.ndarray,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Draw segmentation masks overlaid on the original MRI image.

    Args:
        image_rgb: Original image (RGB, HWC, uint8).
        masks: Binary masks as [N, H, W] array (same size as image), or None.
        class_ids: Class indices as [N] array.
        mask_alpha: Transparency of mask overlay (0=invisible, 1=opaque).

    Returns:
        Annotated image copy (RGB, HWC, uint8) with mask overlay.
    """
    annotated = image_rgb.copy()

    if masks is None or len(masks) == 0:
        return annotated

    n_detections = len(masks)
    mask_overlay = annotated.copy()

    for i in range(n_detections):
        class_id = int(class_ids[i])
        color = get_color_for_class(class_id)
        binary_mask = masks[i].astype(bool)
        mask_overlay[binary_mask] = color

    # Blend mask overlay with original
    annotated = cv2.addWeighted(
        annotated, 1 - mask_alpha, mask_overlay, mask_alpha, 0
    )

    return annotated


# ─── Detection and Classification ────────────────────────────────────────────

def draw_detection_result(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
) -> np.ndarray:
    """Draw bounding boxes with class names and confidence percentages.

    Label format: "ClassName XX.X%" (pure ASCII to avoid OpenCV ??? glyphs).

    Args:
        image_rgb: Original image (RGB, HWC, uint8).
        boxes: Detection boxes as [N, 4] array in (x1, y1, x2, y2) format.
        scores: Confidence scores as [N] array.
        class_ids: Class indices as [N] array.
        class_names: List of class name strings.

    Returns:
        Annotated image copy (RGB, HWC, uint8) with bounding boxes and labels.
    """
    annotated = image_rgb.copy()
    n_detections = len(boxes)

    if n_detections == 0:
        return annotated

    for i in range(n_detections):
        x1, y1, x2, y2 = boxes[i].astype(int)
        class_id = int(class_ids[i])
        score = float(scores[i])
        color = get_color_for_class(class_id)

        # Get class name safely
        if class_id < len(class_names):
            cls_name = class_names[class_id]
        else:
            cls_name = f"Class_{class_id}"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label text (pure ASCII)
        label = f"{cls_name} {score * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Position label above the box
        label_y1 = max(y1 - text_h - baseline - 6, 0)
        label_y2 = max(y1, text_h + baseline + 6)

        # Label background
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w + 8, label_y2),
            color, -1, cv2.LINE_AA,
        )

        # Label text (white on colored background)
        cv2.putText(
            annotated, label,
            (x1 + 4, label_y2 - baseline - 3),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return annotated


# ─── Combined Segmentation + Detection (All-in-One Image) ────────────────────

def draw_full_result(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    masks: Optional[np.ndarray],
    class_names: List[str],
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Draw segmentation masks, bounding boxes, and labels together on an image.

    Args:
        image_rgb: Original image (RGB, HWC, uint8).
        boxes: Detection boxes as [N, 4] array in (x1, y1, x2, y2) format.
        scores: Confidence scores as [N] array.
        class_ids: Class indices as [N] array.
        masks: Binary masks as [N, H, W] array, or None.
        class_names: List of class name strings.
        mask_alpha: Transparency of mask overlay.

    Returns:
        Annotated image copy (RGB, HWC, uint8).
    """
    # Draw masks first
    annotated = draw_segmentation_overlay(image_rgb, masks, class_ids, mask_alpha)

    # Draw boxes and labels on top
    n_detections = len(boxes)
    if n_detections == 0:
        return annotated

    for i in range(n_detections):
        x1, y1, x2, y2 = boxes[i].astype(int)
        class_id = int(class_ids[i])
        score = float(scores[i])
        color = get_color_for_class(class_id)

        if class_id < len(class_names):
            cls_name = class_names[class_id]
        else:
            cls_name = f"Class_{class_id}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label text (pure ASCII)
        label = f"{cls_name} {score * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y1 = max(y1 - text_h - baseline - 6, 0)
        label_y2 = max(y1, text_h + baseline + 6)

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w + 8, label_y2),
            color, -1, cv2.LINE_AA,
        )

        cv2.putText(
            annotated, label,
            (x1 + 4, label_y2 - baseline - 3),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return annotated


# ─── Mask Image ───────────────────────────────────────────────────────────────

def create_mask_image(
    masks: np.ndarray,
    class_ids: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Create a composite colored mask image on a black background.

    Args:
        masks: Binary masks as [N, H, W] array.
        class_ids: Class indices as [N] array.
        image_shape: (height, width) of the target image.

    Returns:
        Colored mask image (RGB, HWC, uint8) with black background.
    """
    h, w = image_shape[:2]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8)

    if masks is None or len(masks) == 0:
        return mask_image

    for i in range(len(masks)):
        class_id = int(class_ids[i])
        color = get_color_for_class(class_id)
        binary_mask = masks[i].astype(bool)
        mask_image[binary_mask] = color

    return mask_image
