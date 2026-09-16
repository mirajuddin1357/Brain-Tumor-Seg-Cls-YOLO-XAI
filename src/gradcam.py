"""
Grad-CAM Explainability Module for Brain Tumor Segmentation

Provides ProtoCAM-based saliency attribution heatmaps for YOLO segmentation
models deployed via ONNX Runtime.

IMPORTANT SCIENTIFIC NOTE:
    True gradient-based Grad-CAM requires PyTorch autograd hooks, which are
    not available through ONNX Runtime inference. This module implements
    ProtoCAM — a saliency attribution method that uses the model's prototype
    mask coefficients to generate activation heatmaps. ProtoCAM reflects the
    model's actual learned feature representations and provides a scientifically
    defensible explainability visualization.

    This is NOT a fabricated or random heatmap. The saliency map is derived
    directly from the model's internal prototype mask activations.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any


@dataclass
class GradCAMResult:
    """Container for Grad-CAM / ProtoCAM attribution results.

    Attributes:
        overlay: RGB image with heatmap overlaid on the original MRI.
        raw_heatmap_rgb: Colormapped heatmap image (without overlay).
        heatmap: Normalized [0,1] float32 heatmap.
        target_class_name: Name of the target class.
        method: Description of the attribution method used.
        is_valid: Whether a valid explainability result was produced.
        peak_coord: (y, x) coordinates of the peak activation.
    """
    overlay: np.ndarray
    raw_heatmap_rgb: np.ndarray
    heatmap: np.ndarray
    target_class_name: str = ""
    method: str = ""
    is_valid: bool = False
    peak_coord: Tuple[int, int] = (0, 0)


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a [0,1] float heatmap to RGB using the Jet colormap.

    Args:
        heatmap: 2D float32 array normalized to [0, 1].

    Returns:
        RGB colormapped image (uint8).
    """
    norm = np.clip(heatmap, 0.0, 1.0)
    uint8_map = (norm * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _create_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
    threshold: float = 0.05,
) -> np.ndarray:
    """Blend image with colormapped heatmap.

    Only regions with activation above the threshold are blended,
    preserving the original MRI appearance in low-activation areas.

    Args:
        image_rgb: Original MRI image (RGB, uint8).
        heatmap: Normalized [0,1] heatmap.
        alpha: Maximum blend opacity for high-activation regions.
        threshold: Minimum activation to start blending.

    Returns:
        Blended overlay image (RGB, uint8).
    """
    h, w = image_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    colored_heatmap = _apply_colormap(heatmap)

    # Smooth transition: only apply where activation exceeds threshold
    mask_weights = np.clip(
        (heatmap - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    effective_alpha = (mask_weights * alpha)[:, :, np.newaxis]

    blended = (
        (1.0 - effective_alpha) * image_rgb.astype(np.float32)
        + effective_alpha * colored_heatmap.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


class GradCAMEngine:
    """Grad-CAM saliency attribution engine for YOLO segmentation models.

    Uses prototype feature maps (ProtoCAM) from the YOLO segmentation head
    to generate explainability heatmaps. This method uses the model's actual
    learned prototype masks and mask coefficients — it does not fabricate
    or randomize the visualization.

    If prototype data is not available (e.g., no detections), the engine
    reports that Grad-CAM is unavailable rather than producing a fake result.
    """

    DISCLAIMER = (
        "**Note:** True gradient-based Grad-CAM requires PyTorch autograd "
        "hooks, which are unavailable in an ONNX-only deployment. This "
        "visualization uses **ProtoCAM** — a saliency attribution derived "
        "from the model's prototype mask coefficients and feature maps."
    )

    def explain(
        self,
        image_rgb: np.ndarray,
        segmentation_result: Any,
        class_names: Optional[List[str]] = None,
        alpha: float = 0.50,
    ) -> GradCAMResult:
        """Compute ProtoCAM heatmap and overlay.

        Uses the model's mask coefficients and prototype masks to generate
        a saliency map showing which regions the model focused on.

        Args:
            image_rgb: Original MRI image (RGB, uint8).
            segmentation_result: SegmentationResult from inference.
            class_names: List of class name strings.
            alpha: Overlay opacity (0=transparent, 1=opaque).

        Returns:
            GradCAMResult with heatmap, overlay, and validity flag.
        """
        orig_h, orig_w = image_rgb.shape[:2]

        # Check for detections
        n_dets = getattr(segmentation_result, "n_detections", 0)
        if n_dets == 0:
            empty = np.zeros((orig_h, orig_w), dtype=np.float32)
            return GradCAMResult(
                overlay=image_rgb.copy(),
                raw_heatmap_rgb=_apply_colormap(empty),
                heatmap=empty,
                target_class_name="None",
                method="Unavailable (no detections)",
                is_valid=False,
            )

        # Get prototype data
        mask_coeffs = getattr(segmentation_result, "mask_coefficients", None)
        proto_masks = getattr(segmentation_result, "proto_masks", None)

        # Determine target class name
        class_ids = getattr(segmentation_result, "class_ids", None)
        top_cls = int(class_ids[0]) if class_ids is not None and len(class_ids) > 0 else 0
        cls_name = (
            class_names[top_cls]
            if class_names and top_cls < len(class_names)
            else f"Class_{top_cls}"
        )

        # ProtoCAM: use prototype masks and coefficients
        if mask_coeffs is not None and proto_masks is not None:
            n_protos, proto_h, proto_w = proto_masks.shape
            protos_flat = proto_masks.reshape(n_protos, -1)

            # Aggregate all detections (maximum activation across detections)
            activation = np.zeros((proto_h, proto_w), dtype=np.float32)
            for i in range(n_dets):
                raw_i = (mask_coeffs[i] @ protos_flat).reshape(proto_h, proto_w)
                sig_i = 1.0 / (1.0 + np.exp(-np.clip(raw_i, -50, 50)))
                activation = np.maximum(activation, sig_i)

            # Remove letterbox padding from the activation map
            prep = getattr(segmentation_result, "preprocess_info", None)
            if prep is not None:
                input_h, input_w = prep.input_size
                sx = proto_w / input_w
                sy = proto_h / input_h
                py1 = max(0, int(prep.pad_top * sy))
                px1 = max(0, int(prep.pad_left * sx))
                py2 = min(proto_h, py1 + int((input_h - 2 * prep.pad_top) * sy))
                px2 = min(proto_w, px1 + int((input_w - 2 * prep.pad_left) * sx))
                activation = activation[py1:py2, px1:px2]

            # Resize to original image dimensions
            heatmap_full = cv2.resize(
                activation, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )

            # Normalize to [0, 1]
            h_min, h_max = heatmap_full.min(), heatmap_full.max()
            if h_max > h_min:
                heatmap_norm = (heatmap_full - h_min) / (h_max - h_min)
            else:
                heatmap_norm = heatmap_full

            # Find peak activation coordinate
            peak_y, peak_x = np.unravel_index(
                np.argmax(heatmap_norm), heatmap_norm.shape
            )

            # Create visualizations
            raw_rgb = _apply_colormap(heatmap_norm)
            overlay = _create_overlay(image_rgb, heatmap_norm, alpha)

            return GradCAMResult(
                overlay=overlay,
                raw_heatmap_rgb=raw_rgb,
                heatmap=heatmap_norm,
                target_class_name=cls_name,
                method="Grad-CAM Saliency (ProtoCAM)",
                is_valid=True,
                peak_coord=(int(peak_y), int(peak_x)),
            )

        # No prototype data available — cannot generate legitimate Grad-CAM
        empty = np.zeros((orig_h, orig_w), dtype=np.float32)
        return GradCAMResult(
            overlay=image_rgb.copy(),
            raw_heatmap_rgb=_apply_colormap(empty),
            heatmap=empty,
            target_class_name=cls_name,
            method="Unavailable (no prototype data)",
            is_valid=False,
        )
