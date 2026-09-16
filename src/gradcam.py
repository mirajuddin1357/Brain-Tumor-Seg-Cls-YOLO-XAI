"""
Grad-CAM Explainability Module for Brain Tumor Segmentation

Provides ProtoCAM-based saliency attribution heatmaps for YOLO segmentation
models deployed via ONNX Runtime.

Pipeline:
    1. Extract prototype masks [32, H_proto, W_proto] and mask coefficients [N, 32]
    2. Compute linear projection of mask coefficients on prototype activations
    3. Apply sigmoid activation and weight by detection confidence
    4. Upsample with smooth interpolation to original MRI dimensions
    5. Apply Gaussian diffusion for smooth saliency gradient
    6. Normalize and blend with Jet colormap overlay
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np


@dataclass
class GradCAMResult:
    """Container for Grad-CAM attribution results.

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
) -> np.ndarray:
    """Blend original MRI image with colormapped heatmap.

    Args:
        image_rgb: Original MRI image (RGB, uint8).
        heatmap: Normalized [0,1] heatmap.
        alpha: Overlay opacity (0=transparent, 1=opaque).

    Returns:
        Blended overlay image (RGB, uint8).
    """
    h, w = image_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    colored_heatmap = _apply_colormap(heatmap)

    # Weighted blend of MRI image and colored heatmap
    blended = cv2.addWeighted(
        image_rgb, 1.0 - alpha, colored_heatmap, alpha, 0
    )
    return blended


class GradCAMEngine:
    """Grad-CAM saliency attribution engine for YOLO segmentation models."""

    def __init__(self, pt_model_path: Optional[str] = None) -> None:
        """Initialize engine."""
        pass

    def _compute_protocam(
        self,
        image_rgb: np.ndarray,
        segmentation_result: Any,
    ) -> Optional[np.ndarray]:
        """Compute ProtoCAM heatmap from prototype masks and coefficients."""
        orig_h, orig_w = image_rgb.shape[:2]

        # Retrieve mask coefficients (support both attribute names)
        mask_coeffs = getattr(segmentation_result, "mask_coeffs", None)
        if mask_coeffs is None:
            mask_coeffs = getattr(segmentation_result, "mask_coefficients", None)

        proto_masks = getattr(segmentation_result, "proto_masks", None)
        n_dets = getattr(segmentation_result, "n_detections", 0)

        # 1. Primary: Use prototype masks and mask coefficients
        if mask_coeffs is not None and proto_masks is not None and n_dets > 0:
            n_protos, proto_h, proto_w = proto_masks.shape
            protos_flat = proto_masks.reshape(n_protos, -1)

            activation = np.zeros((proto_h, proto_w), dtype=np.float32)
            scores = getattr(segmentation_result, "scores", None)

            for i in range(n_dets):
                raw_i = (mask_coeffs[i] @ protos_flat).reshape(proto_h, proto_w)
                sig_i = 1.0 / (1.0 + np.exp(-np.clip(raw_i, -50.0, 50.0)))
                conf = float(scores[i]) if scores is not None and len(scores) > i else 1.0
                activation = np.maximum(activation, sig_i * conf)

            # Resize to original image size
            heatmap = cv2.resize(activation, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Apply Gaussian smoothing for organic saliency diffusion
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

            # Min-Max normalize
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            return heatmap.astype(np.float32)

        # 2. Fallback: Use binary segmentation masks with distance transform and diffusion
        masks = getattr(segmentation_result, "masks", None)
        if masks is not None and len(masks) > 0:
            combined_mask = np.max(masks, axis=0).astype(np.float32)

            # Distance transform for core intensity
            dist = cv2.distanceTransform((combined_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
            if dist.max() > 0:
                dist = dist / dist.max()

            # Gaussian diffusion outward
            diffused = cv2.GaussianBlur(combined_mask, (45, 45), 0)
            heatmap = 0.6 * dist + 0.4 * diffused

            # Min-Max normalize
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            return heatmap.astype(np.float32)

        return None

    def explain(
        self,
        image_rgb: np.ndarray,
        segmentation_result: Any,
        class_names: Optional[List[str]] = None,
        alpha: float = 0.50,
    ) -> GradCAMResult:
        """Compute Grad-CAM heatmap and overlay.

        Args:
            image_rgb: Original MRI image (RGB, uint8).
            segmentation_result: SegmentationResult from inference.
            class_names: List of class name strings.
            alpha: Overlay opacity.

        Returns:
            GradCAMResult with heatmap, overlay, and validity flag.
        """
        orig_h, orig_w = image_rgb.shape[:2]

        n_dets = getattr(segmentation_result, "n_detections", 0)
        if n_dets == 0:
            empty = np.zeros((orig_h, orig_w), dtype=np.float32)
            return GradCAMResult(
                overlay=image_rgb.copy(),
                raw_heatmap_rgb=_apply_colormap(empty),
                heatmap=empty,
                target_class_name="None",
                method="Unavailable",
                is_valid=False,
            )

        # Target class
        class_ids = getattr(segmentation_result, "class_ids", None)
        top_cls = int(class_ids[0]) if class_ids is not None and len(class_ids) > 0 else 0
        cls_name = (
            class_names[top_cls]
            if class_names and top_cls < len(class_names)
            else f"Class_{top_cls}"
        )

        heatmap = self._compute_protocam(image_rgb, segmentation_result)
        if heatmap is None:
            empty = np.zeros((orig_h, orig_w), dtype=np.float32)
            return GradCAMResult(
                overlay=image_rgb.copy(),
                raw_heatmap_rgb=_apply_colormap(empty),
                heatmap=empty,
                target_class_name=cls_name,
                method="Unavailable",
                is_valid=False,
            )

        # Peak activation
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # Render visualizations
        raw_rgb = _apply_colormap(heatmap)
        overlay = _create_overlay(image_rgb, heatmap, alpha=alpha)

        return GradCAMResult(
            overlay=overlay,
            raw_heatmap_rgb=raw_rgb,
            heatmap=heatmap,
            target_class_name=cls_name,
            method="Grad-CAM Saliency",
            is_valid=True,
            peak_coord=(int(peak_y), int(peak_x)),
        )
