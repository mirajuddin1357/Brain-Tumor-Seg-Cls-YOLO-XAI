"""
YOLO Segmentation Pre/Post-Processing Module

Implements the complete preprocessing and two distinct postprocessing pipelines:

1. **YOLOv11x-Seg** (Standard YOLO, end2end=False):
   - output0 shape: [1, C, N]  ->  N candidates x (4 bbox + n_classes + 32 proto coeffs)
   - Transpose to [N, C]
   - Filter by confidence threshold
   - Apply Non-Maximum Suppression (NMS)
   - Matrix multiply mask coefficients with prototype masks [32, H_proto, W_proto]
   - Sigmoid -> Threshold at 0.5 -> Crop to bbox -> Upscale to original image size

2. **YOLOv26x-Seg** (End-to-End, end2end=True):
   - output0 shape: [1, N, 6]  ->  N detections x (x1, y1, x2, y2, score, class_id)
   - Already NMS-filtered by the ONNX model
   - Output includes prototype masks (output1)
   - Process masks for valid detections
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Container for preprocessed image and transformation metadata."""
    tensor: np.ndarray          # [1, 3, H, W] float32 normalized tensor
    scale: float                # Scale factor applied during letterboxing
    pad_top: int                # Top padding in pixels
    pad_left: int               # Left padding in pixels
    original_shape: Tuple[int, int]  # (height, width) of original image
    input_size: Tuple[int, int]      # (height, width) of model input


@dataclass
class SegmentationResult:
    """Container for complete segmentation and detection results."""
    boxes: np.ndarray           # [N, 4] float32 in (x1, y1, x2, y2) original image coords
    scores: np.ndarray          # [N] float32 confidence scores
    class_ids: np.ndarray       # [N] int64 class indices
    masks: Optional[np.ndarray] # [N, H_orig, W_orig] uint8 binary masks (0 or 1)
    proto_masks: Optional[np.ndarray] = None  # [32, H_proto, W_proto] prototype masks
    mask_coeffs: Optional[np.ndarray] = None  # [N, 32] mask coefficients
    inference_time_ms: float = 0.0            # Model forward pass time in ms
    preprocess_time_ms: float = 0.0           # Preprocessing time in ms
    postprocess_time_ms: float = 0.0          # Postprocessing time in ms

    @property
    def n_detections(self) -> int:
        """Number of detected objects."""
        return len(self.boxes) if self.boxes is not None else 0


# ─── Preprocessing ────────────────────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int]:
    """Resize image with letterboxing to preserve aspect ratio.

    Args:
        image: Input image in HWC format.
        target_size: Target (height, width) for the output.
        color: Padding fill color.

    Returns:
        Tuple of (letterboxed_image, scale_factor, pad_top, pad_left).
    """
    img_h, img_w = image.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / img_w, target_h / img_h)

    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return letterboxed, scale, pad_top, pad_left


def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (256, 256),
) -> PreprocessResult:
    """Complete YOLO preprocessing pipeline.

    Args:
        image: Input image as numpy array (HWC, BGR, uint8).
        input_size: Model input size as (height, width).

    Returns:
        PreprocessResult with tensor and transformation metadata.
    """
    original_h, original_w = image.shape[:2]

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Letterbox resize
    letterboxed, scale, pad_top, pad_left = letterbox(img_rgb, input_size)

    # Normalize to [0.0, 1.0] and convert to float32
    tensor = letterboxed.astype(np.float32) / 255.0

    # HWC -> CHW
    tensor = np.transpose(tensor, (2, 0, 1))

    # Add batch dimension: CHW -> NCHW
    tensor = np.expand_dims(tensor, axis=0)

    # Ensure contiguous array for ONNX Runtime
    tensor = np.ascontiguousarray(tensor)

    return PreprocessResult(
        tensor=tensor,
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
        original_shape=(original_h, original_w),
        input_size=input_size,
    )


# ─── NMS (Non-Maximum Suppression) ────────────────────────────────────────────

def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    """Standard Non-Maximum Suppression using OpenCV.

    Args:
        boxes: Bounding boxes as [N, 4] array in (x1, y1, x2, y2) format.
        scores: Confidence scores as [N] array.
        iou_threshold: IoU overlap threshold for suppression.

    Returns:
        List of indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    # OpenCV NMSBoxes expects (x, y, w, h)
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    return [int(i) for i in np.array(indices).flatten()]


# ─── Coordinate Transformations ───────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        boxes: Array of shape [N, 4] with (cx, cy, w, h).

    Returns:
        Array of shape [N, 4] with (x1, y1, x2, y2).
    """
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return xyxy


def scale_boxes(
    boxes: np.ndarray,
    prep: PreprocessResult,
) -> np.ndarray:
    """Map bounding boxes from letterbox input space back to original image coordinates.

    Args:
        boxes: Array of shape [N, 4] in letterbox space (x1, y1, x2, y2).
        prep: PreprocessResult containing scale and padding.

    Returns:
        Array of shape [N, 4] in original image coordinates, clipped to bounds.
    """
    scaled = boxes.copy()
    scaled[:, [0, 2]] -= prep.pad_left
    scaled[:, [1, 3]] -= prep.pad_top
    scaled /= prep.scale

    # Clip to original image boundaries
    orig_h, orig_w = prep.original_shape
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, orig_w)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, orig_h)

    return scaled


# ─── Mask Processing ─────────────────────────────────────────────────────────

def process_masks(
    proto: np.ndarray,
    mask_coeffs: np.ndarray,
    boxes_letterbox: np.ndarray,
    prep: PreprocessResult,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Generate instance segmentation masks from prototype masks and coefficients.

    Pipeline:
        1. Matrix multiply coefficients [N, 32] x prototypes [32, H*W] -> [N, H, W]
        2. Apply sigmoid activation to get probabilities in [0, 1]
        3. Upsample masks to letterbox input size
        4. Crop each mask to its bounding box
        5. Remove padding and scale to original image dimensions
        6. Threshold at mask_threshold to produce binary masks

    Args:
        proto: Prototype masks [32, H_proto, W_proto].
        mask_coeffs: Mask coefficients [N, 32].
        boxes_letterbox: Bounding boxes in letterbox space [N, 4].
        prep: PreprocessResult with transformation metadata.
        mask_threshold: Binarization threshold.

    Returns:
        Binary masks as [N, H_orig, W_orig] uint8 array (values 0 or 1).
    """
    n_detections = len(mask_coeffs)
    if n_detections == 0:
        orig_h, orig_w = prep.original_shape
        return np.zeros((0, orig_h, orig_w), dtype=np.uint8)

    c, mh, mw = proto.shape
    proto_flat = proto.reshape(c, -1)  # [32, H_proto * W_proto]

    # Matrix multiplication: [N, 32] @ [32, H_proto * W_proto] -> [N, H_proto * W_proto]
    raw_masks = np.matmul(mask_coeffs, proto_flat)

    # Sigmoid activation: 1 / (1 + exp(-x))
    masks = 1.0 / (1.0 + np.exp(-np.clip(raw_masks, -50.0, 50.0)))
    masks = masks.reshape(n_detections, mh, mw)  # [N, H_proto, W_proto]

    # Process each mask: upscale, crop to bbox, remove padding, threshold
    orig_h, orig_w = prep.original_shape
    input_h, input_w = prep.input_size
    final_masks = np.zeros((n_detections, orig_h, orig_w), dtype=np.uint8)

    for i in range(n_detections):
        mask_i = masks[i]

        # Upscale prototype mask to letterbox input size
        mask_upscaled = cv2.resize(
            mask_i,
            (input_w, input_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Crop to bounding box in letterbox space
        x1, y1, x2, y2 = boxes_letterbox[i].astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(input_w, x2)
        y2 = min(input_h, y2)

        cropped_mask = np.zeros_like(mask_upscaled)
        if x2 > x1 and y2 > y1:
            cropped_mask[y1:y2, x1:x2] = mask_upscaled[y1:y2, x1:x2]

        # Remove letterbox padding
        pad_top = prep.pad_top
        pad_left = prep.pad_left
        unpadded_h = input_h - 2 * pad_top
        unpadded_w = input_w - 2 * pad_left

        if unpadded_h > 0 and unpadded_w > 0:
            unpadded_mask = cropped_mask[
                pad_top : pad_top + unpadded_h,
                pad_left : pad_left + unpadded_w,
            ]
        else:
            unpadded_mask = cropped_mask

        # Resize to original image dimensions
        orig_mask = cv2.resize(
            unpadded_mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Binarize
        final_masks[i] = (orig_mask >= mask_threshold).astype(np.uint8)

    return final_masks


# ─── Postprocessing Pipelines ─────────────────────────────────────────────────

def postprocess_yolo11_seg(
    output0: np.ndarray,
    output1: np.ndarray,
    prep: PreprocessResult,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    mask_threshold: float = 0.5,
) -> SegmentationResult:
    """Postprocess outputs from standard YOLOv11x-Seg (non-end2end).

    output0: [1, C, N] -> N candidate detections x (4 bbox + n_classes + 32 proto coeffs)
    output1: [1, 32, H_proto, W_proto] -> Prototype masks

    Args:
        output0: Raw detection output tensor.
        output1: Prototype mask tensor.
        prep: PreprocessResult with transformation metadata.
        conf_threshold: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.
        mask_threshold: Binarization threshold for masks.

    Returns:
        SegmentationResult with mapped detections and masks.
    """
    # Squeeze batch dimension: [1, C, N] -> [C, N] -> transpose to [N, C]
    preds = np.squeeze(output0, axis=0)  # [C, N]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T  # Now [N, C]

    n_candidates, total_channels = preds.shape
    n_proto_coeffs = 32
    n_classes = total_channels - 4 - n_proto_coeffs

    # Parse columns: [cx, cy, w, h | class_0, ..., class_K | coeff_0, ..., coeff_31]
    boxes_cxcywh = preds[:, :4]
    class_probs = preds[:, 4 : 4 + n_classes]
    coeffs = preds[:, 4 + n_classes :]

    # Find best class and confidence for each candidate
    class_ids = np.argmax(class_probs, axis=1)
    scores = np.max(class_probs, axis=1)

    # Filter by confidence threshold
    conf_mask = scores >= conf_threshold
    if not np.any(conf_mask):
        orig_h, orig_w = prep.original_shape
        return SegmentationResult(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int64),
            masks=np.zeros((0, orig_h, orig_w), dtype=np.uint8),
        )

    boxes_cxcywh = boxes_cxcywh[conf_mask]
    scores = scores[conf_mask]
    class_ids = class_ids[conf_mask]
    coeffs = coeffs[conf_mask]

    # Convert cxcywh -> xyxy in letterbox space
    boxes_xyxy_lb = xywh2xyxy(boxes_cxcywh)

    # Apply NMS
    keep_indices = nms_boxes(boxes_xyxy_lb, scores, iou_threshold)
    if len(keep_indices) == 0:
        orig_h, orig_w = prep.original_shape
        return SegmentationResult(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int64),
            masks=np.zeros((0, orig_h, orig_w), dtype=np.uint8),
        )

    # Filter kept detections
    boxes_xyxy_lb = boxes_xyxy_lb[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    coeffs = coeffs[keep_indices]

    # Scale boxes to original image space
    boxes_orig = scale_boxes(boxes_xyxy_lb, prep)

    # Process masks from output1 (prototype masks)
    proto = np.squeeze(output1, axis=0)  # [32, H_proto, W_proto]
    masks = process_masks(
        proto, coeffs, boxes_xyxy_lb, prep, mask_threshold
    )

    return SegmentationResult(
        boxes=boxes_orig.astype(np.float32),
        scores=scores.astype(np.float32),
        class_ids=class_ids.astype(np.int64),
        masks=masks,
        proto_masks=proto,
        mask_coeffs=coeffs,
    )


def postprocess_yolo26_seg(
    output0: np.ndarray,
    output1: np.ndarray,
    prep: PreprocessResult,
    conf_threshold: float = 0.25,
    mask_threshold: float = 0.5,
) -> SegmentationResult:
    """Postprocess outputs from end-to-end YOLOv26x-Seg model.

    output0: [1, N, 6] -> N detections x (x1, y1, x2, y2, score, class_id)
    output1: [1, 32, H_proto, W_proto] -> Prototype masks

    Args:
        output0: Detection output tensor.
        output1: Prototype mask tensor.
        prep: PreprocessResult with transformation metadata.
        conf_threshold: Minimum confidence threshold.
        mask_threshold: Binarization threshold for masks.

    Returns:
        SegmentationResult with mapped detections and masks.
    """
    preds = np.squeeze(output0, axis=0)  # [N, 6] or similar

    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    if len(preds) == 0:
        orig_h, orig_w = prep.original_shape
        return SegmentationResult(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int64),
            masks=np.zeros((0, orig_h, orig_w), dtype=np.uint8),
        )

    # Columns: [x1, y1, x2, y2, score, class_id]
    boxes_lb = preds[:, :4]
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(np.int64)

    # Filter by confidence
    valid = scores >= conf_threshold
    if not np.any(valid):
        orig_h, orig_w = prep.original_shape
        return SegmentationResult(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int64),
            masks=np.zeros((0, orig_h, orig_w), dtype=np.uint8),
        )

    boxes_lb = boxes_lb[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    # Scale boxes to original image space
    boxes_orig = scale_boxes(boxes_lb, prep)

    # Handle prototype masks if present
    proto = None
    coeffs = None
    masks = None

    if output1 is not None and output1.size > 0:
        proto = np.squeeze(output1, axis=0)
        # For end-to-end models with masks, generate from prototypes if coeffs available
        if preds.shape[1] > 6:
            coeffs = preds[valid, 6:]
            masks = process_masks(
                proto, coeffs, boxes_lb, prep, mask_threshold
            )
        else:
            orig_h, orig_w = prep.original_shape
            masks = np.zeros((len(boxes_orig), orig_h, orig_w), dtype=np.uint8)
            for i, box in enumerate(boxes_orig):
                x1, y1, x2, y2 = box.astype(int)
                masks[i, y1:y2, x1:x2] = 1
    else:
        orig_h, orig_w = prep.original_shape
        masks = np.zeros((len(boxes_orig), orig_h, orig_w), dtype=np.uint8)
        for i, box in enumerate(boxes_orig):
            x1, y1, x2, y2 = box.astype(int)
            masks[i, y1:y2, x1:x2] = 1

    return SegmentationResult(
        boxes=boxes_orig.astype(np.float32),
        scores=scores.astype(np.float32),
        class_ids=class_ids,
        masks=masks,
        proto_masks=proto,
        mask_coeffs=coeffs,
    )
