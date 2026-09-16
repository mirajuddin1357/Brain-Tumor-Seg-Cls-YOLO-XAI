"""
ONNX Model Inference Engine for Brain Tumor Segmentation

Provides the BrainTumorSegmenter class that orchestrates the complete
inference pipeline:
    Image -> Preprocess -> ONNX Runtime -> Postprocess -> SegmentationResult

Automatically handles:
- Model variant detection (YOLOv11x-Seg vs YOLOv26x-Seg)
- Metadata parsing (class names, head type, end2end flag)
- Preprocessing with letterbox resizing to native resolution
- Postprocessing routed to the correct architecture
- Fallback class names for brain tumor types
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from .segmentation import (
    PreprocessResult,
    SegmentationResult,
    preprocess_image,
    postprocess_yolo11_seg,
    postprocess_yolo26_seg,
)
from .utils import (
    BRAIN_TUMOR_CLASSES,
    parse_class_names_from_metadata,
)


class BrainTumorSegmenter:
    """ONNX Runtime inference engine for brain tumor MRI segmentation.

    Encapsulates session creation, metadata inspection, input preprocessing,
    forward pass execution, and architecture-specific postprocessing.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the segmenter with an ONNX model.

        Args:
            model_path: Absolute or relative path to the .onnx file.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If ONNX Runtime fails to load the model.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                f"Please place the model file in the models/ directory."
            )

        self.model_path = model_path
        self.model_name = os.path.basename(model_path).replace(".onnx", "")

        # Create ONNX Runtime session (CPU only for deployment compatibility)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Extract input metadata
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape  # e.g., [1, 3, 256, 256]

        # Determine input size (height, width)
        try:
            self.native_input_size: Tuple[int, int] = (
                int(self.input_shape[2]),
                int(self.input_shape[3]),
            )
        except (IndexError, TypeError, ValueError):
            self.native_input_size = (256, 256)

        # Extract output metadata
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.output_shapes = [o.shape for o in self.session.get_outputs()]

        # Parse ONNX custom metadata
        meta = self.session.get_modelmeta()
        custom = meta.custom_metadata_map

        # Detect model variant
        self.is_end2end = custom.get("end2end", "False").lower() == "true"
        self.head_type = custom.get("head", "Segment")

        # Parse class names from metadata
        names_str = custom.get("names", "")
        if names_str:
            self.class_names = parse_class_names_from_metadata(names_str)
        else:
            self.class_names = BRAIN_TUMOR_CLASSES
        self.n_classes = len(self.class_names)

        # Store stride for reference
        self.stride = int(custom.get("stride", "32"))

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> SegmentationResult:
        """Run complete segmentation inference on an image.

        Pipeline: preprocess -> ONNX forward pass -> postprocess

        Args:
            image: Input image as numpy array (HWC, BGR, uint8).
            conf_threshold: Minimum confidence score to keep detections.
            iou_threshold: IoU threshold for NMS (only used by non-end2end models).
            input_size: Override input size as (height, width). If None,
                        uses the model's native input size.

        Returns:
            SegmentationResult with detections mapped to original image coordinates.
        """
        # Prioritize model's native input size to prevent dimension mismatch
        if (
            hasattr(self, "native_input_size")
            and isinstance(self.native_input_size[0], int)
            and isinstance(self.native_input_size[1], int)
            and self.native_input_size[0] > 0
            and self.native_input_size[1] > 0
        ):
            input_size = self.native_input_size
        elif input_size is None:
            input_size = (256, 256)

        # Step 1: Preprocess
        prep = preprocess_image(image, input_size)

        # Step 2: ONNX Runtime inference (timed)
        start_time = time.perf_counter()
        outputs = self.session.run(
            self.output_names,
            {self.input_name: prep.tensor},
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        output0 = outputs[0]  # Detection outputs
        output1 = outputs[1] if len(outputs) > 1 else None  # Prototype masks

        # Step 3: Postprocess (route based on model type)
        if self.is_end2end:
            result = postprocess_yolo26_seg(
                output0, output1, prep,
                conf_threshold=conf_threshold,
            )
        else:
            result = postprocess_yolo11_seg(
                output0, output1, prep,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        result.inference_time_ms = inference_time_ms
        return result

    def get_model_info(self) -> dict:
        """Return a dictionary of model metadata for display.

        Returns:
            Dictionary with model name, type, input/output shapes, etc.
        """
        return {
            "Model Name": self.model_name,
            "Head Type": self.head_type,
            "End-to-End NMS": self.is_end2end,
            "Input Shape": str(self.input_shape),
            "Number of Classes": self.n_classes,
            "Class Names": ", ".join(self.class_names),
            "Stride": self.stride,
        }

    def __repr__(self) -> str:
        return (
            f"BrainTumorSegmenter(model='{self.model_name}', "
            f"end2end={self.is_end2end}, "
            f"classes={self.n_classes})"
        )
