# Brain Tumor MRI Segmentation and Classification

**YOLO-Based Deep Learning with Grad-CAM Explainability**

Research prototype Streamlit application for the IEEE paper:
*"Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning with Grad-CAM Explainability and Streamlit Deployment"*

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Model Files

Copy your trained ONNX models into the `models/` directory:

```
models/
├── YOLOv11x_Seg_best.onnx
└── YOLOv26x_Seg_best.onnx
```

### 3. Place Demo Image

Place a representative brain tumor MRI image at:

```
assets/demo_image.png
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## Features

- **Model Selection**: Choose between YOLOv11x-Seg and YOLOv26x-Seg
- **MRI Input**: Upload MRI images (PNG, JPG, JPEG) or use the demo image
- **Tumor Segmentation**: Overlay predicted segmentation masks on MRI
- **Detection & Classification**: Bounding boxes with tumor class and confidence
- **Grad-CAM Explainability**: ProtoCAM-based saliency attribution (when enabled)
- **Prediction Summary**: Model, class, confidence, detections, inference time
- **Download Results**: Export segmentation, detection, and Grad-CAM visualizations

## Project Structure

```
brain_tumor_streamlit/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── models/                 # ONNX model files (user-provided)
├── assets/                 # Demo MRI image (user-provided)
└── src/
    ├── __init__.py
    ├── inference.py         # ONNX model loading & inference
    ├── segmentation.py      # Pre/post-processing pipelines
    ├── visualization.py     # Result visualization
    ├── gradcam.py           # Grad-CAM explainability
    └── utils.py             # Utility functions
```

## Tumor Classes

The models are trained on three brain tumor categories:
- **Glioma**
- **Meningioma**
- **Pituitary**

## Disclaimer

This is a **research prototype** and is not a certified clinical diagnostic tool.
Do not use this application for medical diagnosis or treatment decisions.

## Citation

If you use this work, please cite the associated IEEE paper.
