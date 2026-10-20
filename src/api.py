"""
FastAPI Production REST API for Industrial Surface Defect Detection.

Endpoints:
- GET  /              : Welcome & API documentation info
- GET  /health        : Health check & model status
- GET  /model/classes : List supported defect classes
- GET  /model/metrics : Validation performance metrics
- POST /predict       : Multipart image upload returning JSON detections
- POST /predict/image : Multipart image upload returning annotated JPEG image
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    API_HOST,
    API_PORT,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFECT_DESCRIPTIONS,
    DEVICE,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_SIZE_MB,
    MODEL_PATH,
)
from src.predict import DefectPredictor, resolve_path


# --- Pydantic Response Schemas ---
class DetectionItem(BaseModel):
    class_id: int = Field(..., description="Numeric class identifier (0-5)")
    class_name: str = Field(..., description="Defect category name")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    confidence_pct: str = Field(..., description="Human-readable confidence percentage")
    bbox_xyxy: List[float] = Field(..., description="Bounding box coordinates [xmin, ymin, xmax, ymax]")
    width: float = Field(..., description="Bounding box width in pixels")
    height: float = Field(..., description="Bounding box height in pixels")
    area: float = Field(..., description="Bounding box area in square pixels")


class PredictionResponse(BaseModel):
    status: str = "success"
    filename: str
    defect_detected: bool
    detection_count: int
    detected_classes: List[str]
    detections: List[DetectionItem]
    latency_ms: float
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    defect_classes: List[str]
    uptime_seconds: float


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Industrial Surface Defect Detection API",
    description=(
        "High-performance computer vision API for automated industrial quality inspection. "
        "Detects and localizes 6 steel surface defect types: crazing, inclusion, patches, "
        "pitted_surface, rolled-in_scale, and scratches using YOLOv8."
    ),
    version="1.0.0",
)

# Enable CORS for cross-origin web/industrial dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App Global State
START_TIME = time.time()
PREDICTOR: Optional[DefectPredictor] = None


def get_predictor() -> DefectPredictor:
    """Retrieve or lazily initialize the DefectPredictor instance."""
    global PREDICTOR
    if PREDICTOR is None:
        weights_path = Path(MODEL_PATH)
        if not weights_path.exists():
            weights_path = ROOT_DIR / "outputs" / "weights" / "best.pt"
        if not weights_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Trained model weights not found at: {weights_path}",
            )
        PREDICTOR = DefectPredictor(model_path=str(weights_path), device=DEVICE)
    return PREDICTOR


@app.on_event("startup")
def startup_event():
    """Preload YOLO model weights into memory at startup."""
    try:
        get_predictor()
        print("DefectPredictor model loaded successfully into FastAPI application.")
    except Exception as e:
        print(f"Warning: Could not preload model on startup: {e}")


# --- API Endpoints ---
@app.get("/", tags=["General"])
def root():
    """API Welcome & service info."""
    return {
        "service": "Industrial Surface Defect Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict_json": "/predict",
            "predict_annotated_image": "/predict/image",
            "defect_classes": "/model/classes",
            "validation_metrics": "/model/metrics",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint confirming API readiness and model status."""
    try:
        predictor = get_predictor()
        classes = list(predictor.class_names.values())
        model_loaded = True
        model_path = str(predictor.model_path)
    except Exception:
        classes = []
        model_loaded = False
        model_path = "unavailable"

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": model_path,
        "defect_classes": classes,
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.get("/model/classes", tags=["Model Information"])
def get_defect_classes():
    """Return dictionary and descriptions of supported defect classes."""
    predictor = get_predictor()
    return {
        "count": len(predictor.class_names),
        "classes": [
            {
                "id": cid,
                "name": cname,
                "description": DEFECT_DESCRIPTIONS.get(cname, "Surface defect"),
            }
            for cid, cname in predictor.class_names.items()
        ],
    }


@app.get("/model/metrics", tags=["Model Information"])
def get_model_metrics():
    """Return ground-truth evaluation metrics on the validation dataset."""
    metrics_path = ROOT_DIR / "outputs" / "evaluation" / "evaluation_metrics.json"
    if not metrics_path.exists():
        metrics_path = ROOT_DIR / "outputs" / "training_metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Evaluation metrics report not found.",
    )


async def validate_and_load_image(file: UploadFile) -> Image.Image:
    """Helper to validate file content and parse into PIL RGB Image."""
    valid_content_types = {
        "image/jpeg",
        "image/jpg",
        "image/pjpeg",
        "image/png",
        "image/x-png",
        "image/bmp",
        "image/webp",
        "image/tiff",
        "application/octet-stream",
    }
    if file.content_type and file.content_type.lower() not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: '{file.content_type}'. Must be an image (JPEG, PNG, BMP, WEBP).",
        )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
        if len(contents) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum allowed size ({MAX_IMAGE_SIZE_MB} MB).",
            )
        img = Image.open(io.BytesIO(contents))
        img.verify()  # Verify image integrity
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return img
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file could not be parsed as a valid image: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_defect(
    file: UploadFile = File(..., description="Industrial component image file (.jpg, .png, etc.)"),
    conf: float = Query(DEFAULT_CONFIDENCE, ge=0.01, le=1.0, description="Detection confidence threshold"),
    imgsz: int = Query(DEFAULT_IMGSZ, ge=32, le=1280, description="Inference image resolution"),
    return_base64: bool = Query(False, description="Whether to include base64-encoded annotated image in JSON"),
):
    """
    Run defect detection inference on an uploaded industrial inspection image.

    Returns:
    - Bounding box coordinates [xmin, ymin, xmax, ymax]
    - Defect class names and IDs
    - Confidence scores (0.0 to 1.0)
    - Defect count and latency measurements
    """
    predictor = get_predictor()
    img = await validate_and_load_image(file)

    start_time = time.time()
    result = predictor.predict_image(
        image_input=img,
        conf=conf,
        save_annotated=False,
        imgsz=imgsz,
    )
    latency_ms = round((time.time() - start_time) * 1000, 2)

    detections = result.get("detections", [])
    detected_classes = sorted(list(set(d["class_name"] for d in detections)))

    b64_image = None
    if return_base64 and "annotated_image" in result:
        annotated_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", annotated_bgr)
        b64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "success",
        "filename": file.filename or "uploaded_image.jpg",
        "defect_detected": len(detections) > 0,
        "detection_count": len(detections),
        "detected_classes": detected_classes,
        "detections": detections,
        "latency_ms": latency_ms,
        "annotated_image_base64": b64_image,
    }


@app.post("/predict/image", tags=["Inference"])
async def predict_and_visualize(
    file: UploadFile = File(..., description="Industrial inspection image"),
    conf: float = Query(DEFAULT_CONFIDENCE, ge=0.01, le=1.0, description="Detection confidence threshold"),
    imgsz: int = Query(DEFAULT_IMGSZ, ge=32, le=1280, description="Inference resolution"),
):
    """
    Run defect detection and return the annotated JPEG image directly.
    Ideal for direct display in industrial HMI screens and web dashboards.
    """
    predictor = get_predictor()
    img = await validate_and_load_image(file)

    result = predictor.predict_image(
        image_input=img,
        conf=conf,
        save_annotated=False,
        imgsz=imgsz,
    )

    annotated_rgb = result["annotated_image"]
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode annotated output image.",
        )

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=False)
