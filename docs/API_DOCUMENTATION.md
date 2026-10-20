# Production API Documentation & Architecture Reference

Industrial Surface Defect Detection REST Microservice powered by **FastAPI** and **YOLOv8**.

---

## 🏛️ Architecture Overview

The system features a decoupled dual-tier architecture:

```
┌─────────────────────────────────┐         ┌────────────────────────────────┐
│  Streamlit Inspection Frontend   │         │    Third-Party Factory HMI     │
│  (Port 8501)                    │         │    / Camera Edge Controller    │
└────────────────┬────────────────┘         └────────────────┬───────────────┘
                 │ HTTP REST (JSON / Multipart)              │
                 └───────────────────┬───────────────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │   FastAPI REST Service    │
                       │   (Port 8000)             │
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │   DefectPredictor Engine  │
                       │   (YOLOv8n - best.pt)     │
                       └───────────────────────────┘
```

1. **Embedded Mode (Standalone)**: Streamlit loads YOLO weights directly into memory for isolated single-machine inspection.
2. **Microservice Mode (Distributed)**: Streamlit (or factory HMI) sends image payloads over HTTP to the centralized FastAPI server.

---

## ⚙️ Configuration & Environment Variables

All settings can be configured via environment variables or a `.env` file (see `.env.example`):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `outputs/weights/best.pt` | Path to trained YOLO weights file |
| `DEVICE` | `cpu` | Compute device (`cpu`, `cuda`, `mps`) |
| `DEFAULT_CONFIDENCE` | `0.25` | Default detection confidence threshold (0.01 - 1.00) |
| `DEFAULT_IMGSZ` | `224` | Default input image resolution |
| `MAX_IMAGE_SIZE_MB` | `15` | Maximum allowed image upload size in MB |
| `API_HOST` | `0.0.0.0` | API server binding host |
| `API_PORT` | `8000` | API server binding port |
| `API_URL` | `http://localhost:8000` | Backend API URL used by the UI client |

---

## 📡 API Endpoints Reference

Base URL: `http://localhost:8000`  
Interactive OpenAPI/Swagger UI: `http://localhost:8000/docs`  
ReDoc UI: `http://localhost:8000/redoc`

---

### 1. Root & Discovery
- **Method**: `GET`
- **Path**: `/`
- **Description**: Returns service identity and available endpoint paths.
- **Response** (`200 OK`):
```json
{
  "service": "Industrial Surface Defect Detection API",
  "version": "1.0.0",
  "documentation": "/docs",
  "endpoints": {
    "health": "/health",
    "predict_json": "/predict",
    "predict_annotated_image": "/predict/image",
    "defect_classes": "/model/classes",
    "validation_metrics": "/model/metrics"
  }
}
```

---

### 2. Health & Readiness Check
- **Method**: `GET`
- **Path**: `/health`
- **Description**: Verifies service uptime, predictor initialization status, model path, and supported classes.
- **Response** (`200 OK`):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "d:/industrial-defect-detection_DL/outputs/weights/best.pt",
  "defect_classes": [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
  ],
  "uptime_seconds": 128.45
}
```

---

### 3. Defect Classes Catalog
- **Method**: `GET`
- **Path**: `/model/classes`
- **Description**: Returns all 6 surface defect categories with descriptions.
- **Response** (`200 OK`):
```json
{
  "count": 6,
  "classes": [
    {"id": 0, "name": "crazing", "description": "Network of fine surface cracks and micro-fissures"},
    {"id": 1, "name": "inclusion", "description": "Non-metallic embedded particles or impurities"},
    {"id": 2, "name": "patches", "description": "Irregular localized surface patches and discoloration"},
    {"id": 3, "name": "pitted_surface", "description": "Small cavities, indentations, or erosion pits"},
    {"id": 4, "name": "rolled-in_scale", "description": "Oxidized scale rolled directly into metal sheet"},
    {"id": 5, "name": "scratches", "description": "Long linear scratches or abrasive mechanical grooves"}
  ]
}
```

---

### 4. Ground-Truth Validation Metrics
- **Method**: `GET`
- **Path**: `/model/metrics`
- **Description**: Returns model evaluation results on the NEU validation dataset.
- **Response** (`200 OK`):
```json
{
  "num_classes": 6,
  "overall_metrics": {
    "precision": 0.6623,
    "recall": 0.7563,
    "mAP50": 0.7373,
    "mAP50_95": 0.4270
  },
  "per_class_metrics": { ... }
}
```

---

### 5. Defect Detection (JSON Detections)
- **Method**: `POST`
- **Path**: `/predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (*required*): Image binary (`.jpg`, `.png`, `.bmp`, `.webp`, `.tiff`).
  - `conf` (*optional*, float, default: `0.25`): Minimum detection confidence (0.01 - 1.00).
  - `imgsz` (*optional*, integer, default: `224`): Forward-pass image dimension.
  - `return_base64` (*optional*, boolean, default: `false`): Include base64-encoded annotated image in response.
- **Response** (`200 OK`):
```json
{
  "status": "success",
  "filename": "patches_1.jpg",
  "defect_detected": true,
  "detection_count": 2,
  "detected_classes": ["patches"],
  "detections": [
    {
      "class_id": 2,
      "class_name": "patches",
      "confidence": 0.8841,
      "confidence_pct": "88.4%",
      "bbox_xyxy": [12.45, 34.12, 110.8, 145.6],
      "width": 98.35,
      "height": 111.48,
      "area": 10964.06
    }
  ],
  "latency_ms": 38.45,
  "annotated_image_base64": null
}
```

---

### 6. Defect Detection (Direct Annotated JPEG)
- **Method**: `POST`
- **Path**: `/predict/image`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (*required*): Image binary.
  - `conf` (*optional*, float, default: `0.25`).
  - `imgsz` (*optional*, integer, default: `224`).
- **Response** (`200 OK`):
  - `Content-Type`: `image/jpeg`
  - Body: Raw binary JPEG stream with rendered bounding boxes, labels, and confidence tags.

---

## 🛡️ Error Handling Specifications

| HTTP Code | Error Condition | Detail Response Example |
| :---: | :--- | :--- |
| **`400 Bad Request`** | Empty file uploaded (0 bytes) | `{"detail": "Uploaded file is empty (0 bytes)."}` |
| **`400 Bad Request`** | Non-image or corrupted payload | `{"detail": "Uploaded file could not be parsed as a valid image: ..."}` |
| **`400 Bad Request`** | Unsupported MIME content type | `{"detail": "Invalid file type: 'application/pdf'. Must be an image ..."}` |
| **`404 Not Found`** | Evaluation metrics file missing | `{"detail": "Evaluation metrics report not found."}` |
| **`413 Payload Too Large`** | Image exceeds `MAX_IMAGE_SIZE_MB` | `{"detail": "Uploaded file exceeds maximum allowed size (15 MB)."}` |
| **`500 Internal Error`** | JPEG image encoding failure | `{"detail": "Failed to encode annotated output image."}` |
| **`503 Service Unavailable`** | Model weights (`best.pt`) not found | `{"detail": "Trained model weights not found at: ..."}` |

---

## 💻 Client Integration Examples

### Python (`requests`)
```python
import requests

url = "http://localhost:8000/predict"
params = {"conf": 0.25, "imgsz": 224, "return_base64": False}

with open("data/yolo/images/val/patches_1.jpg", "rb") as f:
    files = {"file": ("patches_1.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files, params=params)

if response.status_code == 200:
    data = response.json()
    print(f"Defects detected: {data['detection_count']}")
    for det in data["detections"]:
        print(f" - {det['class_name']}: {det['confidence_pct']} at {det['bbox_xyxy']}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict?conf=0.25" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/yolo/images/val/patches_1.jpg"
```
