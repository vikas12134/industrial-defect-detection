"""
Integration and Unit Tests for Industrial Defect Detection FastAPI Service.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Industrial Surface Defect Detection API"
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["defect_classes"]) == 6
    assert "pitted_surface" in data["defect_classes"]


def test_model_classes_endpoint():
    response = client.get("/model/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 6
    class_names = [c["name"] for c in data["classes"]]
    assert "crazing" in class_names
    assert "scratches" in class_names


def test_model_metrics_endpoint():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "overall_metrics" in data or "precision" in data


def test_predict_endpoint_valid_image():
    sample_image_path = ROOT_DIR / "data" / "yolo" / "images" / "val" / "patches_1.jpg"
    assert sample_image_path.exists(), f"Sample image not found: {sample_image_path}"

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("patches_1.jpg", f, "image/jpeg")},
            params={"conf": 0.25, "imgsz": 224, "return_base64": True},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "patches_1.jpg"
    assert "detections" in data
    assert "latency_ms" in data
    assert data["annotated_image_base64"] is not None
    assert len(data["annotated_image_base64"]) > 100


def test_predict_image_endpoint():
    sample_image_path = ROOT_DIR / "data" / "yolo" / "images" / "val" / "inclusion_1.jpg"
    assert sample_image_path.exists()

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/predict/image",
            files={"file": ("inclusion_1.jpg", f, "image/jpeg")},
            params={"conf": 0.25},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000  # Non-trivial image bytes


def test_predict_empty_file_error():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_predict_invalid_image_content():
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", b"NOT_AN_IMAGE_STRING", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "valid image" in data["detail"].lower() or "unidentified" in data["detail"].lower()


if __name__ == "__main__":
    print("Running integration tests for FastAPI defect detection service...")
    test_root_endpoint()
    print("  [PASS] GET /")
    test_health_endpoint()
    print("  [PASS] GET /health")
    test_model_classes_endpoint()
    print("  [PASS] GET /model/classes")
    test_model_metrics_endpoint()
    print("  [PASS] GET /model/metrics")
    test_predict_endpoint_valid_image()
    print("  [PASS] POST /predict (valid image)")
    test_predict_image_endpoint()
    print("  [PASS] POST /predict/image (JPEG visual output)")
    test_predict_empty_file_error()
    print("  [PASS] POST /predict (empty file error handling)")
    test_predict_invalid_image_content()
    print("  [PASS] POST /predict (corrupt file error handling)")
    print("\nAll 8 FastAPI integration tests passed successfully!")
