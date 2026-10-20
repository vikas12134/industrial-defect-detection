"""
Comparison script to verify ONNX model inference against original PyTorch model.
"""

import json
import time
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent

PT_MODEL_PATH = ROOT_DIR / "outputs" / "weights" / "best.pt"
ONNX_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "best.onnx"
IMAGE_PATH = ROOT_DIR / "data" / "yolo" / "images" / "val" / "patches_1.jpg"


def run_benchmark(model_path: Path, image_path: Path, conf: float = 0.25, imgsz: int = 224):
    print(f"Loading {model_path.name}...")
    model = YOLO(str(model_path), task="detect")

    # Warmup
    _ = model(str(image_path), conf=conf, imgsz=imgsz, verbose=False)

    # Timed runs (average of 3)
    timings = []
    result = None
    for _ in range(3):
        t0 = time.perf_counter()
        results = model(str(image_path), conf=conf, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
        result = results[0]

    avg_latency_ms = sum(timings) / len(timings)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf_val = float(box.conf[0].item())
            xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf_val, 4),
                "bbox_xyxy": xyxy,
            })

    return {
        "model": model_path.name,
        "format": "PyTorch (.pt)" if model_path.suffix == ".pt" else "ONNX (.onnx)",
        "detection_count": len(detections),
        "detections": detections,
        "avg_latency_ms": round(avg_latency_ms, 2),
    }


def main():
    pt_results = run_benchmark(PT_MODEL_PATH, IMAGE_PATH)
    onnx_results = run_benchmark(ONNX_MODEL_PATH, IMAGE_PATH)

    output = {
        "image": IMAGE_PATH.name,
        "pytorch": pt_results,
        "onnx": onnx_results,
    }
    print("\n--- BENCHMARK RESULTS ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
