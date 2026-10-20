"""
Reproducible ONNX Model Export Script for Industrial Surface Defect Detection.

Exports trained YOLOv8 weights (outputs/weights/best.pt) to ONNX format
for edge inference and cross-platform deployment.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(
    weights_path: str = "outputs/weights/best.pt",
    output_dir: str = "outputs/models",
    imgsz: int = 224,
) -> Path:
    """Export YOLO weights to ONNX format and place in output_dir."""
    project_root = Path(__file__).resolve().parent.parent
    weights_file = (project_root / weights_path).resolve()
    target_dir = (project_root / output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not weights_file.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_file}")

    print(f"Loading weights from: {weights_file}")
    model = YOLO(str(weights_file))

    print(f"Exporting model to ONNX (resolution: {imgsz}x{imgsz})...")
    exported_path_str = model.export(format="onnx", imgsz=imgsz, dynamic=False)
    exported_file = Path(exported_path_str).resolve()

    target_file = target_dir / exported_file.name
    if exported_file != target_file:
        shutil.copy2(exported_file, target_file)
        print(f"Saved ONNX model to: {target_file}")

    file_size_bytes = target_file.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    print("\n" + "=" * 60)
    print(" ONNX EXPORT SUMMARY ")
    print("=" * 60)
    print(f"Source Weights: {weights_file}")
    print(f"Export Path:    {target_file}")
    print(f"File Size:      {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
    print("=" * 60)

    return target_file


if __name__ == "__main__":
    export_to_onnx()
