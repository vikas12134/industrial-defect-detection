"""
YOLO Object Detection Training Script for Industrial Product Defect Detection.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def train_yolo(
    data_yaml: str = "data/yolo/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 224,
    batch_size: int = 16,
    project_dir: str = "outputs",
    run_name: str = "train_yolo",
    device: str = None,
):
    """
    Train YOLO object detection model on industrial defect dataset.
    """
    root_dir = Path(__file__).resolve().parent.parent
    data_yaml_path = (root_dir / data_yaml).resolve()
    project_path = (root_dir / project_dir).resolve()
    weights_target_dir = project_path / "weights"
    weights_target_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(" Industrial Defect Detection - YOLO Training ")
    print("=" * 60)
    print(f"Data YAML:    {data_yaml_path}")
    print(f"Model:        {model_name}")
    print(f"Device:       {device}")
    print(f"Epochs:       {epochs}")
    print(f"Image Size:   {imgsz}")
    print(f"Batch Size:   {batch_size}")
    print(f"Project Dir:  {project_path}")
    print("=" * 60)

    # 1. Initialize YOLO model
    print(f"\n[1/3] Loading pretrained model: {model_name}...")
    model = YOLO(model_name)

    # 2. Train model
    print(f"\n[2/3] Training model for {epochs} epochs...")
    train_results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(project_path),
        name=run_name,
        exist_ok=True,
        workers=2,
        plots=True,
        save=True,
        verbose=True,
    )
    print("\nTraining completed successfully!")

    # 3. Locate and copy best weights
    run_dir = project_path / run_name
    best_weights_src = run_dir / "weights" / "best.pt"
    target_best_pt = weights_target_dir / "best.pt"

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, target_best_pt)
        print(f"Copied best weights to: {target_best_pt}")
    else:
        last_weights_src = run_dir / "weights" / "last.pt"
        if last_weights_src.exists():
            shutil.copy2(last_weights_src, target_best_pt)
            print(f"Copied last weights to: {target_best_pt}")
        else:
            print("Warning: Neither best.pt nor last.pt was found in training run.")

    # 4. Independent Validation
    print("\n[3/3] Running validation on validation dataset...")
    val_model_weights = str(target_best_pt) if target_best_pt.exists() else model_name
    val_model = YOLO(val_model_weights)

    val_results = val_model.val(
        data=str(data_yaml_path),
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(project_path),
        name="val_yolo",
        exist_ok=True,
        plots=True,
    )

    # 5. Extract and format metrics
    precision = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    recall = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    metrics = {
        "status": "completed",
        "model_used": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "best_weights": str(target_best_pt),
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "fitness": float(val_results.fitness) if hasattr(val_results, "fitness") else None,
    }

    metrics_file = project_path / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nTraining metrics saved to: {metrics_file}")

    print("\n" + "=" * 60)
    print(" TRAINING & VALIDATION SUMMARY ")
    print("=" * 60)
    print(f"Status:       {metrics['status']}")
    print(f"Model:        {metrics['model_used']}")
    print(f"Epochs:       {metrics['epochs']}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"mAP@50:       {map50:.4f}")
    print(f"mAP@50-95:    {map50_95:.4f}")
    print(f"Best Weights: {target_best_pt}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Object Detector for Defect Detection")
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model name/path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (e.g. 224, 320, 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="outputs", help="Output project directory")
    parser.add_argument("--name", type=str, default="train_yolo", help="Run name")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        project_dir=args.project,
        run_name=args.name,
        device=args.device,
    )
