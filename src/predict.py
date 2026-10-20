"""
YOLO Object Detection Inference and Evaluation Module for Industrial Defect Detection.

Supports:
- Single image inference from file path, PIL Image, or NumPy array.
- Directory batch inference for all images in a folder.
- Prediction image annotation with bounding boxes, labels, and confidence scores.
- Model evaluation on validation dataset with metrics export.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def resolve_path(path_input: Union[str, Path]) -> Path:
    """Resolve a path against cwd and project root."""
    p = Path(path_input)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    # Check relative to project root (2 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / path_input
    if candidate.exists():
        return candidate.resolve()
    return p.resolve()


class DefectPredictor:
    """
    Inference and evaluation handler for trained YOLO defect detection models.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "outputs/weights/best.pt",
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.model_path = resolve_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model weights not found at: {self.model_path}. "
                "Please verify that outputs/weights/best.pt exists before running inference."
            )

        print(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = self.model.names  # Dict {id: name}
        print(f"Model loaded successfully. Defect classes ({len(self.class_names)}): {self.class_names}")

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        output_dir: Optional[Union[str, Path]] = "outputs/predictions",
        conf: Optional[float] = None,
        save_annotated: bool = True,
        imgsz: int = 224,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image (path, PIL Image, or numpy array).

        Returns:
            Dict containing:
            - image_name: filename or identifier
            - image_path: path if loaded from disk, else None
            - saved_prediction: path of saved output image if saved
            - detection_count: number of defect bounding boxes detected
            - detections: list of detection dicts (class_id, class_name, confidence, bbox_xyxy)
            - annotated_image: RGB numpy array for direct display in Streamlit or OpenCV
            - speed: inference latency dictionary
        """
        confidence = conf if conf is not None else self.conf_threshold

        # Handle different input types
        img_name = "image_input"
        img_path_str = None
        source = image_input

        if isinstance(image_input, (str, Path)):
            p = resolve_path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")
            img_name = p.name
            img_path_str = str(p)
            source = str(p)
        elif isinstance(image_input, Image.Image):
            img_name = getattr(image_input, "filename", "uploaded_image.jpg")
            if img_name:
                img_name = Path(img_name).name
            else:
                img_name = "uploaded_image.jpg"
            source = image_input
        elif isinstance(image_input, np.ndarray):
            img_name = "array_image.jpg"
            source = image_input

        # Run YOLO inference
        results = self.model.predict(
            source=source,
            conf=confidence,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        # Extract bounding box detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                conf_val = float(box.conf[0].item())
                xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]

                # Calculate width and height
                xmin, ymin, xmax, ymax = xyxy
                width = round(xmax - xmin, 2)
                height = round(ymax - ymin, 2)

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "confidence_pct": f"{conf_val * 100:.1f}%",
                    "bbox_xyxy": xyxy,
                    "width": width,
                    "height": height,
                    "area": round(width * height, 2),
                })

        # Generate annotated image (plot() returns BGR numpy array)
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Save annotated image if requested
        saved_path = None
        if save_annotated and output_dir:
            out_dir = Path(output_dir)
            if not out_dir.is_absolute():
                out_dir = resolve_path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"pred_{img_name}"
            # Ensure valid extension
            if output_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                output_file = output_file.with_suffix(".jpg")

            cv2.imwrite(str(output_file), annotated_bgr)
            saved_path = str(output_file)

        return {
            "image_name": img_name,
            "image_path": img_path_str,
            "saved_prediction": saved_path,
            "detection_count": len(detections),
            "detections": detections,
            "annotated_image": annotated_rgb,
            "speed": result.speed if hasattr(result, "speed") else {},
        }

    def predict_source(
        self,
        source_path: Union[str, Path],
        output_dir: Union[str, Path] = "outputs/predictions",
        conf: Optional[float] = None,
        imgsz: int = 224,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single image file or an entire directory of images.
        """
        src = resolve_path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source path not found: {src}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        if src.is_file():
            image_files = [src]
        elif src.is_dir():
            image_files = sorted([p for p in src.iterdir() if p.suffix.lower() in valid_extensions])
        else:
            raise ValueError(f"Invalid source path: {src}")

        if not image_files:
            print(f"No valid image files found in {src}")
            return []

        out_path = Path(output_dir)
        if not out_path.is_absolute():
            out_path = resolve_path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {len(image_files)} image(s) from: {src}")
        all_results = []
        for idx, img_file in enumerate(image_files, 1):
            res = self.predict_image(
                image_input=img_file,
                output_dir=out_path,
                conf=conf,
                save_annotated=True,
                imgsz=imgsz,
            )
            all_results.append(res)
            det_summary = (
                ", ".join([f"{d['class_name']} ({d['confidence']:.2f})" for d in res["detections"]])
                if res["detections"]
                else "No defects detected"
            )
            print(f"[{idx}/{len(image_files)}] {img_file.name} -> {res['detection_count']} detections [{det_summary}]")

        print(f"\nAll prediction images saved to: {out_path}")
        return all_results

    def evaluate(
        self,
        data_yaml: Union[str, Path] = "data/yolo/data.yaml",
        output_dir: Union[str, Path] = "outputs/evaluation",
        imgsz: int = 224,
        batch: int = 16,
    ) -> Dict[str, Any]:
        """
        Run validation on the dataset defined in data.yaml and generate an evaluation report.
        """
        yaml_path = resolve_path(data_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset configuration YAML not found: {yaml_path}")

        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = resolve_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(" RUNNING MODEL EVALUATION ON VALIDATION DATASET ")
        print("=" * 60)
        print(f"Model:     {self.model_path}")
        print(f"Data YAML: {yaml_path}")
        print(f"Img Size:  {imgsz}")

        # Run validation
        val_results = self.model.val(
            data=str(yaml_path),
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=str(out_dir),
            name="val_artifacts",
            exist_ok=True,
            plots=True,
            verbose=True,
        )

        precision = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
        recall = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
        map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

        # Inspect validation image count
        val_img_dir = yaml_path.parent / "images" / "val"
        val_images_count = len(list(val_img_dir.glob("*.jpg"))) if val_img_dir.exists() else 0

        # Build class-level metrics safely
        per_class_metrics = {}
        if hasattr(val_results, "box"):
            box = val_results.box
            p_arr = getattr(box, "p", None)
            r_arr = getattr(box, "r", None)
            maps_arr = getattr(box, "maps", None)
            ap50_arr = getattr(box, "ap50", None)
            all_ap = getattr(box, "all_ap", None)

            for idx, cname in self.class_names.items():
                p = float(p_arr[idx]) if p_arr is not None and hasattr(p_arr, "__len__") and len(p_arr) > idx else None
                r = float(r_arr[idx]) if r_arr is not None and hasattr(r_arr, "__len__") and len(r_arr) > idx else None
                m50_95 = float(maps_arr[idx]) if maps_arr is not None and hasattr(maps_arr, "__len__") and len(maps_arr) > idx else None

                m50 = None
                if ap50_arr is not None and hasattr(ap50_arr, "__len__") and len(ap50_arr) > idx:
                    m50 = float(ap50_arr[idx])
                elif all_ap is not None and hasattr(all_ap, "__len__") and len(all_ap) > idx:
                    try:
                        m50 = float(all_ap[idx][0])
                    except (IndexError, TypeError):
                        m50 = None

                per_class_metrics[cname] = {
                    "precision": round(p, 4) if p is not None else None,
                    "recall": round(r, 4) if r is not None else None,
                    "mAP50": round(m50, 4) if m50 is not None else None,
                    "mAP50_95": round(m50_95, 4) if m50_95 is not None else None,
                }

        report = {
            "model_path": str(self.model_path),
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values()),
            "validation_image_count": val_images_count,
            "overall_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
            },
            "per_class_metrics": per_class_metrics,
        }

        # Save JSON report
        metrics_json_path = out_dir / "evaluation_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        # Save Markdown report
        markdown_report_path = out_dir / "evaluation_report.md"
        with open(markdown_report_path, "w", encoding="utf-8") as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"- **Model Path**: `{report['model_path']}`\n")
            f.write(f"- **Number of Classes**: {report['num_classes']}\n")
            f.write(f"- **Class Names**: {', '.join(report['class_names'])}\n")
            f.write(f"- **Validation Image Count**: {report['validation_image_count']}\n\n")
            f.write("## Overall Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **Precision (B)** | {precision:.4f} |\n")
            f.write(f"| **Recall (B)** | {recall:.4f} |\n")
            f.write(f"| **mAP@50 (B)** | {map50:.4f} |\n")
            f.write(f"| **mAP@50-95 (B)** | {map50_95:.4f} |\n\n")
            f.write("## Per-Class Performance\n\n")
            f.write("| Class Name | Precision | Recall | mAP@50 | mAP@50-95 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for cname, m in per_class_metrics.items():
                p_str = f"{m['precision']:.4f}" if m['precision'] is not None else "N/A"
                r_str = f"{m['recall']:.4f}" if m['recall'] is not None else "N/A"
                m50_str = f"{m['mAP50']:.4f}" if m['mAP50'] is not None else "N/A"
                m50_95_str = f"{m['mAP50_95']:.4f}" if m['mAP50_95'] is not None else "N/A"
                f.write(f"| `{cname}` | {p_str} | {r_str} | {m50_str} | {m50_95_str} |\n")

        print(f"\nEvaluation metrics saved to: {metrics_json_path}")
        print(f"Evaluation report saved to:  {markdown_report_path}")

        print("\n" + "=" * 60)
        print(" EVALUATION SUMMARY ")
        print("=" * 60)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"mAP@50:    {map50:.4f}")
        print(f"mAP@50-95: {map50_95:.4f}")
        print("=" * 60)

        return report


def main():
    parser = argparse.ArgumentParser(description="YOLO Defect Detection Inference & Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="outputs/weights/best.pt", help="Path to best.pt weights")
    parser.add_argument("--source", type=str, default=None, help="Path to single image or image directory for inference")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="Directory to save predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on validation dataset")
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml", help="Path to data.yaml for evaluation")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image resolution")
    args = parser.parse_args()

    predictor = DefectPredictor(model_path=args.model, conf_threshold=args.conf)

    if args.evaluate:
        predictor.evaluate(data_yaml=args.data, output_dir="outputs/evaluation", imgsz=args.imgsz)

    if args.source:
        predictor.predict_source(source_path=args.source, output_dir=args.output, conf=args.conf, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
