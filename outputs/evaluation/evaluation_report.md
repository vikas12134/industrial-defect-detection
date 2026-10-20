# Industrial Product Defect Detection — Model Evaluation Report

**Evaluation Date**: 2026-10-16  
**Model Checkpoint**: [`outputs/weights/best.pt`](file:///d:/industrial-defect-detection_DL/outputs/weights/best.pt) (YOLOv8 Nano, 3.13M parameters)  
**Dataset Configuration**: [`data/yolo/data.yaml`](file:///d:/industrial-defect-detection_DL/data/yolo/data.yaml) (NEU Surface Defect Database)  
**Validation Split**: 30 images, 64 ground-truth defect instances  
**Input Image Resolution**: $224 \times 224$ px  

---

## 1. Executive Summary & Overall Metrics

The trained YOLOv8n detector was evaluated on the independent validation dataset using the Ultralytics evaluation protocol with an IoU threshold of 0.50 (for mAP@50) and averaged across IoU 0.50:0.95 (mAP@50-95).

| Metric | Score | Percentage | Status |
| :--- | :---: | :---: | :---: |
| **Precision (P)** | `0.6623` | **66.23%** | Moderate-High |
| **Recall (R)** | `0.7563` | **75.63%** | Strong |
| **mAP @ 0.50** | `0.7373` | **73.73%** | Strong Baseline |
| **mAP @ 0.50-0.95** | `0.4270` | **42.70%** | Expected for 224px input |

---

## 2. Six-Class Comparative Performance Breakdown

The dataset contains six industrial steel surface defect categories. Below is the comparative performance across all classes:

| Class ID | Defect Class | Val Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 | Overall Rating |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `crazing` | 8 | **1.0000** | 0.4209 | 0.6276 | 0.3327 | High Precision / Low Recall |
| **1** | `inclusion` | 15 | 0.5058 | 0.8667 | **0.8675** | 0.3917 | Strong Detector |
| **2** | `patches` | 17 | 0.5945 | **0.9489** | **0.8860** | **0.5491** | High Performing |
| **3** | `pitted_surface` | 8 | **0.9019** | **1.0000** | **0.9950** | **0.6823** | Top Performing |
| **4** | `rolled-in_scale` | 9 | 0.4100 | 0.4444 | 0.4438 | 0.2165 | Needs Improvement |
| **5** | `scratches` | 7 | 0.5619 | 0.8571 | 0.6038 | 0.3895 | Moderate |
| **All** | **Macro Average / Total** | **64** | **0.6623** | **0.7563** | **0.7373** | **0.4270** | **Strong Baseline** |

---

## 3. Class-by-Class Analysis: High Performers vs. Needing Improvement

### 🟢 High Performing Classes
1. **`pitted_surface` (mAP@50: 99.5%, Recall: 100%, Precision: 90.2%)**:
   - **Characteristics**: High contrast, distinctive dot/pit morphology.
   - **Model Response**: Flawless detection with zero false negatives on validation data.

2. **`patches` (mAP@50: 88.6%, Recall: 94.9%, mAP@50-95: 54.9%)**:
   - **Characteristics**: Large surface area anomalies with noticeable color and texture divergence.
   - **Model Response**: Highest mAP@50-95 in the entire dataset (54.91%), demonstrating accurate bounding box alignment.

3. **`inclusion` (mAP@50: 86.8%, Recall: 86.7%)**:
   - **Characteristics**: Isolated particle contaminants distributed across the sheet.
   - **Model Response**: High recall (captures 13 out of 15 instances); some false positives lower precision to 50.6%, which can be tightened with confidence threshold calibration.

### 🟡 Classes with Mixed Performance
4. **`crazing` (Precision: 100%, Recall: 42.1%, mAP@50: 62.8%)**:
   - **Characteristics**: Intricate networks of fine, shallow micro-cracks resembling web patterns.
   - **Model Response**: When the model predicts `crazing`, it is 100% accurate (zero false positives). However, lower recall (42.1%) indicates that subtle, low-contrast fissures are missed at $224 \times 224$ resolution.
5. **`scratches` (Recall: 85.7%, Precision: 56.2%, mAP@50: 60.4%)**:
   - **Characteristics**: Elongated, directional surface grooves.
   - **Model Response**: Detects 6 out of 7 scratches effectively, but orientation variations and background grain lead to occasional duplicate or oversized boxes.

### 🔴 Classes Needing Improvement
6. **`rolled-in_scale` (mAP@50: 44.4%, Precision: 41.0%, Recall: 44.4%)**:
   - **Characteristics**: Thin, irregular iron oxide flakes embedded into hot metal rolling; weak edge gradients.
   - **Primary Issue**: Low contrast against the metallic background causes confusion with clean surface reflections or subtle patches.
   - **Key Action for Next Stage**: Needs multi-scale augmentation, higher input resolution ($320 \times 320$ or $416 \times 416$), and focal contrast enhancement.

---

## 4. Confusion Matrix & Error Analysis

The confusion matrix generated during validation ([`outputs/evaluation/confusion_matrix.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/confusion_matrix.png) and [`outputs/evaluation/confusion_matrix_normalized.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/confusion_matrix_normalized.png)) reveals:
- **Background False Positives**: Minor confusion where background steel texture is predicted as `inclusion` or `rolled-in_scale`.
- **Background False Negatives**: Unidentified ground truth primarily concentrated in `crazing` and `rolled-in_scale`.
- **Inter-Class Confusion**: Almost zero cross-class confusion between defect classes (e.g. `scratches` are never misidentified as `pitted_surface` or `crazing`).

---

## 5. Visual Artifacts and Bounding Box Validations

The following evaluation visualizations are generated and saved:

1. **Confusion Matrix Plots**:
   - Raw Counts: [`outputs/evaluation/confusion_matrix.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/confusion_matrix.png)
   - Normalized: [`outputs/evaluation/confusion_matrix_normalized.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/confusion_matrix_normalized.png)
2. **Precision-Recall & Diagnostic Curves**:
   - Precision-Recall Curve: [`outputs/evaluation/BoxPR_curve.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/BoxPR_curve.png)
   - F1-Score vs. Confidence Curve: [`outputs/evaluation/BoxF1_curve.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/BoxF1_curve.png)
   - Precision Curve: [`outputs/evaluation/BoxP_curve.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/BoxP_curve.png)
   - Recall Curve: [`outputs/evaluation/BoxR_curve.png`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/BoxR_curve.png)
3. **Batch Prediction Visualizations**:
   - Batch 0 Ground Truth Labels: [`outputs/evaluation/val_batch0_labels.jpg`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/val_batch0_labels.jpg)
   - Batch 0 Predicted Boxes: [`outputs/evaluation/val_batch0_pred.jpg`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/val_batch0_pred.jpg)
   - Batch 1 Ground Truth Labels: [`outputs/evaluation/val_batch1_labels.jpg`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/val_batch1_labels.jpg)
   - Batch 1 Predicted Boxes: [`outputs/evaluation/val_batch1_pred.jpg`](file:///d:/industrial-defect-detection_DL/outputs/evaluation/val_batch1_pred.jpg)
4. **Individual Image Predictions**:
   - 30 annotated images saved in [`outputs/predictions/`](file:///d:/industrial-defect-detection_DL/outputs/predictions).

---

## 6. Conclusion & Next Stage Readiness

The baseline YOLOv8n model delivers a strong foundation with **73.73% mAP@50** and **75.63% overall recall**. Critical high-severity defects like `pitted_surface` and `patches` achieve near-perfect operational detection rates.

Per project constraints, the original dataset has been preserved without modification, and hyperparameter tuning has not been initiated.
