# Industrial Surface Defect Detection using YOLOv8

## 📌 Description
An end-to-end computer vision project that automates visual quality inspection and defect localization on industrial metal surfaces using a custom-trained **YOLOv8** deep learning model. The system accurately identifies, classifies, and draws bounding boxes around surface defects in real-time through an interactive Streamlit web application.

---

## ✨ Key Features
- **Real-Time Defect Detection**: Localizes and classifies surface flaws with high accuracy.
- **Side-by-Side Visualization**: Displays original component images alongside AI-annotated predictions.
- **Configurable Confidence**: Interactive threshold slider for real-time sensitivity tuning.
- **Detailed Inspection Table**: Reports defect class, confidence score, and bounding-box coordinates for each detected instance.
- **Clear Pass/Fail Indicators**: Prominently displays *"No defect detected"* when the component passes quality inspection.

---

## 🏷️ Detected Defect Classes
1. **`crazing`**: Network of fine surface cracks and micro-fissures
2. **`inclusion`**: Non-metallic embedded particles or impurities
3. **`patches`**: Localized surface irregularities and discoloration
4. **`pitted_surface`**: Small cavities, indentations, or erosion pits
5. **`rolled-in_scale`**: Oxidized mill scale rolled into sheet metal
6. **`scratches`**: Linear abrasive scratches or mechanical grooves

---

## 🛠️ Technologies Used
- **Deep Learning**: YOLOv8 (Ultralytics), PyTorch
- **Computer Vision & Image Processing**: OpenCV, Pillow
- **Data Analysis**: NumPy, Pandas
- **Web Framework**: Streamlit
- **Language**: Python 3.10+

---

## 📁 Project Structure
```text
industrial-defect-detection_DL/
├── app/
│   └── main.py              # Streamlit web application
├── data/
│   └── yolo/                # Dataset (train/val images, labels, data.yaml)
├── outputs/
│   ├── weights/
│   │   └── best.pt          # Trained YOLOv8 model weights
│   └── plots/               # Training curves & confusion matrices
├── src/
│   ├── predict.py           # Inference & evaluation engine
│   └── train.py             # Model training script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📦 Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-username/industrial-defect-detection_DL.git
cd industrial-defect-detection_DL

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501` to start inspecting component surface images.

---

## 📊 Example Detection Result
- **Sample Image**: `patches_1.jpg`
- **Inspection Status**: ⚠️ Defects Detected
- **Total Defect Count**: 3 instances
- **Detected Class**: `patches`
- **Highest Confidence**: **91.6%** (`0.9158`)
- **Bounding Box Coordinates**: `[47.16, 11.21, 107.61, 94.22]`
