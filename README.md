# Industrial Defect Detection

## Overview
This project uses a YOLO-based computer vision model to detect and localize industrial surface defects in images using bounding boxes and confidence scores. It provides an automated, visual quality inspection workflow through an interactive web interface.

## Features
- Industrial defect detection
- YOLO object detection
- Bounding-box localization
- Confidence scores
- Defect class identification
- Streamlit web interface
- Sample image testing

## Tech Stack
- **Language**: Python
- **Model**: YOLO (Ultralytics / PyTorch)
- **Computer Vision**: OpenCV, Pillow
- **Web Interface**: Streamlit

## How It Works
Image → YOLO model → defect detection → bounding boxes/confidence → Streamlit results

1. **Input**: A component surface image is uploaded or selected from built-in sample images.
2. **Inference**: The custom-trained YOLO model inspects the image.
3. **Detection**: Defect instances are identified with bounding box coordinates and confidence levels.
4. **Display**: Streamlit displays the annotated detection image, overall metrics, and individual defect details.

## Project Structure
```text
industrial-defect-detection/
├── app/
│   └── main.py          # Streamlit web application
├── data/
│   └── yolo/            # Dataset images and labels
├── outputs/
│   └── weights/
│       └── best.pt      # Trained YOLO model weights
├── src/
│   ├── predict.py       # Inference and prediction engine
│   └── train.py         # YOLO training script
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Run Locally

```bash
# Clone the repository
git clone https://github.com/vikas12134/industrial-defect-detection.git
cd industrial-defect-detection

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit application
streamlit run app/main.py
```

## Live Demo
Access the deployed Streamlit application here:  
👉 [Live Demo](https://industrial-defect-detection-4vmwlqdoxee8e6pr6v6tqm.streamlit.app/)

## Results
The application performs real-time inspection and displays:
- Total detected defect instances
- Unique defect classes
- Highest confidence
- Individual defect class
- Confidence percentage
- Bounding box coordinates (`[xmin, ymin, xmax, ymax]`)

## GitHub
- Repository: [https://github.com/vikas12134/industrial-defect-detection](https://github.com/vikas12134/industrial-defect-detection)
