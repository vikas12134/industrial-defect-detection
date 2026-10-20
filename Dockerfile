# Industrial Surface Defect Detection - Production Dockerfile
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install OS dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY outputs/weights/ ./outputs/weights/
COPY outputs/evaluation/ ./outputs/evaluation/
COPY outputs/plots/ ./outputs/plots/
COPY outputs/train_yolo/ ./outputs/train_yolo/
COPY data/yolo/data.yaml ./data/yolo/data.yaml
COPY data/yolo/images/val/ ./data/yolo/images/val/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: launch production FastAPI service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
