"""
Centralized Configuration Module for Industrial Surface Defect Detection.
Provides environment-variable overrides with safe production defaults.
"""

import os
from pathlib import Path
from typing import Dict, List

# Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Model Configuration
DEFAULT_WEIGHTS = ROOT_DIR / "outputs" / "weights" / "best.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_WEIGHTS))).resolve()
DEVICE = os.getenv("DEVICE", "cpu")

# Inference Defaults
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "224"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "15"))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# API Server Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Streamlit UI Configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Defect Classes Metadata (NEU Surface Defect Database)
DEFECT_CLASSES: Dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

DEFECT_DESCRIPTIONS: Dict[str, str] = {
    "crazing": "Network of fine surface cracks and micro-fissures",
    "inclusion": "Non-metallic embedded particles or impurities",
    "patches": "Irregular localized surface patches and discoloration",
    "pitted_surface": "Small cavities, indentations, or erosion pits",
    "rolled-in_scale": "Oxidized scale rolled directly into metal sheet",
    "scratches": "Long linear scratches or abrasive mechanical grooves",
}
