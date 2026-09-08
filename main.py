import os
import io
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import gdown

from fertilizer_logic import recommend_fertilizer
from disease_cure import get_cure
from irrigation_model import get_weather


# ==============================
# MODEL PATHS
# ==============================

LEAF_MODEL_PATH = "leaf_model.pth"
CLASSES_PATH = "classes.pth"

CROP_MODEL_PATH = "crop_model.pkl"
YIELD_MODEL_PATH = "yield_model.pkl"


# Google Drive IDs for large files
LEAF_MODEL_ID = "1w5F68buoOIMjuDntzn2-Pv1k2lD6ZY-y"
CLASSES_ID = "1YGZu2J0dYTQCYeWKTFrxyNPx_lPsxkQM"


# ==============================
# DOWNLOAD LARGE MODELS
# ==============================

def download_google_drive_model(file_id: str, output: str):

    # If Render already has a valid file, don't download again.
    if os.path.exists(output) and os.path.getsize(output) > 0:
        print(f"✅ {output} already exists. Skipping download.")
        return

    print(f"⬇️ Downloading {output} from Google Drive...")

    try:
        downloaded = gdown.download(
            id=file_id,
            output=output,
            quiet=False
        )

        if not downloaded:
            raise RuntimeError(
                f"gdown failed to download {output}"
            )

        if not os.path.exists(output):
            raise RuntimeError(
                f"{output} was not created"
            )

        if os.path.getsize(output) == 0:
            raise RuntimeError(
                f"{output} is empty after download"
            )

        print(
            f"✅ {output} downloaded successfully "
            f"({os.path.getsize(output)} bytes)"
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to download {output} from Google Drive. "
            f"Make sure the Google Drive file is shared as "
            f"'Anyone with the link' -> 'Viewer'. "
            f"Also verify the file ID. "
            f"Original error: {e}"
        ) from e


# ONLY the two large models are downloaded.
download_google_drive_model(
    LEAF_MODEL_ID,
    LEAF_MODEL_PATH
)

download_google_drive_model(
    CLASSES_ID,
    CLASSES_PATH
)


# ==============================
# CHECK LOCAL MODELS
# ==============================

for required_file in [
    CROP_MODEL_PATH,
    YIELD_MODEL_PATH,
]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(
            f"Required local model file is missing: "
            f"{required_file}"
        )


# ==============================
# INIT APP
# ==============================

app = FastAPI(
    title="AgriAI API",
    description="Smart Farming AI API",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# LOAD MODELS
# ==============================

print("🔄 Loading models...")


crop_model = joblib.load(
    CROP_MODEL_PATH
)


yield_model = joblib.load(
    YIELD_MODEL_PATH
)


classes = torch.load(
    CLASSES_PATH,
    map_location="cpu"
)


if isinstance(classes, torch.Tensor):
    classes = classes.tolist()


model = models.resnet18(
    weights=None
)


model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)


model.load_state_dict(
    torch.load(
        LEAF_MODEL_PATH,
        map_location="cpu"
    )
)


model.eval()


# ==============================
# IMAGE TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


print("✅ All models loaded successfully!")
