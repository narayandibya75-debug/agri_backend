# ==============================
# IMPORTS
# ==============================
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
# DOWNLOAD MODELS
# ==============================

def download_model(file_id, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url,
            output,
            quiet=False
        )


download_model(
    "1w5F68buoOIMjuDntzn2-Pv1k2lD6ZY-y",
    "leaf_model.pth"
)

download_model(
    "1YGZu2J0dYTQCYeWKTFrxyNPx_lPsxkQM",
    "classes.pth"
)

download_model(
    "1cZvhsTBkXqij2mqIhvK7MDosfOQjCba1",
    "crop_model.pkl"
)

download_model(
    "1LOEAAL4nzUNb_X5S1B7HaQX5ZZZeyHxG",
    "yield_model.pkl"
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


# Crop model
crop_model = joblib.load("crop_model.pkl")


# Yield model
yield_model = joblib.load("yield_model.pkl")


# Disease classes
classes = torch.load(
    "classes.pth",
    map_location="cpu"
)


# Make sure classes is a normal Python list
if isinstance(classes, torch.Tensor):
    classes = classes.tolist()


# Disease model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)


model.load_state_dict(
    torch.load(
        "leaf_model.pth",
        map_location="cpu"
    )
)

model.eval()


# ==============================
# IMAGE TRANSFORM
# ==============================

# IMPORTANT:
# This MUST match the transformations used during training.

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


print("✅ All models loaded successfully!")


# ==============================
# HOME
# ==============================

@app.get("/")
def home():

    return {
        "message": "AgriAI Running 🚀"
    }


# ==============================
# 📡 LIVE SOIL SENSOR (ESP32 / RS485)
# ==============================
#
# Confirmed live registers on the ZTS-3002-TR-ECTHNPKPH-N01: moisture,
# temperature, EC, pH. N/P/K registers on this sensor are documented as
# temporary/read-write values and must NOT be treated as verified live
# measurements — we store and pass through whatever the ESP32 actually
# sends, and NEVER invent, zero-fill, or fall back to default numbers.
# Rainfall is not measured by this sensor at all; it stays null unless a
# separate source explicitly supplies it.

class SensorPayload(BaseModel):
    moisture: float
    temperature: float
    ec: float
    ph: float
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    rainfall: Optional[float] = None


_sensor_lock = threading.Lock()
_latest_sensor_data = {
    "moisture": None,
    "temperature": None,
    "ec": None,
    "ph": None,
    "N": None,
    "P": None,
    "K": None,
    "rainfall": None,
    "updated_at": None,
    "source": "ESP32 / RS485 soil sensor",
}


@app.post("/sensor_data")
def post_sensor_data(payload: SensorPayload):

    with _sensor_lock:
        _latest_sensor_data.update({
            "moisture": payload.moisture,
            "temperature": payload.temperature,
            "ec": payload.ec,
            "ph": payload.ph,
            "N": payload.N,
            "P": payload.P,
            "K": payload.K,
            "rainfall": payload.rainfall,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "ESP32 / RS485 soil sensor",
        })
        snapshot = dict(_latest_sensor_data)

    return {
        "status": "ok",
        "data": snapshot
    }


@app.get("/sensor_data")
def get_sensor_data():

    with _sensor_lock:
        snapshot = dict(_latest_sensor_data)

    return {
        "status": "ok",
        "data": snapshot
    }


# ==============================
# 🌾 CROP PREDICTION
# ==============================

@app.get("/predict_crop")
def predict_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
):

    # Basic validation
    if N < 0 or P < 0 or K < 0:
        return {
            "error": "N, P and K cannot be negative"
        }

    if humidity < 0 or humidity > 100:
        return {
            "error": "Humidity must be between 0 and 100"
        }

    if ph < 0 or ph > 14:
        return {
            "error": "pH must be between 0 and 14"
        }

    if rainfall < 0:
        return {
            "error": "Rainfall cannot be negative"
        }


    df = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])


    prediction = crop_model.predict(df)


    return {
        "crop": str(prediction[0])
    }


# ==============================
# 🌱 FERTILIZER
# ==============================

@app.get("/fertilizer")
def fertilizer(
    N: float,
    P: float,
    K: float,
    ph: float
):

    if N < 0 or P < 0 or K < 0:
        return {
            "error": "N, P and K cannot be negative"
        }

    if ph < 0 or ph > 14:
        return {
            "error": "pH must be between 0 and 14"
        }


    result = recommend_fertilizer(
        N,
        P,
        K,
        ph
    )


    return {
        "fertilizer": result
    }


# ==============================
# 📈 YIELD PREDICTION
# ==============================

@app.get("/predict_yield")
def predict_yield(
    temperature: float,
    rainfall: float,
    ph: float,
    N: float,
    P: float,
    K: float
):

    if N < 0 or P < 0 or K < 0:
        return {
            "error": "N, P and K cannot be negative"
        }

    if ph < 0 or ph > 14:
        return {
            "error": "pH must be between 0 and 14"
        }

    if rainfall < 0:
        return {
            "error": "Rainfall cannot be negative"
        }


    df = pd.DataFrame([{
        "temperature": temperature,
        "rainfall": rainfall,
        "ph": ph,
        "N": N,
        "P": P,
        "K": K
    }])


    prediction = yield_model.predict(df)


    predicted_yield = float(prediction[0])


    # Prevent negative yield from being displayed
    predicted_yield = max(0.0, predicted_yield)


    return {
        "predicted_yield": round(
            predicted_yield,
            2
        )
    }


# ==============================
# 🦠 DISEASE DETECTION
# ==============================

@app.post("/predict_disease")
async def predict_disease(
    file: UploadFile = File(...)
):

    # ------------------------------
    # Check file
    # ------------------------------

    if not file.content_type:
        return {
            "error": "No file type detected"
        }


    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]


    if file.content_type not in allowed_types:
        return {
            "error": "Please upload a JPG, PNG or WEBP image"
        }


    # ------------------------------
    # Read image
    # ------------------------------

    image_bytes = await file.read()


    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        return {
            "error": "Invalid image file"
        }


    # ------------------------------
    # Preprocess
    # ------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)


    # ------------------------------
    # Prediction
    # ------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )


    # ------------------------------
    # Get prediction
    # ------------------------------

    predicted_index = predicted.item()

    disease_raw = classes[
        predicted_index
    ]


    # PyTorch confidence is 0-1
    confidence_value = float(
        confidence.item()
    )


    # Convert ONCE to percentage
    confidence_percent = (
        confidence_value * 100
    )


    # ------------------------------
    # Treatment information
    # ------------------------------

    result = get_cure(
        disease_raw,
        confidence_percent
    )


    # Make sure result is a dictionary
    if not isinstance(result, dict):

        result = {
            "name": str(disease_raw),
            "cure": str(result)
        }


    # ------------------------------
    # Final response
    # ------------------------------

    result["name"] = str(
        result.get(
            "name",
            disease_raw
        )
    )

    result["confidence"] = round(
        confidence_percent,
        2
    )


    return result


# ==============================
# 🌧️ SMART IRRIGATION
# ==============================

@app.post("/irrigation_ai")
def irrigation_ai(data: dict):

    moisture = data.get("moisture")
    ph = data.get("ph")
    city = data.get("city")


    # ------------------------------
    # Validate input
    # ------------------------------

    if moisture is None:
        return {
            "error": "Soil moisture is required"
        }


    if city is None or not str(city).strip():
        return {
            "error": "City name is required"
        }


    try:

        moisture = float(moisture)

    except:

        return {
            "error": "Invalid soil moisture"
        }


    if moisture < 0 or moisture > 100:
        return {
            "error": "Soil moisture must be between 0 and 100"
        }


    # ------------------------------
    # Get weather
    # ------------------------------

    weather = get_weather(
        city
    )


    if weather is None:

        return {
            "error": "Invalid city name"
        }


    temp, humidity, rainfall = weather


    # ------------------------------
    # Irrigation logic
    # ------------------------------

    if rainfall > 3:

        pump = "OFF"

        reason = (
            "Rain expected 🌧️"
        )


    elif moisture < 30 and temp > 30:

        pump = "ON"

        reason = (
            "Hot + Dry soil 🔥💧"
        )


    elif humidity > 85:

        pump = "OFF"

        reason = (
            "High humidity, "
            "avoid unnecessary irrigation"
        )


    elif moisture < 40:

        pump = "ON"

        reason = (
            "Soil moisture low"
        )


    else:

        pump = "OFF"

        reason = (
            "Optimal conditions"
        )


    # ------------------------------
    # Response
    # ------------------------------

    return {

        "pump": pump,

        "reason": reason,

        "weather": {

            "temperature": temp,

            "humidity": humidity,

            "rainfall": rainfall

        }

    }
