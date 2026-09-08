import os
import io
import threading
from datetime import datetime, timezone
from typing import Optional

import requests
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import gdown

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fertilizer_logic import recommend_fertilizer
from disease_cure import get_cure
from irrigation_model import get_weather


# ============================================================
# CONFIGURATION
# ============================================================

LEAF_MODEL_ID = "1w5F68buoOIMjuDntzn2-Pv1k2lD6ZY-y"
CLASSES_ID = "1YGZu2J0dYTQCYeWKTFrxyNPx_lPsxkQM"

CROP_MODEL_ID = "1cZvhsTBkXqij2mqIhvK7MDosfOQjCba1"
YIELD_MODEL_ID = "1LOEAAL4nzUNb_X5S1B7HaQX5ZZZeyHxG"

LEAF_MODEL_PATH = "leaf_model.pth"
CLASSES_PATH = "classes.pth"

CROP_MODEL_PATH = "crop_model.pkl"
YIELD_MODEL_PATH = "yield_model.pkl"

# These are acknowledged trained files.
# Their exact feature schemas should be confirmed before
# using them in prediction endpoints.
SOIL_ML_MODEL_ID = "1bXDiPYOXAFKcxVMIyylZa_TcYorufDtA"
IRRIGATION_ML_MODEL_ID = "1ny4qXolifwvh2hZAjiYwsCzAZOEDpi62"

# ============================================================
# GOOGLE DRIVE MODEL DOWNLOAD
# ============================================================

def download_google_drive_model(file_id: str, output: str) -> None:
    """
    Download a model from Google Drive.

    Strategy:
    1. Use an existing non-empty local file if available.
    2. Try gdown using the Google Drive file URL.
    3. Try Google's normal /uc download endpoint.
    4. Handle Google's confirmation page for large files.
    5. Never save an HTML page as a model file.
    """

    if os.path.exists(output) and os.path.getsize(output) > 0:
        print(
            f"✓ Using existing {output} "
            f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
        )
        return

    print(f"⬇️ Downloading {output} from Google Drive...")

    # --------------------------------------------------------
    # Attempt 1: gdown using the full Google Drive URL
    # --------------------------------------------------------
    drive_url = f"https://drive.google.com/file/d/{file_id}/view"

    try:
        print(f"⬇️ Trying gdown for {output}...")

        downloaded = gdown.download(
            url=drive_url,
            output=output,
            quiet=False,
            fuzzy=True
        )

        if (
            downloaded
            and os.path.exists(output)
            and os.path.getsize(output) > 0
        ):
            print(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        print(f"⚠️ gdown failed for {output}: {exc}")

    # Remove failed/incomplete file
    if os.path.exists(output):
        try:
            os.remove(output)
        except OSError:
            pass

    # --------------------------------------------------------
    # Attempt 2: Google Drive normal download endpoint
    # --------------------------------------------------------
    try:
        print(f"⬇️ Trying Google Drive direct download for {output}...")

        session = requests.Session()

        download_url = (
            "https://drive.google.com/uc"
            f"?export=download&id={file_id}"
        )

        response = session.get(
            download_url,
            stream=True,
            timeout=180,
            allow_redirects=True
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        # ----------------------------------------------------
        # Google may return an HTML confirmation page instead
        # of the actual file, especially for larger files.
        # ----------------------------------------------------
        if "text/html" in content_type:
            html = response.text

            import re

            confirm_token = None

            token_match = re.search(
                r"confirm=([0-9A-Za-z_-]+)",
                html
            )

            if token_match:
                confirm_token = token_match.group(1)

            # Another Google Drive confirmation pattern
            if not confirm_token:
                token_match = re.search(
                    r'name="confirm"\s+value="([^"]+)"',
                    html
                )

                if token_match:
                    confirm_token = token_match.group(1)

            if not confirm_token:
                raise RuntimeError(
                    "Google Drive returned an HTML page instead "
                    "of the model file. The file may not be publicly "
                    "accessible."
                )

            print(
                f"⚠️ Google confirmation required for {output}. "
                f"Retrying..."
            )

            confirmed_url = (
                "https://drive.google.com/uc"
                f"?export=download&id={file_id}"
                f"&confirm={confirm_token}"
            )

            response = session.get(
                confirmed_url,
                stream=True,
                timeout=180,
                allow_redirects=True
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "content-type",
                ""
            ).lower()

            if "text/html" in content_type:
                raise RuntimeError(
                    "Google Drive still returned HTML after "
                    "confirmation instead of the model file."
                )

        # ----------------------------------------------------
        # Save to a temporary file first.
        # This prevents a failed download from becoming a
        # seemingly valid model file.
        # ----------------------------------------------------
        temp_output = output + ".part"

        if os.path.exists(temp_output):
            os.remove(temp_output)

        with open(temp_output, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

        if (
            not os.path.exists(temp_output)
            or os.path.getsize(temp_output) == 0
        ):
            raise RuntimeError(
                "Downloaded file is empty."
            )

        os.replace(temp_output, output)

        print(
            f"✓ Successfully downloaded {output} "
            f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
        )

        return

    except Exception as exc:
        print(
            f"⚠️ Google direct download failed for "
            f"{output}: {exc}"
        )

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------
    raise RuntimeError(
        f"\n"
        f"Failed to download {output} from Google Drive.\n"
        f"\n"
        f"File ID: {file_id}\n"
        f"\n"
        f"Please verify that the Google Drive file is:\n"
        f"  • Anyone with the link\n"
        f"  • Viewer permission\n"
        f"  • Not inside a restricted/shared-drive location\n"
        f"\n"
        f"Google Drive URL:\n"
        f"https://drive.google.com/file/d/{file_id}/view"
    )
# ============================================================
# DOWNLOAD ALL MODELS FROM GOOGLE DRIVE
# ============================================================

download_google_drive_model(
    LEAF_MODEL_ID,
    LEAF_MODEL_PATH
)

download_google_drive_model(
    CLASSES_ID,
    CLASSES_PATH
)

download_google_drive_model(
    CROP_MODEL_ID,
    CROP_MODEL_PATH
)

download_google_drive_model(
    YIELD_MODEL_ID,
    YIELD_MODEL_PATH
)

download_google_drive_model(
    SOIL_ML_MODEL_ID,
    SOIL_ML_MODEL_PATH
)

download_google_drive_model(
    IRRIGATION_ML_MODEL_ID,
    IRRIGATION_ML_MODEL_PATH
)
# ============================================================
# LOAD SOIL ML MODEL
# ============================================================

soil_ml_model = None

try:
    soil_ml_model = joblib.load(SOIL_ML_MODEL_PATH)
    print(f"✓ {SOIL_ML_MODEL_PATH} loaded")
except Exception as exc:
    print(
        f"⚠️ Could not load {SOIL_ML_MODEL_PATH}: {exc}"
    )


# ============================================================
# LOAD IRRIGATION ML MODEL
# ============================================================

irrigation_ml_model = None

try:
    irrigation_ml_model = joblib.load(
        IRRIGATION_ML_MODEL_PATH
    )
    print(
        f"✓ {IRRIGATION_ML_MODEL_PATH} loaded"
    )
except Exception as exc:
    print(
        f"⚠️ Could not load "
        f"{IRRIGATION_ML_MODEL_PATH}: {exc}"
    )
    
# ============================================================
# LOAD CROP MODEL
# ============================================================

crop_model = joblib.load(CROP_MODEL_PATH)

print("✓ crop_model.pkl loaded")


# ============================================================
# LOAD YIELD MODEL
# ============================================================

yield_model = joblib.load(YIELD_MODEL_PATH)

print("✓ yield_model.pkl loaded")


# ============================================================
# LOAD DISEASE CLASS NAMES
# ============================================================

classes = torch.load(
    CLASSES_PATH,
    map_location="cpu"
)

if isinstance(classes, torch.Tensor):
    classes = classes.tolist()

# Make sure class names are usable as list values
classes = list(classes)

print(f"✓ Loaded {len(classes)} disease classes")


# ============================================================
# LOAD LEAF DISEASE MODEL
# ============================================================

model = models.resnet18(weights=None)

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

print("✓ leaf_model.pth loaded")


# ============================================================
# DISEASE IMAGE PREPROCESSING
# ============================================================

disease_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AgriAI Smart Farming API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SENSOR STATE
# ============================================================

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


# ============================================================
# REQUEST MODELS
# ============================================================

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class FertilizerRequest(BaseModel):
    N: float
    P: float
    K: float
    crop: str


class YieldRequest(BaseModel):
    temperature: float
    rainfall: float
    ph: float
    N: float
    P: float
    K: float


class IrrigationRequest(BaseModel):
    moisture: float
    ph: float
    city: str


class SensorPayload(BaseModel):
    moisture: float
    temperature: float
    ec: float
    ph: float
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    rainfall: Optional[float] = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AgriAI Smart Farming API is running",
        "status": "online",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "crop_model": crop_model is not None,
        "yield_model": yield_model is not None,
        "leaf_model": model is not None,
        "disease_classes": len(classes),
        "soil_ml_model_loaded": soil_ml_model is not None,
        "irrigation_ml_model_loaded": irrigation_ml_model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# CROP RECOMMENDATION
# ============================================================

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
    """
    Crop model feature order:

    [N, P, K, temperature, humidity, ph, rainfall]

    IMPORTANT:
    humidity is AIR humidity.
    It is NOT soil moisture.
    """

    try:
        data = pd.DataFrame([[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]], columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ])

        prediction = crop_model.predict(data)[0]

        return {
            "recommended_crop": str(prediction)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Crop prediction failed: {str(exc)}"
        )


# ============================================================
# FERTILIZER RECOMMENDATION
# ============================================================

@app.post("/fertilizer")
def fertilizer(request: FertilizerRequest):
    try:
        result = recommend_fertilizer(
            request.N,
            request.P,
            request.K,
            request.crop
        )

        return {
            "recommendation": result
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fertilizer recommendation failed: {str(exc)}"
        )


# ============================================================
# YIELD PREDICTION
# ============================================================

@app.get("/predict_yield")
def predict_yield(
    temperature: float,
    rainfall: float,
    ph: float,
    N: float,
    P: float,
    K: float
):
    """
    Yield model feature order:

    [temperature, rainfall, ph, N, P, K]
    """

    try:
        data = pd.DataFrame([[
            temperature,
            rainfall,
            ph,
            N,
            P,
            K
        ]], columns=[
            "temperature",
            "rainfall",
            "ph",
            "N",
            "P",
            "K"
        ])

        prediction = yield_model.predict(data)[0]

        # Prevent negative yield predictions
        prediction = max(0, float(prediction))

        return {
            "predicted_yield": prediction
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Yield prediction failed: {str(exc)}"
        )


# ============================================================
# DISEASE DETECTION
# ============================================================

@app.post("/predict_disease")
async def predict_disease(
    file: UploadFile = File(...)
):
    """
    Predict plant disease from uploaded leaf image.

    Confidence is converted:
        model probability 0-1
        ->
        percentage 0-100

    Conversion happens exactly once here.
    """

    try:
        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="File content type is missing."
            )

        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload an image file."
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = disease_transform(image)
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        class_index = int(predicted.item())

        disease_name = str(
            classes[class_index]
        )

        # Convert 0-1 probability to percentage exactly once.
        confidence_percent = float(
            confidence.item() * 100.0
        )

        cure = get_cure(disease_name)

        return {
            "disease": disease_name,
            "confidence": round(confidence_percent, 2),
            "cure": cure
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Disease prediction failed: {str(exc)}"
        )


# ============================================================
# IRRIGATION AI
# ============================================================

@app.post("/irrigation_ai")
def irrigation_ai(request: IrrigationRequest):
    """
    Current irrigation logic uses:
    - soil moisture
    - weather/rainfall
    - temperature
    - humidity
    - city

    pH is accepted as an input but is not used by the current
    rule-based irrigation logic.
    """

    try:
        moisture = float(request.moisture)
        ph = float(request.ph)
        city = request.city.strip()

        if not city:
            raise HTTPException(
                status_code=400,
                detail="City is required."
            )

        if moisture < 0 or moisture > 100:
            raise HTTPException(
                status_code=400,
                detail="Soil moisture must be between 0 and 100."
            )

        weather = get_weather(city)

        if not isinstance(weather, dict):
            raise RuntimeError(
                "Weather service returned an invalid response."
            )

        temperature = float(
            weather.get("temperature", 0)
        )

        humidity = float(
            weather.get("humidity", 0)
        )

        rainfall = float(
            weather.get("rainfall", 0)
        )

        # ----------------------------------------------------
        # Irrigation rules
        # ----------------------------------------------------

        if rainfall > 3:
            irrigation = "OFF"
            reason = "Rainfall is sufficient."

        elif moisture < 30 and temperature > 30:
            irrigation = "ON"
            reason = "Soil is very dry and temperature is high."

        elif humidity > 85:
            irrigation = "OFF"
            reason = "Air humidity is very high."

        elif moisture < 40:
            irrigation = "ON"
            reason = "Soil moisture is below the irrigation threshold."

        else:
            irrigation = "OFF"
            reason = "Soil moisture is sufficient."

        return {
            "irrigation": irrigation,
            "reason": reason,
            "moisture": moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "ph": ph,
            "city": city
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Irrigation prediction failed: {str(exc)}"
        )


# ============================================================
# RECEIVE ESP32 SENSOR DATA
# ============================================================

@app.post("/sensor_data")
def receive_sensor_data(payload: SensorPayload):

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
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "source": "ESP32 / RS485 soil sensor",
        })

        return dict(_latest_sensor_data)


# ============================================================
# GET LATEST ESP32 SENSOR DATA
# ============================================================

@app.get("/sensor_data")
def get_sensor_data():

    with _sensor_lock:
        return dict(_latest_sensor_data)
