import os
import io
import re
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

SOIL_MODEL_ID = "1bXDiPYOXAFKcxVMIyylZa_TcYorufDtA"
IRRIGATION_MODEL_ID = "1ny4qXolifwvh2hZAjiYwsCzAZOEDpi62"

LEAF_MODEL_PATH = "leaf_model.pth"
CLASSES_PATH = "classes.pth"

CROP_MODEL_PATH = "crop_model.pkl"
YIELD_MODEL_PATH = "yield_model.pkl"

SOIL_ML_MODEL_PATH = "soil_ml_model.pkl"
IRRIGATION_ML_MODEL_PATH = "irrigation_model.pkl"


# ============================================================
# GOOGLE DRIVE MODEL DOWNLOAD
# ============================================================

def _is_valid_download(path: str) -> bool:
    """
    Return True only when a non-empty file exists.
    """
    return (
        os.path.exists(path)
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def _looks_like_html(path: str) -> bool:
    """
    Google Drive may return an HTML login/error/confirmation page
    instead of the requested binary file.
    """
    if not _is_valid_download(path):
        return False

    try:
        with open(path, "rb") as file:
            sample = file.read(4096).lower()

        return (
            b"<html" in sample
            or b"<!doctype html" in sample
            or b"<head" in sample
            or b"google drive" in sample and b"<" in sample
        )
    except Exception:
        return False


def _save_response_to_file(response: requests.Response, output: str) -> None:
    """
    Stream a successful binary HTTP response to disk safely.
    """
    temp_path = output + ".part"

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    content_type = response.headers.get("content-type", "").lower()

    if "text/html" in content_type:
        raise RuntimeError(
            "Google returned an HTML page instead of the model file."
        )

    total_bytes = 0

    with open(temp_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                total_bytes += len(chunk)

    if total_bytes == 0:
        raise RuntimeError(
            "Google returned an empty response."
        )

    os.replace(temp_path, output)


def _extract_confirm_token(text: str) -> Optional[str]:
    """
    Extract Google's confirmation token from an HTML response.

    Google Drive sometimes returns a confirmation page for larger
    files instead of immediately returning the binary content.
    """

    patterns = [
        r'name="confirm"\s+value="([^"]+)"',
        r'name="confirm"\s*value="([^"]+)"',
        r"confirm=([0-9A-Za-z_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return None


def _google_drive_requests_download(
    file_id: str,
    output: str
) -> None:
    """
    Download a Google Drive file using requests.

    This is the fallback when gdown cannot download the file.
    It supports Google's confirmation-token flow.
    """

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    })

    # --------------------------------------------------------
    # First request
    # --------------------------------------------------------

    url = "https://drive.google.com/uc"

    params = {
        "export": "download",
        "id": file_id,
    }

    print("⬇️ Trying Google Drive direct download...")

    response = session.get(
        url,
        params=params,
        stream=True,
        timeout=120,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    # --------------------------------------------------------
    # Direct binary response
    # --------------------------------------------------------

    if "text/html" not in content_type:
        _save_response_to_file(response, output)

        if _is_valid_download(output) and not _looks_like_html(output):
            return

        raise RuntimeError(
            "Google Drive returned an invalid file."
        )

    # --------------------------------------------------------
    # HTML confirmation page
    # --------------------------------------------------------

    html = response.text

    confirm_token = _extract_confirm_token(html)

    if not confirm_token:
        # Some Google Drive responses contain a download link
        # rather than a simple confirm form.
        match = re.search(
            r'href="([^"]+download[^"]+)"',
            html
        )

        if match:
            download_url = match.group(1)

            download_url = (
                download_url
                .replace("&amp;", "&")
            )

            response2 = session.get(
                download_url,
                stream=True,
                timeout=120,
                allow_redirects=True,
            )

            response2.raise_for_status()

            _save_response_to_file(
                response2,
                output
            )

            if (
                _is_valid_download(output)
                and not _looks_like_html(output)
            ):
                return

        raise RuntimeError(
            "Google Drive returned an HTML page and "
            "no download confirmation token could be found."
        )

    # --------------------------------------------------------
    # Confirmation-token request
    # --------------------------------------------------------

    confirm_params = {
        "export": "download",
        "id": file_id,
        "confirm": confirm_token,
    }

    response2 = session.get(
        url,
        params=confirm_params,
        stream=True,
        timeout=120,
        allow_redirects=True,
    )

    response2.raise_for_status()

    _save_response_to_file(
        response2,
        output
    )

    if not _is_valid_download(output):
        raise RuntimeError(
            "Google Drive confirmation download "
            "produced an empty file."
        )

    if _looks_like_html(output):
        raise RuntimeError(
            "Google Drive confirmation download "
            "returned HTML instead of the model."
        )


def download_google_drive_model(
    file_id: str,
    output: str
) -> None:
    """
    Download a model from Google Drive.

    Order:
        1. Existing local file
        2. gdown using file ID
        3. gdown using standard Drive URL
        4. requests + Google Drive confirmation flow
    """

    # --------------------------------------------------------
    # Existing local file
    # --------------------------------------------------------

    if _is_valid_download(output) and not _looks_like_html(output):
        print(
            f"✓ Using existing {output} "
            f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
        )
        return

    print(
        f"\n⬇️ Downloading {output} from Google Drive..."
    )

    # Remove invalid/incomplete previous file
    for path in [output, output + ".part"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # --------------------------------------------------------
    # Attempt 1: gdown with file ID
    # --------------------------------------------------------

    try:
        print(
            f"⬇️ Trying gdown for {output}..."
        )

        downloaded = gdown.download(
            id=file_id,
            output=output,
            quiet=False,
        )

        if (
            downloaded
            and _is_valid_download(output)
            and not _looks_like_html(output)
        ):
            print(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        print(
            f"⚠️ gdown ID download failed for "
            f"{output}: {exc}"
        )

    # Clean up after failed attempt
    for path in [output, output + ".part"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # --------------------------------------------------------
    # Attempt 2: gdown using standard Drive URL
    # --------------------------------------------------------

    try:
        drive_url = (
            f"https://drive.google.com/file/d/"
            f"{file_id}/view"
        )

        print(
            f"⬇️ Trying gdown Drive URL for {output}..."
        )

        downloaded = gdown.download(
            url=drive_url,
            output=output,
            quiet=False,
        )

        if (
            downloaded
            and _is_valid_download(output)
            and not _looks_like_html(output)
        ):
            print(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        print(
            f"⚠️ gdown URL download failed for "
            f"{output}: {exc}"
        )

    # Clean up after failed attempt
    for path in [output, output + ".part"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # --------------------------------------------------------
    # Attempt 3: requests fallback
    # --------------------------------------------------------

    try:
        _google_drive_requests_download(
            file_id,
            output
        )

        if (
            _is_valid_download(output)
            and not _looks_like_html(output)
        ):
            print(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        print(
            f"⚠️ Google Drive direct download failed "
            f"for {output}: {exc}"
        )

    # --------------------------------------------------------
    # Final failure
    # --------------------------------------------------------

    raise RuntimeError(
        f"\n"
        f"Failed to download {output} from Google Drive.\n"
        f"\n"
        f"File ID:\n"
        f"{file_id}\n"
        f"\n"
        f"Please verify that the Google Drive file is:\n"
        f"  • Anyone with the link\n"
        f"  • Viewer permission\n"
        f"  • Not inside a restricted/shared-drive location\n"
        f"  • Not blocked by an organization policy\n"
        f"\n"
        f"Google Drive URL:\n"
        f"https://drive.google.com/file/d/{file_id}/view\n"
    )


# ============================================================
# DOWNLOAD TRAINED MODELS FROM GOOGLE DRIVE
# ============================================================

print("\n============================================================")
print("DOWNLOADING AGRIAI MODELS")
print("============================================================")

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

print("============================================================")
print("MODEL DOWNLOADS COMPLETE")
print("============================================================\n")


# ============================================================
# VERIFY LOCAL MODELS
# ============================================================

def require_local_model(path: str) -> None:
    """
    Verify that a downloaded model exists and is non-empty.
    """

    if not os.path.exists(path):
        raise RuntimeError(
            f"Required model '{path}' was not found."
        )

    if os.path.getsize(path) == 0:
        raise RuntimeError(
            f"Required model '{path}' is empty."
        )

    print(
        f"✓ Found model {path} "
        f"({os.path.getsize(path) / (1024 * 1024):.2f} MB)"
    )


require_local_model(LEAF_MODEL_PATH)
require_local_model(CLASSES_PATH)
require_local_model(CROP_MODEL_PATH)
require_local_model(YIELD_MODEL_PATH)


# ============================================================
# OPTIONAL SOIL / IRRIGATION TRAINED MODELS
# ============================================================

soil_ml_model = None
irrigation_ml_model = None

# ------------------------------------------------------------
# Soil ML model
# ------------------------------------------------------------

if _is_valid_download(SOIL_ML_MODEL_PATH):
    try:
        soil_ml_model = joblib.load(
            SOIL_ML_MODEL_PATH
        )

        print(
            f"✓ Loaded {SOIL_ML_MODEL_PATH}"
        )

    except Exception as exc:
        print(
            f"⚠️ Could not load "
            f"{SOIL_ML_MODEL_PATH}: {exc}"
        )

# ------------------------------------------------------------
# Irrigation ML model
# ------------------------------------------------------------

if _is_valid_download(IRRIGATION_ML_MODEL_PATH):
    try:
        irrigation_ml_model = joblib.load(
            IRRIGATION_ML_MODEL_PATH
        )

        print(
            f"✓ Loaded {IRRIGATION_ML_MODEL_PATH}"
        )

    except Exception as exc:
        print(
            f"⚠️ Could not load "
            f"{IRRIGATION_ML_MODEL_PATH}: {exc}"
        )


# ============================================================
# LOAD CROP MODEL
# ============================================================

crop_model = joblib.load(
    CROP_MODEL_PATH
)

print("✓ crop_model.pkl loaded")


# ============================================================
# LOAD YIELD MODEL
# ============================================================

yield_model = joblib.load(
    YIELD_MODEL_PATH
)

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

classes = list(classes)

print(
    f"✓ Loaded {len(classes)} disease classes"
)


# ============================================================
# LOAD LEAF DISEASE MODEL
# ============================================================

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

print(
    "✓ leaf_model.pth loaded"
)


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
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
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

    humidity is AIR humidity.
    It is NOT soil moisture.
    """

    try:
        data = pd.DataFrame(
            [[
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )

        prediction = crop_model.predict(
            data
        )[0]

        return {
            "recommended_crop": str(
                prediction
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Crop prediction failed: {str(exc)}"
            )
        )


# ============================================================
# FERTILIZER RECOMMENDATION
# ============================================================

@app.post("/fertilizer")
def fertilizer(
    request: FertilizerRequest
):
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
            detail=(
                "Fertilizer recommendation failed: "
                f"{str(exc)}"
            )
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
        data = pd.DataFrame(
            [[
                temperature,
                rainfall,
                ph,
                N,
                P,
                K
            ]],
            columns=[
                "temperature",
                "rainfall",
                "ph",
                "N",
                "P",
                "K"
            ]
        )

        prediction = yield_model.predict(
            data
        )[0]

        prediction = max(
            0,
            float(prediction)
        )

        return {
            "predicted_yield": prediction
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Yield prediction failed: {str(exc)}"
            )
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

    Model probability:
        0.0 - 1.0

    API confidence:
        0 - 100

    Conversion happens exactly once here.
    """

    try:

        # ----------------------------------------------------
        # Validate content type
        # ----------------------------------------------------

        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    "File content type is missing."
                )
            )

        if not file.content_type.startswith(
            "image/"
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Please upload an image file."
                )
            )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded image is empty."
                )
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # ----------------------------------------------------
        # EXACT preprocessing
        # ----------------------------------------------------

        tensor = disease_transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        class_index = int(
            predicted.item()
        )

        if (
            class_index < 0
            or class_index >= len(classes)
        ):
            raise RuntimeError(
                "Predicted class index is outside "
                "the available disease classes."
            )

        disease_name = str(
            classes[class_index]
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------
        # Convert probability 0-1 to percentage 0-100.
        # DO NOT multiply anywhere else.

        confidence_percent = float(
            confidence.item() * 100.0
        )

        # ----------------------------------------------------
        # Cure information
        # ----------------------------------------------------

        cure = get_cure(
            disease_name
        )

        return {
            "disease": disease_name,
            "confidence": round(
                confidence_percent,
                2
            ),
            "cure": cure
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Disease prediction failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# IRRIGATION AI
# ============================================================

@app.post("/irrigation_ai")
def irrigation_ai(
    request: IrrigationRequest
):
    """
    Current irrigation logic uses:

    - soil moisture
    - weather/rainfall
    - temperature
    - humidity
    - city

    pH is accepted as an input but is not used by
    the current rule-based irrigation logic.
    """

    try:

        moisture = float(
            request.moisture
        )

        ph = float(
            request.ph
        )

        city = request.city.strip()

        # ----------------------------------------------------
        # Validate city
        # ----------------------------------------------------

        if not city:
            raise HTTPException(
                status_code=400,
                detail="City is required."
            )

        # ----------------------------------------------------
        # Validate moisture
        # ----------------------------------------------------

        if moisture < 0 or moisture > 100:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Soil moisture must be "
                    "between 0 and 100."
                )
            )

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        weather = get_weather(
            city
        )

        if not isinstance(
            weather,
            dict
        ):
            raise RuntimeError(
                "Weather service returned "
                "an invalid response."
            )

        temperature = float(
            weather.get(
                "temperature",
                0
            )
        )

        humidity = float(
            weather.get(
                "humidity",
                0
            )
        )

        rainfall = float(
            weather.get(
                "rainfall",
                0
            )
        )

        # ----------------------------------------------------
        # Irrigation rules
        # ----------------------------------------------------

        if rainfall > 3:

            irrigation = "OFF"

            reason = (
                "Rainfall is sufficient."
            )

        elif (
            moisture < 30
            and temperature > 30
        ):

            irrigation = "ON"

            reason = (
                "Soil is very dry and "
                "temperature is high."
            )

        elif humidity > 85:

            irrigation = "OFF"

            reason = (
                "Air humidity is very high."
            )

        elif moisture < 40:

            irrigation = "ON"

            reason = (
                "Soil moisture is below "
                "the irrigation threshold."
            )

        else:

            irrigation = "OFF"

            reason = (
                "Soil moisture is sufficient."
            )

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
            detail=(
                f"Irrigation prediction failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# RECEIVE ESP32 SENSOR DATA
# ============================================================

@app.post("/sensor_data")
def receive_sensor_data(
    payload: SensorPayload
):
    """
    Receive real sensor data from ESP32 / RS485.

    The sensor state starts as None and is only populated
    when an ESP32 sends actual data.

    rainfall remains optional because the soil sensor itself
    does not measure rainfall.
    """

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
            "source": (
                "ESP32 / RS485 soil sensor"
            ),
        })

        return dict(
            _latest_sensor_data
        )


# ============================================================
# GET LATEST ESP32 SENSOR DATA
# ============================================================

@app.get("/sensor_data")
def get_sensor_data():

    with _sensor_lock:
        return dict(
            _latest_sensor_data
        )
