"""
Shayekli (شيّكلي) — FastAPI backend
Cloud-deployable scam detection API.

Designed to run on Railway. All host/port specifics come from environment.
Tesseract OCR is loaded lazily and is OPTIONAL — the service still boots
on a clean Railway/Nixpacks image without the native binary installed.
The full OCR pipeline will be replaced by Google Cloud Vision in Part 3
of the upgrade plan; until then the /ocr-predict endpoint degrades
gracefully when Tesseract is unavailable.
"""

import io
import os
import re
import pickle
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("shayekli")

# ---------------------------------------------------------------------------
# Configuration (env-driven, Railway-friendly)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "checkley_model.pkl")
MODEL_VERSION = int(os.environ.get("MODEL_VERSION", "1"))
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
APP_NAME = "Shayekli"

# CORS — keep permissive for the mobile app. Tighten via env in production.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(
    title="Shayekli AI Backend",
    description="Palestinian-Arabic SMS / messaging scam detection API",
    version=f"1.0.{MODEL_VERSION}",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional Tesseract OCR
# ---------------------------------------------------------------------------
_tesseract = None
_tesseract_error: Optional[str] = None


def _try_load_tesseract():
    """Load pytesseract lazily so cloud builds without the native binary still boot."""
    global _tesseract, _tesseract_error
    if _tesseract is not None or _tesseract_error is not None:
        return _tesseract
    try:
        import pytesseract  # noqa: WPS433 (intentional lazy import)

        custom_path = os.environ.get("TESSERACT_CMD")
        if custom_path:
            pytesseract.pytesseract.tesseract_cmd = custom_path
        elif os.name == "nt":
            default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_win):
                pytesseract.pytesseract.tesseract_cmd = default_win
        # Probe — will raise if the binary isn't actually callable.
        pytesseract.get_tesseract_version()
        _tesseract = pytesseract
        log.info("Tesseract OCR available.")
    except Exception as exc:  # noqa: BLE001
        _tesseract_error = str(exc)
        log.warning("Tesseract OCR unavailable (%s). /ocr-predict will return a friendly error.", exc)
    return _tesseract


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SMSRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    is_scam: bool
    risk_level: str
    confidence: float
    message: str
    reasons: List[str]
    model_version: int = MODEL_VERSION


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: int
    environment: str
    app: str = APP_NAME


class ModelVersionResponse(BaseModel):
    model_version: int
    model_loaded: bool


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            log.info("Loaded model from %s (version=%s)", MODEL_PATH, MODEL_VERSION)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load model: %s", exc)
            model = None
    else:
        log.warning("Model file not found at %s. /predict will return 503 until trained.", MODEL_PATH)


# ---------------------------------------------------------------------------
# Reason extraction (heuristic explanations layered on top of the classifier)
# ---------------------------------------------------------------------------
def extract_reasons(text: str, is_scam: bool) -> List[str]:
    reasons: List[str] = []
    if not is_scam:
        reasons.append("لم يتم العثور على أنماط مشبوهة أو روابط خبيثة في النص.")
        return reasons

    if re.search(r"(https?://|www\.)[^\s]+", text, re.IGNORECASE):
        reasons.append("الرسالة تحتوي على رابط إلكتروني خارجي. الروابط في الرسائل المجهولة غالباً ما تكون ضارة لمحاولة اختراق جهازك.")

    if re.search(r"(فورا|تحديث|ايقاف|إيقاف|تجميد|حظر|انذار|إنذار|مؤقت|عاجل|ضروري)", text):
        reasons.append("استخدام لغة تخويف أو استعجال لإجبارك على التصرف بسرعة دون تفكير.")

    if re.search(r"(بنك|البنك|صراف|حسابك|بطاقتك|العميل|عزيزي العميل|بطاقة)", text):
        reasons.append("ادعاء كاذب يمثل جهة رسمية أو بنكية لطلب تحديث بياناتك الشخصية.")

    if re.search(r"(مبروك|ربحت|جائزة|مسابقة|دولار|₪|\$|اربح|كاش|نقدا)", text):
        reasons.append("وعود مالية أو جوائز وهمية تستخدم كطعم لخداعك.")

    if re.search(r"(اتصل|الاتصال|تواصل|الرقم|00\d{3,})", text):
        reasons.append("طلب منك التواصل أو الاتصال برقم مجهول أو دولي غريب.")

    if not reasons:
        reasons.append("تعرف الذكاء الاصطناعي على أسلوب صياغة مشبوه مطابق للرسائل الاحتيالية السابقة.")

    return reasons


def _classify(text: str) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    scam_prob = float(probs[1])
    is_scam = bool(pred == 1)

    if scam_prob > 0.8:
        risk_level, msg = "HIGH", "خطر مرتفع: تصيد واحتيال"
    elif scam_prob > 0.5:
        risk_level, msg = "MEDIUM", "خطر متوسط: يرجى الحذر"
    else:
        risk_level, msg = "LOW", "آمن: لا يوجد خطر واضح"

    return PredictionResponse(
        is_scam=is_scam,
        risk_level=risk_level,
        confidence=round(scam_prob * 100, 2),
        message=msg,
        reasons=extract_reasons(text, is_scam),
        model_version=MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": APP_NAME,
        "message": "Welcome to the Shayekli AI API",
        "environment": ENVIRONMENT,
        "model_version": MODEL_VERSION,
    }


@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "pong"


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Cloud health check used by the mobile app on launch and by Railway.
    Returns model status and version so OTA updates can compare.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    )


@app.get("/model/version", response_model=ModelVersionResponse)
def model_version():
    return ModelVersionResponse(model_version=MODEL_VERSION, model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict_sms(request: SMSRequest):
    return _classify(request.text)


# Alias path expected by the upgrade spec.
@app.post("/analyze", response_model=PredictionResponse)
def analyze_message(request: SMSRequest):
    return _classify(request.text)


@app.post("/ocr-predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")

    tess = _try_load_tesseract()
    if tess is None:
        # Graceful degradation — the cloud build doesn't ship Tesseract.
        return JSONResponse(
            status_code=503,
            content={
                "is_scam": False,
                "risk_level": "LOW",
                "confidence": 0.0,
                "message": "فحص الصور غير متاح حالياً",
                "reasons": [
                    "خدمة استخراج النص من الصور قيد الترقية. الرجاء استخدام فحص النص في الوقت الحالي.",
                ],
                "model_version": MODEL_VERSION,
            },
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        try:
            extracted_text = tess.image_to_string(image, lang="ara")
        except Exception as exc:  # noqa: BLE001
            log.warning("Tesseract OCR failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"OCR Processing failed: {exc}")

        text = extracted_text.strip()
        if not text:
            return PredictionResponse(
                is_scam=False,
                risk_level="LOW",
                confidence=0.0,
                message="تعذر استخراج نص",
                reasons=["لم يتم العثور على نص واضح في هذه الصورة، يرجى المحاولة بصورة أوضح."],
                model_version=MODEL_VERSION,
            )

        return _classify(text)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}")


@app.post("/feedback")
def submit_feedback(payload: dict):
    """
    Stub for the user-feedback loop (Part 3.6 of the upgrade plan).
    Accepts and logs feedback; full Firestore wiring lands later.
    """
    log.info("feedback received keys=%s", list(payload.keys()))
    return {"status": "received"}
