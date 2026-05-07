"""
Shayekli (شيّكلي) — FastAPI backend.

Phase 2a detection pipeline:
    text → preprocess_arabic
         → primary classifier (current scikit-learn; AraBERT ONNX in a follow-up)
         → URL extraction + multi-signal reputation engine
         → if classifier confidence is in the [40%, 65%] band → consult Claude
         → fuse all signals into final verdict + reasons
         → respond.
Designed to run on Railway. All host/port specifics come from environment.
"""

import io
import os
import re
import json
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from preprocess import preprocess_arabic, extract_urls
from url_reputation import analyze_urls, URLReport
import claude_fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("shayekli")

# ---------------------------------------------------------------------------
# Configuration (env-driven, Railway-friendly)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "checkley_model.pkl")
MODEL_VERSION = int(os.environ.get("MODEL_VERSION", "1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "scikit-arabic-tfidf")
MODEL_DOWNLOAD_URL = os.environ.get("MODEL_DOWNLOAD_URL", "")  # optional OTA host
MODEL_CHANGELOG_AR = os.environ.get(
    "MODEL_CHANGELOG_AR",
    "تحسينات على الكشف عن رسائل التصيد باللغة العربية.",
)
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
APP_NAME = "Shayekli"

CLAUDE_LOW = float(os.environ.get("CLAUDE_FALLBACK_LOW", "0.40"))
CLAUDE_HIGH = float(os.environ.get("CLAUDE_FALLBACK_HIGH", "0.65"))

FEEDBACK_PATH = Path(os.environ.get("FEEDBACK_PATH", "feedback.jsonl"))

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
# Optional Tesseract OCR (will be replaced by Cloud Vision in Part 3).
# ---------------------------------------------------------------------------
_tesseract = None
_tesseract_error: Optional[str] = None


def _try_load_tesseract():
    global _tesseract, _tesseract_error
    if _tesseract is not None or _tesseract_error is not None:
        return _tesseract
    try:
        import pytesseract  # noqa: WPS433
        custom_path = os.environ.get("TESSERACT_CMD")
        if custom_path:
            pytesseract.pytesseract.tesseract_cmd = custom_path
        elif os.name == "nt":
            default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_win):
                pytesseract.pytesseract.tesseract_cmd = default_win
        pytesseract.get_tesseract_version()
        _tesseract = pytesseract
        log.info("Tesseract OCR available.")
    except Exception as exc:  # noqa: BLE001
        _tesseract_error = str(exc)
        log.warning("Tesseract OCR unavailable (%s).", exc)
    return _tesseract


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SMSRequest(BaseModel):
    text: str
    sender: Optional[str] = None


class URLReportOut(BaseModel):
    original: str
    final_url: str
    domain: str
    tld: str
    risk_score: int
    flags_ar: List[str]
    safe_browsing_threat: Optional[str] = None
    virustotal_malicious: int = 0
    virustotal_suspicious: int = 0
    domain_age_days: Optional[int] = None
    phishtank_match: bool = False
    is_shortener: bool = False
    redirect_count: int = 0


class PredictionResponse(BaseModel):
    is_scam: bool
    risk_level: str
    confidence: float
    message: str
    reasons: List[str]
    model_version: int = MODEL_VERSION
    urls: List[URLReportOut] = []
    used_claude: bool = False
    pipeline: List[str] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: int
    environment: str
    app: str = APP_NAME
    claude_enabled: bool = False
    safe_browsing_enabled: bool = False
    virustotal_enabled: bool = False


class ModelVersionResponse(BaseModel):
    model_version: int
    model_name: str
    model_loaded: bool
    download_url: Optional[str] = None
    changelog_ar: Optional[str] = None


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
# Heuristic reasons (layered on top of the classifier verdict)
# ---------------------------------------------------------------------------
def heuristic_flags(text: str) -> List[str]:
    flags: List[str] = []
    if re.search(r"(فورا|تحديث|ايقاف|إيقاف|تجميد|حظر|انذار|إنذار|مؤقت|عاجل|ضروري|انتهاء|تعليق)", text):
        flags.append("لغة استعجال أو تخويف لإجبارك على التصرف بسرعة دون تفكير.")
    if re.search(r"(بنك|البنك|صراف|حسابك|بطاقتك|العميل|عزيزي العميل|بطاقة|تحويل|تجميد الحساب)", text):
        flags.append("ادعاء تمثيل جهة بنكية وطلب تحديث أو تأكيد بياناتك.")
    if re.search(r"(مبروك|ربحت|جائزة|مسابقة|دولار|₪|\$|اربح|كاش|نقدا|ربح\s)", text):
        flags.append("وعود مالية أو جوائز وهمية كطعم.")
    if re.search(r"(اتصل|الاتصال|تواصل|الرقم|00\d{3,}|\+\d{6,})", text):
        flags.append("طلب الاتصال أو التواصل مع رقم مجهول/دولي.")
    if re.search(r"(otp|رمز|كود|كلمة المرور|الرقم السري|verification)", text, re.IGNORECASE):
        flags.append("طلب رمز تحقق أو كلمة مرور — لا تشاركها أبداً.")
    if re.search(r"(جوال|بالتل|paltel|jawwal|ooredoo|أوريدو)", text, re.IGNORECASE):
        flags.append("ادعاء تمثيل شركات الاتصالات المحلية.")
    return flags


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _risk_label(risk: float) -> tuple[str, str]:
    if risk >= 80:
        return "HIGH", "خطر مرتفع: تصيد واحتيال"
    if risk >= 50:
        return "MEDIUM", "خطر متوسط: يرجى الحذر"
    return "LOW", "آمن: لا يوجد خطر واضح"


def _run_classifier(clean_text: str) -> tuple[float, bool]:
    """Returns (scam_probability_0_to_1, is_scam)."""
    pred = model.predict([clean_text])[0]
    probs = model.predict_proba([clean_text])[0]
    scam_prob = float(probs[1])
    return scam_prob, bool(pred == 1)


async def _gather_url_reports(text: str) -> list[URLReport]:
    urls = extract_urls(text)
    if not urls:
        return []
    try:
        return await analyze_urls(urls)
    except Exception as exc:  # noqa: BLE001
        log.warning("URL analysis failed entirely: %s", exc)
        return []


def _fuse(
    classifier_prob: float,
    url_reports: list[URLReport],
    claude_result: dict[str, Any] | None,
) -> tuple[float, bool, str, list[str], list[str]]:
    """
    Returns (final_risk_0_to_100, is_scam, risk_label_message, reasons_ar, pipeline_steps).
    """
    pipeline = ["preprocess", "classifier"]
    reasons: list[str] = []

    risk = classifier_prob * 100.0

    # Fold URL signals into the score (cap their contribution at +50).
    if url_reports:
        pipeline.append("url_reputation")
        max_url_risk = max((r.risk_score for r in url_reports), default=0)
        url_boost = max_url_risk * 0.5
        risk = min(100.0, max(risk, max_url_risk * 0.85))  # a 90 URL → ≥76 final
        risk = min(100.0, risk + url_boost * 0.3)
        for r in url_reports:
            if r.flags_ar:
                reasons.extend(r.flags_ar)

    # Claude takes precedence when invoked (it sees the URL context too).
    if claude_result is not None:
        pipeline.append("claude_fallback")
        verdict = claude_result.get("verdict", "uncertain")
        c_conf = float(claude_result.get("confidence", 50.0))
        if verdict == "scam":
            risk = max(risk, max(c_conf, 70.0))
        elif verdict == "legitimate":
            risk = min(risk, min(c_conf if c_conf < 50 else 100 - c_conf, 30.0))
        else:
            risk = (risk + c_conf) / 2.0
        for f in claude_result.get("red_flags", []):
            if f and f not in reasons:
                reasons.append(f)

    risk = round(max(0.0, min(100.0, risk)), 2)
    is_scam = risk >= 50.0
    label, message = _risk_label(risk)
    return risk, is_scam, message, reasons, pipeline


async def detect(
    raw_text: str,
    sender: Optional[str] = None,
) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    clean = preprocess_arabic(raw_text)
    classifier_prob, _ = _run_classifier(clean if clean else raw_text)

    # URL reputation runs in parallel with the (sync) classifier above.
    url_reports = await _gather_url_reports(raw_text)

    # Heuristic reasons from raw text (keywords, urgency etc.).
    reasons_local = heuristic_flags(raw_text)

    # Claude only when uncertain AND key configured.
    claude_result = None
    if claude_fallback.is_enabled() and CLAUDE_LOW <= classifier_prob <= CLAUDE_HIGH:
        try:
            claude_result = await asyncio.to_thread(
                claude_fallback.consult,
                raw_text,
                sender,
                [r.to_dict() for r in url_reports],
                reasons_local,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Claude consult failed: %s", exc)

    risk, is_scam, message, reasons_pipeline, steps = _fuse(classifier_prob, url_reports, claude_result)

    # Combine local heuristics with the pipeline's reasons; dedup, keep order.
    seen: set[str] = set()
    reasons: list[str] = []
    for r in (reasons_local + reasons_pipeline):
        if r and r not in seen:
            reasons.append(r)
            seen.add(r)
    if not reasons:
        if is_scam:
            reasons.append("تعرّف الذكاء الاصطناعي على أسلوب صياغة مشبوه.")
        else:
            reasons.append("لم يتم العثور على أنماط مشبوهة أو روابط خبيثة.")

    return PredictionResponse(
        is_scam=is_scam,
        risk_level=_risk_label(risk)[0],
        confidence=risk,
        message=message,
        reasons=reasons,
        urls=[URLReportOut(**r.to_dict()) for r in url_reports],
        used_claude=claude_result is not None,
        pipeline=steps,
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
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
        claude_enabled=claude_fallback.is_enabled(),
        safe_browsing_enabled=bool(os.environ.get("GOOGLE_SAFE_BROWSING_KEY")),
        virustotal_enabled=bool(os.environ.get("VIRUSTOTAL_API_KEY")),
    )


@app.get("/model/version", response_model=ModelVersionResponse)
def model_version():
    return ModelVersionResponse(
        model_version=MODEL_VERSION,
        model_name=MODEL_NAME,
        model_loaded=model is not None,
        download_url=MODEL_DOWNLOAD_URL or None,
        changelog_ar=MODEL_CHANGELOG_AR,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sms(request: SMSRequest):
    return await detect(request.text, request.sender)


@app.post("/analyze", response_model=PredictionResponse)
async def analyze_message(request: SMSRequest):
    return await detect(request.text, request.sender)


@app.post("/ocr-predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")

    tess = _try_load_tesseract()
    if tess is None:
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
                "urls": [],
                "used_claude": False,
                "pipeline": [],
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
                reasons=["لم يتم العثور على نص واضح في هذه الصورة."],
                model_version=MODEL_VERSION,
            )
        return await detect(text)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}")


# ---------------------------------------------------------------------------
# Feedback & OTA
# ---------------------------------------------------------------------------
@app.post("/feedback")
async def submit_feedback(payload: dict):
    """
    User feedback loop (Section 3.6). For now we append to a JSONL file on
    Railway disk; Firestore wiring lands in Part 3.
    """
    payload = dict(payload or {})
    payload["model_version"] = MODEL_VERSION
    try:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("feedback write failed: %s", exc)
    return {"status": "received"}


@app.get("/model/download")
def model_download():
    """
    OTA model download. Redirects to the storage URL set via env if
    configured; otherwise streams the locally-loaded pkl. The mobile
    client compares /model/version then GETs this endpoint.
    """
    if MODEL_DOWNLOAD_URL:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(MODEL_DOWNLOAD_URL, status_code=302)
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="No model available for download.")
    from fastapi.responses import FileResponse
    return FileResponse(
        MODEL_PATH,
        media_type="application/octet-stream",
        filename=f"shayekli_model_v{MODEL_VERSION}.bin",
    )
