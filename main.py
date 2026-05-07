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
import cloud_vision
import sender_reputation
import community_db

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


class SenderReportOut(BaseModel):
    sender: str
    normalized: str
    country_prefix: Optional[str] = None
    is_local: bool = False
    is_short_code: bool = False
    risk_score: int = 0
    flags_ar: List[str] = []
    community_reports: int = 0


class PredictionResponse(BaseModel):
    is_scam: bool
    risk_level: str
    confidence: float
    message: str
    reasons: List[str]
    model_version: int = MODEL_VERSION
    urls: List[URLReportOut] = []
    sender_report: Optional[SenderReportOut] = None
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
    cloud_vision_enabled: bool = False


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

    # Community SimHash lookup — short-circuits the pipeline if this
    # message matches a previously reported scam template.
    community_match = community_db.lookup(raw_text)
    if community_match.matched and community_match.record is not None:
        rec = community_match.record
        sender_report = sender_reputation.analyze_sender(sender, raw_text)
        return PredictionResponse(
            is_scam=True,
            risk_level="HIGH",
            confidence=95.0,
            message="رسالة مُبلَّغ عنها سابقاً من قِبَل المجتمع",
            reasons=[
                f"تم الإبلاغ عن هذه الرسالة من قِبَل {rec.count} مستخدم.",
                "الصياغة تطابق نمط احتيال معروف في قاعدة بيانات شيّكلي.",
            ],
            urls=[],
            sender_report=SenderReportOut(**sender_report.to_dict()) if sender_report else None,
            used_claude=False,
            pipeline=["preprocess", "community_match"],
            model_version=MODEL_VERSION,
        )

    clean = preprocess_arabic(raw_text)
    classifier_prob, _ = _run_classifier(clean if clean else raw_text)

    # URL reputation runs in parallel with the (sync) classifier above.
    url_reports = await _gather_url_reports(raw_text)

    # Heuristic reasons from raw text (keywords, urgency etc.).
    reasons_local = heuristic_flags(raw_text)

    # Sender reputation (Section 3.4) — heuristic + community DB.
    sender_report = sender_reputation.analyze_sender(sender, raw_text)
    if sender_report and sender_report.flags_ar:
        for f in sender_report.flags_ar:
            if f not in reasons_local:
                reasons_local.append(f)

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

    # Fold the sender's risk into the final score (modest weight).
    if sender_report and sender_report.risk_score:
        risk = max(risk, min(100.0, risk + sender_report.risk_score * 0.3))
        is_scam = risk >= 50.0
        _, message = _risk_label(risk)
        steps.append("sender_reputation")

    return PredictionResponse(
        is_scam=is_scam,
        risk_level=_risk_label(risk)[0],
        confidence=round(risk, 2),
        message=message,
        reasons=reasons,
        urls=[URLReportOut(**r.to_dict()) for r in url_reports],
        sender_report=SenderReportOut(**sender_report.to_dict()) if sender_report else None,
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
        cloud_vision_enabled=cloud_vision.is_enabled(),
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
    """
    Image scam detection. OCR pipeline:
      1. Google Cloud Vision (DOCUMENT_TEXT_DETECTION) — preferred.
      2. Tesseract — only if installed locally (dev fallback).
      3. Otherwise return a friendly Arabic 503 explaining we need network.
    Then runs the extracted text through the same `detect()` chain as
    a plain text message, so URL reputation + Claude + classifier all
    apply to image content too.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")

    try:
        contents = await file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}")
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    text = ""
    ocr_engine = None

    # --- Try Cloud Vision first ---
    if cloud_vision.is_enabled():
        try:
            result = await cloud_vision.ocr_image_bytes(contents)
            text = (result.text or "").strip()
            ocr_engine = "cloud_vision"
            log.info("Cloud Vision OCR: %s chars (langs=%s)", len(text), result.raw_languages)
        except cloud_vision.CloudVisionUnavailable:
            pass  # fall through to tesseract
        except Exception as exc:  # noqa: BLE001
            log.warning("Cloud Vision OCR failed: %s", exc)

    # --- Tesseract fallback (dev only — won't exist on Railway) ---
    if not text:
        tess = _try_load_tesseract()
        if tess is not None:
            try:
                image = Image.open(io.BytesIO(contents))
                text = (tess.image_to_string(image, lang="ara") or "").strip()
                ocr_engine = "tesseract"
            except Exception as exc:  # noqa: BLE001
                log.warning("Tesseract OCR failed: %s", exc)

    # --- Neither available ---
    if not text and ocr_engine is None:
        if not cloud_vision.is_enabled():
            return JSONResponse(
                status_code=503,
                content={
                    "is_scam": False,
                    "risk_level": "LOW",
                    "confidence": 0.0,
                    "message": "فحص الصور يتطلب اتصالاً بالإنترنت",
                    "reasons": [
                        "لم يتم تفعيل خدمة استخراج النص السحابية. يرجى المحاولة لاحقاً أو استخدام فحص النص.",
                    ],
                    "model_version": MODEL_VERSION,
                    "urls": [],
                    "used_claude": False,
                    "pipeline": [],
                },
            )
        return JSONResponse(
            status_code=502,
            content={
                "is_scam": False,
                "risk_level": "LOW",
                "confidence": 0.0,
                "message": "تعذّر فحص الصورة",
                "reasons": ["تعذّر استخراج النص من الصورة الآن. حاول مرة أخرى."],
                "model_version": MODEL_VERSION,
                "urls": [],
                "used_claude": False,
                "pipeline": [],
            },
        )

    if not text:
        return PredictionResponse(
            is_scam=False,
            risk_level="LOW",
            confidence=0.0,
            message="تعذر استخراج نص",
            reasons=["لم يتم العثور على نص واضح في هذه الصورة."],
            model_version=MODEL_VERSION,
            pipeline=[f"ocr:{ocr_engine}"] if ocr_engine else [],
        )

    response = await detect(text)
    if ocr_engine:
        response.pipeline = [f"ocr:{ocr_engine}", *response.pipeline]
    return response


# ---------------------------------------------------------------------------
# Feedback & OTA
# ---------------------------------------------------------------------------
@app.post("/feedback")
async def submit_feedback(payload: dict):
    """
    User feedback loop (Section 3.6). Persists the feedback to a JSONL
    file and, when the user confirmed a scam AND included a sender,
    increments the community sender-reputation counter.
    """
    payload = dict(payload or {})
    payload["model_version"] = MODEL_VERSION
    try:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("feedback write failed: %s", exc)

    new_count = None
    community_count = None
    try:
        verdict = str(payload.get("verdict", "")).lower()
        if verdict in ("scam", "confirm_scam"):
            sender = payload.get("sender") or ""
            if sender:
                new_count = sender_reputation.report_sender(sender, category="scam")
            text = payload.get("text") or ""
            if text:
                rec = community_db.report_scam(text, category=payload.get("category", "scam"))
                community_count = rec.count
    except Exception as exc:  # noqa: BLE001
        log.warning("feedback updates failed: %s", exc)

    return {
        "status": "received",
        "sender_report_count": new_count,
        "community_report_count": community_count,
    }


@app.get("/community/stats")
def community_stats():
    return community_db.stats()


@app.post("/community/debug-strip")
async def community_debug_strip(payload: dict):
    """
    Visibility helper for the SimHash matching path. Returns the stripped
    template + 64-bit hash + nearest stored neighbor's Hamming distance.
    Useful when a near-duplicate fails to match and we need to see why.
    """
    text = (payload or {}).get("text") or ""
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    template = community_db.strip_personal_data(text)
    h = community_db.simhash(template)
    match = community_db.lookup(text)
    return {
        "input": text,
        "stripped_template": template,
        "simhash": h,
        "matched": match.matched,
        "nearest_distance": match.distance,
        "nearest_record": match.record.to_dict() if match.record else None,
    }


@app.get("/blocklist/v1")
async def blocklist_v1():
    """
    Daily-synced blocklist used by the on-device VpnService (Section 3.3).
    Combines:
      - PhishTank cache (already refreshed in url_reputation)
      - Static seed of high-volume Palestinian-context phishing TLDs.

    Response shape is intentionally tiny so the device can cache it:
        { "version": <epoch>, "domains": ["evil.tk", "phish.xyz", ...] }
    """
    from url_reputation import _phishtank_cache, _refresh_phishtank
    import time, httpx
    # Best-effort refresh; ignore failures.
    try:
        async with httpx.AsyncClient() as client:
            await _refresh_phishtank(client)
    except Exception:  # noqa: BLE001
        pass
    domains = sorted(_phishtank_cache)
    # Cap payload — devices don't need the full feed, top N is plenty.
    cap = int(os.environ.get("BLOCKLIST_CAP", "20000"))
    if len(domains) > cap:
        domains = domains[:cap]
    return {
        "version": int(time.time()),
        "count": len(domains),
        "domains": domains,
    }


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
