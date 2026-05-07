"""
Google Cloud Vision DOCUMENT_TEXT_DETECTION wrapper (Section 3.1 of the
upgrade spec).

Replaces the Tesseract path. We hit the REST API directly so the backend
doesn't pull google-cloud-vision (and its protobuf/grpc cascade) — saves
~80 MB on the Railway image.

Authenticates via API key passed in env (GOOGLE_CLOUD_VISION_KEY).
Returns extracted text or raises if the API call fails.

Designed to handle Arabic + bidi mixed text — Cloud Vision's
DOCUMENT_TEXT_DETECTION mode preserves reading order across RTL/LTR
boundaries far better than TEXT_DETECTION.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger("shayekli.cloud_vision")

API_KEY = os.environ.get("GOOGLE_CLOUD_VISION_KEY", "").strip()
ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
TIMEOUT = float(os.environ.get("CLOUD_VISION_TIMEOUT", "20"))


class CloudVisionUnavailable(RuntimeError):
    """Raised when no API key is configured."""


class CloudVisionError(RuntimeError):
    """Raised on API-level failure (4xx/5xx, malformed payload, etc.)."""


@dataclass
class OCRResult:
    text: str
    language_hint: str | None = None
    raw_languages: list[str] | None = None


def is_enabled() -> bool:
    return bool(API_KEY)


async def ocr_image_bytes(image_bytes: bytes) -> OCRResult:
    """
    Run DOCUMENT_TEXT_DETECTION on raw image bytes. Returns extracted
    text (preserving reading order) plus detected language hints.
    """
    if not API_KEY:
        raise CloudVisionUnavailable("GOOGLE_CLOUD_VISION_KEY is not configured.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "requests": [
            {
                "image": {"content": encoded},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                # Hint Arabic + English so the bidi reordering stays sane.
                "imageContext": {"languageHints": ["ar", "en"]},
            }
        ]
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.post(f"{ENDPOINT}?key={API_KEY}", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise CloudVisionError(f"network error: {exc}") from exc

    if resp.status_code != 200:
        raise CloudVisionError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        body = resp.json()
        responses = body.get("responses", []) or []
        first = responses[0] if responses else {}
        if "error" in first:
            raise CloudVisionError(str(first["error"]))
        full = first.get("fullTextAnnotation") or {}
        text = (full.get("text") or "").strip()
        # Pull out detected languages so we can short-circuit non-Arabic later.
        langs: list[str] = []
        for page in full.get("pages") or []:
            for prop in (page.get("property", {}).get("detectedLanguages") or []):
                code = prop.get("languageCode")
                if code and code not in langs:
                    langs.append(code)
        primary = langs[0] if langs else None
        return OCRResult(text=text, language_hint=primary, raw_languages=langs)
    except CloudVisionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CloudVisionError(f"malformed response: {exc}") from exc
