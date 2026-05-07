"""
Arabic preprocessing pipeline used by every detection path.

Per Section 2.3 of the upgrade spec:
  - dediac_ar          (remove tashkeel)
  - normalize_alef_ar  (collapse hamza/alef variants)
  - normalize_teh_marbuta_ar
  - alef-maqsura -> ya
  - collapse runs (هههههه -> هه)
  - keep Arabic, digits, basic latin, whitespace

camel-tools provides the canonical implementations but it's a heavy
optional dep (~200 MB of MADAMIRA models). We try to import it and
silently fall back to a regex-only equivalent if it's missing — same
output for the operations we need, just without camel-tools' broader
NLP features.
"""

from __future__ import annotations

import re
import logging
import unicodedata

log = logging.getLogger("shayekli.preprocess")

# ---------------------------------------------------------------------------
# Optional camel-tools backend
# ---------------------------------------------------------------------------
_CAMEL_OK = False
try:
    from camel_tools.utils.normalize import (
        normalize_alef_ar,
        normalize_teh_marbuta_ar,
        normalize_alef_maksura_ar,
    )
    from camel_tools.utils.dediac import dediac_ar
    _CAMEL_OK = True
    log.info("camel-tools loaded — using full Arabic normalization.")
except Exception as exc:  # noqa: BLE001
    log.warning("camel-tools unavailable (%s) — using regex fallback.", exc)

# ---------------------------------------------------------------------------
# Regex fallbacks (mirror camel-tools behavior for our specific ops)
# ---------------------------------------------------------------------------
_TASHKEEL_RE = re.compile(r"[ً-ٰٟؐ-ؚۖ-ۭ]")
_TATWEEL = "ـ"  # ـ
_ALEF_VARIANTS = re.compile(r"[آأإٱٲٳ]")  # آ أ إ ٱ ٲ ٳ
_ALEF_MAKSURA_RE = re.compile(r"ى")  # ى -> ي
_TEH_MARBUTA_RE = re.compile(r"ة")    # ة -> ه

# Keep Arabic block, digits (Arabic + ASCII), basic latin letters, spaces, common punctuation.
_KEEP_RE = re.compile(
    r"[^"
    r"؀-ۿ"   # Arabic
    r"ݐ-ݿ"   # Arabic Supplement
    r"ﭐ-﷿"   # Arabic Presentation Forms-A
    r"ﹰ-﻿"   # Arabic Presentation Forms-B
    r"a-zA-Z0-9"
    r"\s"
    r"\.\,\!\?\:\;\-\_\/\@\#\$\&\+\=\(\)\[\]"
    r"]"
)
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")
_MULTI_WS_RE = re.compile(r"\s+")


def _regex_pipeline(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL_RE.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF_VARIANTS.sub("ا", text)         # → ا
    text = _ALEF_MAKSURA_RE.sub("ي", text)       # ى → ي
    text = _TEH_MARBUTA_RE.sub("ه", text)        # ة → ه
    text = _REPEATED_CHARS_RE.sub(r"\1\1", text)
    text = _KEEP_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def _camel_pipeline(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = dediac_ar(text)
    text = text.replace(_TATWEEL, "")
    text = normalize_alef_ar(text)
    text = normalize_teh_marbuta_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = _REPEATED_CHARS_RE.sub(r"\1\1", text)
    text = _KEEP_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def preprocess_arabic(text: str) -> str:
    """
    Canonical Arabic normalization pipeline.

    Idempotent: preprocess(preprocess(x)) == preprocess(x).
    Safe for empty / None input.
    """
    if not text:
        return ""
    if _CAMEL_OK:
        try:
            return _camel_pipeline(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("camel-tools pipeline failed (%s) — falling back to regex.", exc)
    return _regex_pipeline(text)


# ---------------------------------------------------------------------------
# URL extraction (handles Arabic-surrounded / RTL-mixed strings)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"(?ix)"
    r"\b("
    r"(?:https?://|www\.)"           # scheme or www
    r"[^\s؀-ۿ<>\"\']+"     # stop at whitespace, Arabic, or quotes
    r")"
)


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from a possibly-bidi Arabic message.

    Strips trailing Arabic punctuation that often hugs URLs in RTL text
    (e.g. "اضغط الرابط https://example.com.").
    """
    if not text:
        return []
    found = _URL_RE.findall(text)
    out: list[str] = []
    for url in found:
        url = url.rstrip(".,;:!?)،؛؟")  # also Arabic comma/semicolon/?
        if not url:
            continue
        if url.lower().startswith("www."):
            url = "http://" + url
        out.append(url)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    uniq: list[str] = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq
