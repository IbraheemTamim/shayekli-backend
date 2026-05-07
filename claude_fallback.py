"""
Claude API second-opinion engine for uncertain detections (Section 2.6).

Fires only when the primary classifier returns 40–65% confidence — the
"can go either way" band where Claude's judgment beats the local model.
Cost-managed by design: confidently-safe and confidently-scam messages
never invoke the API.

Returns a dict that the main pipeline merges into the response, OR
None on any failure (silent degradation — the local model's verdict
stays authoritative).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger("shayekli.claude")

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "").strip()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "600"))

UNCERTAIN_LOW = float(os.environ.get("CLAUDE_FALLBACK_LOW", "0.40"))
UNCERTAIN_HIGH = float(os.environ.get("CLAUDE_FALLBACK_HIGH", "0.65"))

SYSTEM_PROMPT = (
    "You are an Arabic SMS scam detection expert specializing in Palestinian dialect. "
    "Analyze the following message and determine if it is a scam. Consider: urgency "
    "language, impersonation of local banks/government/telecoms, suspicious URLs, "
    "requests for personal data, and financial bait. Respond ONLY with a JSON object "
    "with these exact keys: "
    "{\"verdict\": \"scam\"|\"legitimate\"|\"uncertain\", "
    "\"confidence\": 0-100, "
    "\"red_flags\": [\"reason in Arabic\", ...]}. "
    "Do not include any prose outside the JSON."
)


def is_uncertain(scam_prob: float) -> bool:
    return UNCERTAIN_LOW <= scam_prob <= UNCERTAIN_HIGH


def is_enabled() -> bool:
    return bool(CLAUDE_API_KEY)


# ---------------------------------------------------------------------------
# Lazy SDK import — keep boot time fast and never crash if the dep is absent.
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not CLAUDE_API_KEY:
        return None
    try:
        from anthropic import Anthropic  # type: ignore
        _client = Anthropic(api_key=CLAUDE_API_KEY)
        return _client
    except Exception as exc:  # noqa: BLE001
        log.warning("anthropic SDK unavailable: %s", exc)
        return None


def _build_user_prompt(
    text: str,
    sender: str | None,
    url_reports: list[dict[str, Any]] | None,
    local_flags: list[str] | None,
) -> str:
    parts = ["MESSAGE:", text.strip(), ""]
    if sender:
        parts.append(f"SENDER: {sender}")
    if url_reports:
        parts.append("URL_REPORTS_JSON:")
        parts.append(json.dumps(url_reports, ensure_ascii=False))
    if local_flags:
        parts.append("LOCAL_RED_FLAGS_AR:")
        parts.append(json.dumps(local_flags, ensure_ascii=False))
    parts.append("")
    parts.append("Respond with the JSON object only.")
    return "\n".join(parts)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    candidate = raw.strip()
    # Models sometimes wrap the JSON in ```json ... ``` fences.
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        obj = json.loads(candidate)
    except Exception:
        m = _JSON_BLOCK_RE.search(candidate)
        if not m:
            log.warning("Claude response not parseable as JSON: %r", raw[:200])
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception as exc:
            log.warning("Claude JSON salvage failed: %s", exc)
            return None
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).lower().strip()
    if verdict not in ("scam", "legitimate", "uncertain"):
        verdict = "uncertain"
    try:
        confidence = float(obj.get("confidence", 50))
    except Exception:
        confidence = 50.0
    confidence = max(0.0, min(100.0, confidence))
    flags = obj.get("red_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(f).strip() for f in flags if str(f).strip()]
    return {"verdict": verdict, "confidence": confidence, "red_flags": flags}


def consult(
    text: str,
    sender: str | None = None,
    url_reports: list[dict[str, Any]] | None = None,
    local_flags: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Synchronous Claude consultation. Returns parsed verdict dict or None.

    Sync because it's called from the existing sync FastAPI handlers.
    The Anthropic SDK is itself blocking; if we ever go async we'd swap
    to AsyncAnthropic.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(text, sender, url_reports, local_flags)}],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Claude API call failed: %s", exc)
        return None

    try:
        # The SDK returns a list of content blocks.
        text_out = "".join(
            getattr(block, "text", "") for block in msg.content if getattr(block, "type", "") == "text"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Claude response shape unexpected: %s", exc)
        return None

    return _parse_response(text_out)
