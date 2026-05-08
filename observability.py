"""
Phase 5 — observability: structured JSON logs, in-process metrics, and a
FastAPI middleware that captures per-request latency.

We deliberately log message *hashes* (SHA256, first 16 hex chars) rather
than the raw text so the logs never contain user PII. Same for sender
numbers — only the hash is logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque

# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Structured JSON log lines — one record per stdout line."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Pull through any structured `extra` fields the caller passed.
        for key in (
            "request_id", "route", "method", "status", "latency_ms",
            "model_version", "verdict", "confidence", "pipeline",
            "msg_hash", "sender_hash", "url_count", "claude_used",
            "error_type",
        ):
            v = getattr(record, key, None)
            if v is not None:
                out[key] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def install_json_logging(level: int = logging.INFO) -> None:
    """Replace the root logger's handlers with a single JSON-formatting one."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Hashing helpers (so logs never contain raw text)
# ---------------------------------------------------------------------------
def hash_text(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_sender(sender: str | None) -> str | None:
    if not sender:
        return None
    norm = "".join(c for c in sender if c.isdigit() or c == "+")
    if not norm:
        return None
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# In-process metrics
# ---------------------------------------------------------------------------
# Lightweight counters + a sliding-window latency buffer per route. No
# external monitoring service required; surfaced via /metrics.
_counters: dict[str, int] = defaultdict(int)
_latencies: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=512))
_started_at = time.time()


def incr(name: str, by: int = 1) -> None:
    _counters[name] += by


def observe_latency(route: str, latency_ms: float) -> None:
    _latencies[route].append(latency_ms)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return round(s[k], 2)


def snapshot() -> dict:
    """Return everything /metrics exposes."""
    out: dict[str, Any] = {
        "uptime_seconds": int(time.time() - _started_at),
        "counters": dict(_counters),
        "routes": {},
    }
    for route, samples in _latencies.items():
        vals = list(samples)
        if not vals:
            continue
        out["routes"][route] = {
            "samples": len(vals),
            "p50_ms": _percentile(vals, 50),
            "p95_ms": _percentile(vals, 95),
            "p99_ms": _percentile(vals, 99),
            "max_ms": round(max(vals), 2),
        }
    # Derived signals the spec calls out explicitly.
    total = _counters.get("predict_total", 0)
    errors = _counters.get("predict_error", 0)
    out["error_rate"] = round((errors / total) * 100, 2) if total else 0.0
    fp = _counters.get("feedback_false_positive", 0)
    fn = _counters.get("feedback_confirm_scam", 0)
    out["feedback"] = {
        "confirm_scam": fn,
        "false_positive": fp,
        "fp_rate": round((fp / max(1, fp + fn)) * 100, 2),
    }
    return out


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------
def request_logging_middleware(app):
    """
    Middleware that records latency + emits a JSON log line per request.
    Idempotent — call once at app startup with `app.middleware('http')(...)`.
    """
    log = logging.getLogger("shayekli.http")
    request_seq = [0]

    @app.middleware("http")
    async def _mw(request, call_next):
        request_seq[0] += 1
        rid = f"r{request_seq[0]}"
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            route = request.url.path
            observe_latency(route, elapsed_ms)
            incr(f"req_{route}")
            if status >= 500:
                incr("req_5xx")
            elif status >= 400:
                incr("req_4xx")
            else:
                incr("req_2xx")
            log.info(
                "request",
                extra={
                    "request_id": rid,
                    "route": route,
                    "method": request.method,
                    "status": status,
                    "latency_ms": elapsed_ms,
                },
            )

    return app
