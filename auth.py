"""
Phase 5.3 — optional API key auth.

When BACKEND_API_KEY is set in the environment, every request must include
a matching `X-API-Key` header (or `Authorization: Bearer <key>`) or the
server responds 401. When the env var is empty / unset, auth is OFF — the
backend behaves exactly like before. This makes auth opt-in: you can ship
the public app first, then flip auth on by setting the env var on Railway
(and bumping the same value into the app's `app.config.js` extra).

Endpoints exempted from auth:
    /         /health         /ping         /docs   /openapi.json
    /redoc    /docs/oauth2-redirect

Everything else (predict, analyze, feedback, blocklist, heuristics, etc.)
requires the key when enabled.
"""

from __future__ import annotations

import os
from typing import Iterable

from fastapi import HTTPException, Request

# Paths that never require auth — health checks, schema, root.
PUBLIC_PATHS: set[str] = {
    "/", "/health", "/ping",
    "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect",
}


def _expected_key() -> str:
    return os.environ.get("BACKEND_API_KEY", "").strip()


def is_enabled() -> bool:
    return bool(_expected_key())


def _extract_key(request: Request) -> str | None:
    # Preferred header.
    val = request.headers.get("x-api-key")
    if val:
        return val.strip()
    # Bearer fallback so we play nicely with anything that already speaks Authorization.
    auth = request.headers.get("authorization")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


def install(app, public_extra: Iterable[str] = ()) -> None:
    """Attach the auth middleware to a FastAPI app."""
    public = PUBLIC_PATHS | set(public_extra)

    @app.middleware("http")
    async def _auth_mw(request: Request, call_next):
        expected = _expected_key()
        if not expected:
            # Auth disabled — pass through.
            return await call_next(request)
        if request.url.path in public:
            return await call_next(request)
        provided = _extract_key(request)
        if provided != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        return await call_next(request)
