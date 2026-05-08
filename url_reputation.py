"""
Multi-signal URL reputation engine (Section 2.5 of the upgrade spec).

For every URL we run, in parallel:
  1. Unshorten (follow ≤5 redirects, 3s budget).
  2. Extract registrable domain + TLD.
  3. Google Safe Browsing API (free, 10k/day) — authoritative "DANGEROUS"
     verdict if anything matches.
  4. Domain age via WHOIS — flag if < 30 days.
  5. PhishTank local cache (synced daily) — exact-domain match.
  6. VirusTotal as a tiebreaker when uncertain (free quota: 4 req/min).

All HTTP keys come from env vars. If none are set the engine still
runs and returns a usable risk score from heuristics + PhishTank +
domain-age alone — the cloud APIs just upgrade signal quality when
available.

Output: a dict per URL containing the individual signals plus a
combined 0–100 risk score and a list of human-readable Arabic flags
that get folded into the final detection response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx

log = logging.getLogger("shayekli.url_reputation")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAFE_BROWSING_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_KEY", "").strip()
VIRUSTOTAL_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "").strip()
PHISHTANK_CACHE_PATH = Path(os.environ.get("PHISHTANK_CACHE", "phishtank_cache.json"))
PHISHTANK_FEED = os.environ.get(
    "PHISHTANK_FEED",
    "https://data.phishtank.com/data/online-valid.json",
)
PHISHTANK_REFRESH_HOURS = int(os.environ.get("PHISHTANK_REFRESH_HOURS", "24"))
HTTP_TIMEOUT = float(os.environ.get("URL_HTTP_TIMEOUT", "3.0"))
UNSHORTEN_HOPS = int(os.environ.get("UNSHORTEN_HOPS", "5"))

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc", "lnkd.in",
    "trib.al", "qr.ae", "x.gd",
}

SUSPICIOUS_TLDS = {
    # Cheap / abuse-heavy TLDs frequently used by phishing kits.
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "buzz", "click", "rest",
    "country", "stream", "review", "loan", "trade", "win", "cricket",
    "science", "racing", "party", "date", "men", "work", "lol",
}

LOOKALIKE_KEYWORDS = (
    "paypal", "amazon", "apple", "icloud", "microsoft", "google", "facebook",
    "instagram", "whatsapp", "telegram", "bankofpalestine", "bop", "cab",
    "jawwal", "paltel", "ooredoo", "jordanahli", "arabbank", "post", "dhl",
    "aramex", "fedex", "ups",
)

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
@dataclass
class URLReport:
    original: str
    final_url: str
    domain: str
    tld: str
    risk_score: int = 0
    flags_ar: list[str] = field(default_factory=list)
    safe_browsing_threat: str | None = None
    virustotal_malicious: int = 0
    virustotal_suspicious: int = 0
    domain_age_days: int | None = None
    phishtank_match: bool = False
    is_shortener: bool = False
    redirect_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _domain_and_tld(url: str) -> tuple[str, str]:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "", ""
    if not host:
        return "", ""
    parts = host.split(".")
    if len(parts) < 2:
        return host, ""
    return host, parts[-1]


async def _unshorten(client: httpx.AsyncClient, url: str) -> tuple[str, int]:
    """Follow up to UNSHORTEN_HOPS redirects manually so we can count them."""
    current = url
    hops = 0
    for _ in range(UNSHORTEN_HOPS):
        try:
            resp = await client.head(current, follow_redirects=False, timeout=HTTP_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            log.debug("HEAD %s failed: %s", current, exc)
            return current, hops
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location")
            if not loc:
                return current, hops
            # Resolve relative redirects.
            current = httpx.URL(current).join(loc).human_repr()
            hops += 1
            continue
        return current, hops
    return current, hops


async def _retrying_post(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """
    POST with up to 3 attempts on transient failures (timeouts, 5xx).
    Exponential backoff: 0.4s, 0.8s, 1.6s. Honoured everywhere we hit a
    third-party API so a single transient blip doesn't poison detection.
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.post(url, **kwargs)
            if resp.status_code < 500 or attempt == 2:
                return resp
            log.debug("retrying POST %s after %s (attempt=%s)", url, resp.status_code, attempt + 1)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            log.debug("retrying POST %s after %s (attempt=%s)", url, type(exc).__name__, attempt + 1)
        await asyncio.sleep(0.4 * (2 ** attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("retry path inconsistency")


async def _retrying_get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.get(url, **kwargs)
            if resp.status_code < 500 or attempt == 2:
                return resp
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
        await asyncio.sleep(0.4 * (2 ** attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("retry path inconsistency")


async def _safe_browsing(client: httpx.AsyncClient, url: str) -> str | None:
    """Returns the threat type (e.g. 'SOCIAL_ENGINEERING') or None."""
    if not SAFE_BROWSING_KEY:
        return None
    payload = {
        "client": {"clientId": "shayekli", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = await _retrying_post(
            client,
            f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={SAFE_BROWSING_KEY}",
            json=payload,
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning("Safe Browsing returned %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        matches = data.get("matches") or []
        if matches:
            return matches[0].get("threatType")
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Safe Browsing call failed: %s", exc)
        return None


async def _virustotal(client: httpx.AsyncClient, url: str) -> tuple[int, int]:
    """Returns (malicious_count, suspicious_count). Quota: 4 req/min."""
    if not VIRUSTOTAL_KEY:
        return 0, 0
    try:
        # VirusTotal v3 needs a base64url'd URL identifier without padding.
        import base64
        ident = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
        resp = await _retrying_get(
            client,
            f"https://www.virustotal.com/api/v3/urls/{ident}",
            headers={"x-apikey": VIRUSTOTAL_KEY},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            log.debug("VirusTotal returned %s for %s", resp.status_code, url)
            return 0, 0
        stats = resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {}) or {}
        return int(stats.get("malicious", 0)), int(stats.get("suspicious", 0))
    except Exception as exc:  # noqa: BLE001
        log.warning("VirusTotal call failed: %s", exc)
        return 0, 0


def _domain_age_days(domain: str) -> int | None:
    """
    WHOIS-based domain age. We use python-whois lazily so the dep stays
    optional; fall back to None if the lookup fails or the lib's missing.
    """
    try:
        import whois  # type: ignore
    except Exception:
        return None
    try:
        info = whois.whois(domain)
        created = info.creation_date
        if isinstance(created, list):
            created = created[0] if created else None
        if not created:
            return None
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except Exception:
                return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created
        return max(0, delta.days)
    except Exception as exc:  # noqa: BLE001
        log.debug("WHOIS for %s failed: %s", domain, exc)
        return None


# ---------------------------------------------------------------------------
# PhishTank cache (refresh once per process, persist to disk)
# ---------------------------------------------------------------------------
_phishtank_cache: set[str] = set()
_phishtank_loaded_at: float = 0.0


def _load_phishtank_from_disk() -> None:
    global _phishtank_cache, _phishtank_loaded_at
    if not PHISHTANK_CACHE_PATH.exists():
        return
    try:
        raw = json.loads(PHISHTANK_CACHE_PATH.read_text(encoding="utf-8"))
        _phishtank_cache = set(raw.get("domains", []))
        _phishtank_loaded_at = float(raw.get("loaded_at", 0.0))
        log.info("Loaded %s phishtank entries from disk.", len(_phishtank_cache))
    except Exception as exc:  # noqa: BLE001
        log.warning("PhishTank cache read failed: %s", exc)


async def _refresh_phishtank(client: httpx.AsyncClient) -> None:
    global _phishtank_cache, _phishtank_loaded_at
    age_h = (time.time() - _phishtank_loaded_at) / 3600.0
    if _phishtank_cache and age_h < PHISHTANK_REFRESH_HOURS:
        return
    try:
        resp = await client.get(PHISHTANK_FEED, timeout=20.0)
        if resp.status_code != 200:
            log.warning("PhishTank feed status %s", resp.status_code)
            return
        data = resp.json()
        domains = set()
        for row in data:
            url = row.get("url") or ""
            host = urlparse(url).hostname
            if host:
                domains.add(host.lower())
        _phishtank_cache = domains
        _phishtank_loaded_at = time.time()
        try:
            PHISHTANK_CACHE_PATH.write_text(
                json.dumps({"loaded_at": _phishtank_loaded_at, "domains": sorted(domains)}),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("PhishTank cache write failed: %s", exc)
        log.info("Refreshed PhishTank cache (%s domains).", len(domains))
    except Exception as exc:  # noqa: BLE001
        log.warning("PhishTank refresh failed: %s", exc)


def _phishtank_match(domain: str) -> bool:
    if not _phishtank_cache:
        return False
    if domain in _phishtank_cache:
        return True
    # Match parent domain too (sub.evil.tk → evil.tk).
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        if ".".join(parts[i:]) in _phishtank_cache:
            return True
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score(report: URLReport) -> None:
    score = 0
    flags: list[str] = []

    if report.safe_browsing_threat:
        score += 75
        flags.append("هذا الرابط مصنّف خطير لدى Google Safe Browsing.")

    if report.phishtank_match:
        score += 60
        flags.append("هذا النطاق ضمن قاعدة بيانات PhishTank للروابط الاحتيالية.")

    if report.virustotal_malicious >= 1:
        score += min(40, 15 + 5 * report.virustotal_malicious)
        flags.append(f"VirusTotal: {report.virustotal_malicious} محرّك أمني صنّف الرابط ضارّاً.")
    elif report.virustotal_suspicious >= 2:
        score += 20
        flags.append("عدة محركات أمنية تعتبر الرابط مشبوهاً.")

    if report.domain_age_days is not None and report.domain_age_days < 30:
        score += 25
        flags.append(f"النطاق جديد جداً (عمره {report.domain_age_days} يوماً).")

    if report.is_shortener:
        score += 10
        flags.append("الرابط استخدم خدمة اختصار، ما يخفي الوجهة الفعلية.")

    if report.redirect_count >= 3:
        score += 10
        flags.append("الرابط يمر بعدة تحويلات قبل الوجهة النهائية.")

    if report.tld in SUSPICIOUS_TLDS:
        score += 15
        flags.append(f"امتداد النطاق .{report.tld} شائع في عمليات الاحتيال.")

    # Lookalike: brand keyword in subdomain or hyphen-padded domain.
    dom = report.domain
    for kw in LOOKALIKE_KEYWORDS:
        if kw in dom and not dom.endswith(f".{kw}.com") and not dom == f"{kw}.com":
            # e.g. "paypal-secure-login.tk", "bankofpalestine.support.xyz"
            score += 20
            flags.append(f"النطاق يحاكي علامة معروفة ({kw}).")
            break

    # Raw-IP host or punycode → suspicious.
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", report.domain):
        score += 25
        flags.append("الرابط يستخدم عنوان IP بدلاً من اسم نطاق.")
    if "xn--" in report.domain:
        score += 15
        flags.append("النطاق يستخدم أحرفاً مشفّرة (Punycode) قد تخدع البصر.")

    report.risk_score = max(0, min(100, score))
    report.flags_ar = flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def analyze_urls(urls: Iterable[str]) -> list[URLReport]:
    """Run the full reputation chain in parallel for every URL."""
    urls = [u for u in dict.fromkeys(urls) if u]
    if not urls:
        return []

    async with httpx.AsyncClient(
        follow_redirects=False,
        headers={"User-Agent": "Shayekli-URLChecker/1.0"},
    ) as client:
        # Refresh PhishTank cache opportunistically (non-blocking failure).
        await _refresh_phishtank(client)

        async def one(url: str) -> URLReport:
            domain0, _ = _domain_and_tld(url)
            report = URLReport(original=url, final_url=url, domain=domain0, tld="")

            try:
                final_url, hops = await _unshorten(client, url)
                report.final_url = final_url
                report.redirect_count = hops
                domain, tld = _domain_and_tld(final_url)
                report.domain = domain
                report.tld = tld
                report.is_shortener = domain in KNOWN_SHORTENERS or _domain_and_tld(url)[0] in KNOWN_SHORTENERS

                gsb_task = asyncio.create_task(_safe_browsing(client, final_url))
                vt_task = asyncio.create_task(_virustotal(client, final_url))
                # WHOIS is sync — run in a thread so we don't block the loop.
                age_task = asyncio.create_task(asyncio.to_thread(_domain_age_days, domain))

                report.safe_browsing_threat = await gsb_task
                report.virustotal_malicious, report.virustotal_suspicious = await vt_task
                report.domain_age_days = await age_task
                report.phishtank_match = _phishtank_match(domain)
            except Exception as exc:  # noqa: BLE001
                report.error = str(exc)
                log.warning("URL analysis failed for %s: %s", url, exc)

            _score(report)
            return report

        return await asyncio.gather(*(one(u) for u in urls))


# Load cache eagerly at import time so the first request isn't slow.
_load_phishtank_from_disk()
