"""
Sender phone-number reputation engine (Section 3.4 of the upgrade spec).

Two layers:

  1. Heuristic (always on, free):
       - International prefix mismatch with the message's claimed
         local origin (e.g. an "بنك فلسطين" message sent from a +44
         number scores highly suspicious).
       - Premium-rate / known-spam country prefixes.
       - Disposable-VoIP and toll-free prefixes (often used by mass
         smishing campaigns).
       - SMS short-code shape vs personal number shape.

  2. Community database (Firestore, opt-in):
       - When a user submits feedback that a sender is a scammer,
         we persist the (hashed) number to Firestore. Future lookups
         for that number return a "reported by N users" flag.
       - Falls back to local in-memory cache when Firestore isn't
         configured — reputation still works for the lifetime of
         the process.

Output: a small dataclass we fold into the URLReport-style flag list
emitted by the main detection pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Optional

log = logging.getLogger("shayekli.sender_rep")

# ---------------------------------------------------------------------------
# Country / prefix tables
# ---------------------------------------------------------------------------
# Palestinian / regional local prefixes — anything starting with these is
# considered "local" for the purpose of impersonation checks.
LOCAL_PREFIXES = (
    "+970", "+972",          # Palestine / Israel
    "+962",                  # Jordan
    "+961",                  # Lebanon
    "+963",                  # Syria
    "+20",                   # Egypt
)

# Known abuse-heavy international ranges. Not exhaustive — high-precision
# anchors only. Scoring is conservative.
ABUSE_PREFIXES = {
    "+44 7": "UK mobile (high smishing volume)",
    "+1 ":   "North America (frequent number-spoofing source)",
    "+880":  "Bangladesh",
    "+234":  "Nigeria",
    "+92":   "Pakistan",
    "+91":   "India",
}

# Premium-rate / VoIP-ish that often surface in scams.
SUSPICIOUS_PREFIXES = (
    "+800", "+808",          # international toll-free
    "+881", "+882", "+883",  # global mobile satellite / VoIP allocations
    "+979",                  # international premium rate
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
@dataclass
class SenderReport:
    sender: str
    normalized: str
    country_prefix: Optional[str] = None
    is_local: bool = False
    is_short_code: bool = False
    risk_score: int = 0
    flags_ar: list[str] = field(default_factory=list)
    community_reports: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------------
_DIGITS_RE = re.compile(r"[^\d+]")


def normalize(number: str) -> str:
    if not number:
        return ""
    n = number.strip()
    n = _DIGITS_RE.sub("", n)
    if n.startswith("00"):
        n = "+" + n[2:]
    elif n.startswith("0") and len(n) > 8:
        # Palestine / region default — assume +970 if user dialed local trunk.
        n = "+970" + n[1:]
    return n


def _hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Optional persistent backends (Postgres preferred, then Firestore, then local)
# ---------------------------------------------------------------------------
_firestore_client = None
_community_local: dict[str, int] = {}
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
_pg_pool = None
_pg_init_done = False


def _firestore() -> "object | None":
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client
    try:
        if not os.environ.get("FIRESTORE_PROJECT_ID"):
            return None
        from google.cloud import firestore  # type: ignore
        _firestore_client = firestore.Client(project=os.environ["FIRESTORE_PROJECT_ID"])
        log.info("Firestore initialized for project %s", os.environ["FIRESTORE_PROJECT_ID"])
        return _firestore_client
    except Exception as exc:  # noqa: BLE001
        log.warning("Firestore unavailable: %s", exc)
        return None


def _pg() -> "object | None":
    """Postgres connection pool — preferred over Firestore + local."""
    global _pg_pool, _pg_init_done
    if not DATABASE_URL:
        return None
    if _pg_pool is not None:
        return _pg_pool
    try:
        from psycopg2.pool import SimpleConnectionPool  # type: ignore
        _pg_pool = SimpleConnectionPool(1, 5, DATABASE_URL)
    except Exception as exc:  # noqa: BLE001
        log.warning("Postgres unavailable for sender_reputation: %s", exc)
        return None
    if not _pg_init_done:
        try:
            conn = _pg_pool.getconn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sender_reputation (
                        sender_hash TEXT PRIMARY KEY,
                        category    TEXT NOT NULL DEFAULT 'scam',
                        count       INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            _pg_pool.putconn(conn)
            _pg_init_done = True
            log.info("sender_reputation Postgres schema ready.")
        except Exception as exc:  # noqa: BLE001
            log.error("sender_reputation schema init failed: %s", exc)
            return None
    return _pg_pool


def report_sender(number: str, category: str = "scam") -> int:
    """
    Increment the community report counter for a number. Returns the
    new count. Used by /feedback when a user confirms "yes this was a scam".
    """
    norm = normalize(number)
    if not norm:
        return 0
    digest = _hash(norm)

    # Postgres-first (atomic UPSERT, persistent across deploys).
    pool = _pg()
    if pool is not None:
        conn = pool.getconn()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sender_reputation (sender_hash, category, count)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (sender_hash) DO UPDATE
                        SET count = sender_reputation.count + 1
                    RETURNING count
                    """,
                    (digest, category),
                )
                return int(cur.fetchone()[0])
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres sender write failed: %s", exc)
        finally:
            pool.putconn(conn)

    fc = _firestore()
    if fc is not None:
        try:
            from google.cloud.firestore_v1.transforms import Increment  # type: ignore
            doc = fc.collection("sender_reputation").document(digest)
            doc.set({"category": category, "count": Increment(1)}, merge=True)
            snap = doc.get()
            return int(snap.get("count") or 1) if snap.exists else 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Firestore write failed for sender %s: %s", digest, exc)
    _community_local[digest] = _community_local.get(digest, 0) + 1
    return _community_local[digest]


def _community_count(normalized: str) -> int:
    digest = _hash(normalized)

    pool = _pg()
    if pool is not None:
        conn = pool.getconn()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count FROM sender_reputation WHERE sender_hash = %s",
                    (digest,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:  # noqa: BLE001
            log.debug("Postgres sender read failed: %s", exc)
        finally:
            pool.putconn(conn)

    fc = _firestore()
    if fc is not None:
        try:
            doc = fc.collection("sender_reputation").document(digest).get()
            if doc.exists:
                return int(doc.get("count") or 0)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.debug("Firestore read failed: %s", exc)
    return _community_local.get(digest, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_sender(sender: Optional[str], message_text: str = "") -> Optional[SenderReport]:
    """
    Score a sender. Returns None if `sender` is empty or obviously not
    a phone number (alphanumeric brand IDs like "JAWWAL" don't get
    a reputation report — they go through the regular text classifier).
    """
    if not sender:
        return None
    norm = normalize(sender)
    if not norm:
        return None
    if not norm.startswith("+") and not norm.isdigit():
        return None
    report = SenderReport(sender=sender, normalized=norm)

    # Country prefix detection (longest-match wins).
    for cc in sorted(LOCAL_PREFIXES, key=len, reverse=True):
        if norm.startswith(cc):
            report.country_prefix = cc
            report.is_local = True
            break
    if not report.country_prefix:
        for cc in sorted(SUSPICIOUS_PREFIXES, key=len, reverse=True):
            if norm.startswith(cc):
                report.country_prefix = cc
                break

    # Short-code shape — very short numbers used by aggregators.
    if len(norm.lstrip("+")) <= 6:
        report.is_short_code = True

    # Heuristic scoring.
    score = 0

    if report.country_prefix in SUSPICIOUS_PREFIXES:
        score += 35
        report.flags_ar.append("الرقم يستخدم بادئة دولية تُستخدم كثيراً في عمليات الاحتيال.")

    if not report.is_local and not report.is_short_code:
        # Foreign number sending Arabic banking-style message → red flag.
        if message_text and re.search(
            r"(بنك|البنك|بطاقة|حسابك|بالتل|جوال|ooredoo|paltel|jawwal|عميل|الكهرباء|البريد|فلسطين)",
            message_text,
            re.IGNORECASE,
        ):
            score += 30
            report.flags_ar.append(
                "رقم دولي يدّعي تمثيل جهة محلية فلسطينية — مؤشر احتيال قوي."
            )

    for prefix, label in ABUSE_PREFIXES.items():
        if norm.startswith(prefix.replace(" ", "")):
            score += 12
            report.flags_ar.append(f"الرقم من نطاق ({label}).")
            break

    # Community database.
    count = _community_count(norm)
    if count > 0:
        report.community_reports = count
        score += min(40, 10 + 5 * count)
        report.flags_ar.append(f"تم الإبلاغ عن هذا الرقم من قِبَل {count} مستخدم.")

    report.risk_score = max(0, min(100, score))
    return report
