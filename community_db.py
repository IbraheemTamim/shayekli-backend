"""
Community scam database (Section 3.5 of the upgrade spec).

When a user confirms "yes this was a scam" via /feedback, we strip
personal data (phone numbers, IBANs, account numbers, names mentioned
right after well-known prefixes), compute a SimHash of the resulting
template, and store the hash + scam category.

On every new analysis the pipeline first checks whether the incoming
message's SimHash is within Hamming-distance 3 of any reported scam.
A match short-circuits the rest of the pipeline with a high-confidence
"reported by N users" verdict.

Persistence backends, picked at runtime in this priority order:
  1. Postgres (when DATABASE_URL is set — Railway Postgres injects it).
     This is the recommended production path: persistent across deploys,
     atomic increments, scales across instances.
  2. Firestore (when FIRESTORE_PROJECT_ID is set).
  3. Local JSONL file + in-memory dict (default fallback — fine for dev,
     ephemeral on Railway because container filesystem doesn't persist).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from preprocess import preprocess_arabic

log = logging.getLogger("shayekli.community")

LOCAL_PATH = Path(os.environ.get("COMMUNITY_DB_PATH", "community_db.jsonl"))
SIMHASH_BITS = 64
HAMMING_THRESHOLD = int(os.environ.get("SIMHASH_HAMMING", "3"))
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


# ---------------------------------------------------------------------------
# Personal-data stripping
# ---------------------------------------------------------------------------
# Remove: phone numbers, account numbers, IBANs, OTPs, currency amounts,
# Arabic name-after-honorific patterns. Goal is the residual "template".
#
# IMPORTANT: these regexes run AFTER preprocess_arabic, which normalizes
# ة → ه, أإآٱ → ا, ى → ي, and strips diacritics. So the honorific list
# below has to use the post-normalized forms.
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", re.IGNORECASE)
# `\d+` (not `\d{1,3}`) so 4+ digit amounts like 5000, 7500 get fully eaten
# whether or not they have a thousands-separator.
_AMOUNT_RE = re.compile(
    r"\b\d+(?:[,\.]\d+)*\s*(?:₪|شيكل|دولار|دينار|\$|€|usd|jod|ils)?",
    re.IGNORECASE,
)
_OTP_RE = re.compile(r"\b\d{2,}\b")  # any leftover digit run >=2
_NAME_AFTER_HONORIFIC = re.compile(
    # Both pre-normalized and post-normalized forms — preprocess turns
    # `السيدة` into `السيده` and `الأستاذ` into `الاستاذ` etc.
    r"(?:"
    r"عزيزي|عزيزتي|"
    r"السيد|السيدة|السيده|"
    r"الأستاذ|الاستاذ|"
    r"الأخ|الاخ|"
    r"الأخت|الاخت"
    r")\s+\S+",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def strip_personal_data(text: str) -> str:
    if not text:
        return ""
    t = preprocess_arabic(text)
    t = _URL_RE.sub("URL", t)
    t = _IBAN_RE.sub("IBAN", t)
    t = _PHONE_RE.sub("PHONE", t)
    t = _AMOUNT_RE.sub("AMOUNT", t)
    t = _OTP_RE.sub("NUMBER", t)
    t = _NAME_AFTER_HONORIFIC.sub("HONORIFIC NAME", t)
    return t.strip()


# ---------------------------------------------------------------------------
# SimHash (64-bit, 3-gram features)
# ---------------------------------------------------------------------------
def _features(template: str) -> list[str]:
    # Word-level 3-grams from the stripped template, plus bigram chars
    # for the residual structure. This combo is robust to small reorderings.
    words = re.findall(r"\S+", template)
    grams: list[str] = []
    for i in range(max(0, len(words) - 2)):
        grams.append(" ".join(words[i:i + 3]))
    if not grams and words:
        grams = words
    # Char bigrams cover very short messages.
    flat = "".join(words)
    grams.extend(flat[i:i + 2] for i in range(len(flat) - 1))
    return grams


def simhash(template: str) -> int:
    if not template:
        return 0
    bits = [0] * SIMHASH_BITS
    for feat in _features(template):
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:16], 16)
        for b in range(SIMHASH_BITS):
            bits[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(SIMHASH_BITS):
        if bits[b] >= 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
@dataclass
class ScamRecord:
    simhash: int
    category: str
    count: int
    first_seen: float
    last_seen: float

    def to_dict(self) -> dict:
        return asdict(self)


_local_index: dict[int, ScamRecord] = {}
_loaded = False

# Don't fingerprint trivially-short messages — "مرحبا" alone produces a
# SimHash that matches every other 5-char Arabic greeting. Anything below
# this threshold is silently dropped on report and can't trigger a match.
MIN_TEMPLATE_CHARS = int(os.environ.get("COMMUNITY_MIN_TEMPLATE_CHARS", "20"))


def _firestore() -> "object | None":
    if not os.environ.get("FIRESTORE_PROJECT_ID"):
        return None
    try:
        from google.cloud import firestore  # type: ignore
        return firestore.Client(project=os.environ["FIRESTORE_PROJECT_ID"])
    except Exception as exc:  # noqa: BLE001
        log.debug("Firestore unavailable for community_db: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Postgres backend (preferred when DATABASE_URL is set)
# ---------------------------------------------------------------------------
_pg_pool = None
_pg_init_done = False


def _pg() -> "object | None":
    """Return a ConnectionPool, or None if Postgres isn't configured."""
    global _pg_pool, _pg_init_done
    if not DATABASE_URL:
        return None
    if _pg_pool is not None:
        return _pg_pool
    try:
        from psycopg2.pool import SimpleConnectionPool  # type: ignore
        _pg_pool = SimpleConnectionPool(1, 5, DATABASE_URL)
    except Exception as exc:  # noqa: BLE001
        log.warning("Postgres unavailable for community_db: %s", exc)
        return None
    if not _pg_init_done:
        try:
            _pg_init_schema()
            _pg_init_done = True
        except Exception as exc:  # noqa: BLE001
            log.error("community_db schema init failed: %s", exc)
            return None
    return _pg_pool


def _pg_init_schema() -> None:
    """Idempotent CREATE TABLE for the community_scams table."""
    pool = _pg_pool
    if pool is None:
        return
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS community_scams (
                    simhash       TEXT PRIMARY KEY,
                    category      TEXT NOT NULL DEFAULT 'scam',
                    count         INTEGER NOT NULL DEFAULT 0,
                    first_seen    DOUBLE PRECISION NOT NULL,
                    last_seen     DOUBLE PRECISION NOT NULL
                )
                """
            )
        log.info("community_scams Postgres schema ready.")
    finally:
        pool.putconn(conn)


def _pg_report(h: int, category: str) -> ScamRecord:
    pool = _pg()
    now = time.time()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO community_scams (simhash, category, count, first_seen, last_seen)
                VALUES (%s, %s, 1, %s, %s)
                ON CONFLICT (simhash) DO UPDATE
                    SET count = community_scams.count + 1,
                        last_seen = EXCLUDED.last_seen,
                        category = COALESCE(community_scams.category, EXCLUDED.category)
                RETURNING category, count, first_seen, last_seen
                """,
                (str(h), category, now, now),
            )
            row = cur.fetchone()
            return ScamRecord(
                simhash=h,
                category=row[0],
                count=int(row[1]),
                first_seen=float(row[2]),
                last_seen=float(row[3]),
            )
    finally:
        pool.putconn(conn)


def _pg_lookup(h: int) -> "MatchResult":
    """Postgres lookup: exact-hash hit OR fall back to in-process Hamming scan."""
    pool = _pg()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT category, count, first_seen, last_seen FROM community_scams WHERE simhash = %s",
                (str(h),),
            )
            row = cur.fetchone()
            if row is not None:
                rec = ScamRecord(
                    simhash=h,
                    category=row[0],
                    count=int(row[1]),
                    first_seen=float(row[2]),
                    last_seen=float(row[3]),
                )
                return MatchResult(matched=True, record=rec, distance=0)
            # Near-match: pull all hashes (cheap until we cross ~100k rows)
            # and scan in-process. We only return a HIT if Hamming ≤ threshold.
            cur.execute("SELECT simhash, category, count, first_seen, last_seen FROM community_scams")
            best: Optional[ScamRecord] = None
            best_d = 65
            for r in cur:
                try:
                    stored_h = int(r[0])
                except Exception:
                    continue
                d = hamming(stored_h, h)
                if d < best_d:
                    best_d = d
                    best = ScamRecord(
                        simhash=stored_h,
                        category=r[1],
                        count=int(r[2]),
                        first_seen=float(r[3]),
                        last_seen=float(r[4]),
                    )
                    if d == 0:
                        break
            if best is not None and best_d <= HAMMING_THRESHOLD:
                return MatchResult(matched=True, record=best, distance=best_d)
            return MatchResult(matched=False, record=best, distance=best_d)
    finally:
        pool.putconn(conn)


def _pg_remove(h: int) -> bool:
    """Best-effort cleanup of a near-match neighbourhood for a false_positive."""
    pool = _pg()
    conn = pool.getconn()
    removed = False
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT simhash FROM community_scams")
            rows = [r[0] for r in cur.fetchall()]
            to_delete: list[str] = []
            for s in rows:
                try:
                    if hamming(int(s), h) <= HAMMING_THRESHOLD:
                        to_delete.append(s)
                except Exception:
                    continue
            if to_delete:
                cur.execute(
                    "DELETE FROM community_scams WHERE simhash = ANY(%s)",
                    (to_delete,),
                )
                removed = True
        return removed
    finally:
        pool.putconn(conn)


def _pg_count() -> int:
    pool = _pg()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM community_scams")
            return int(cur.fetchone()[0])
    finally:
        pool.putconn(conn)


def _load_local() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not LOCAL_PATH.exists():
        return
    try:
        with LOCAL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rec = ScamRecord(**row)
                _local_index[rec.simhash] = rec
        log.info("Loaded %s community scam records from %s", len(_local_index), LOCAL_PATH)
    except Exception as exc:  # noqa: BLE001
        log.warning("community_db local load failed: %s", exc)


def _persist_local(rec: ScamRecord) -> None:
    try:
        with LOCAL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("community_db local write failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def report_scam(text: str, category: str = "scam") -> ScamRecord:
    """Record (or increment) a confirmed-scam template."""
    template = strip_personal_data(text)
    h = simhash(template)
    if h == 0 or len(template) < MIN_TEMPLATE_CHARS:
        log.info(
            "Skipping community report — template too short (%s chars).",
            len(template),
        )
        return ScamRecord(simhash=0, category=category, count=0, first_seen=0.0, last_seen=0.0)

    # Postgres-first when available (Railway-managed, persistent, atomic).
    if _pg() is not None:
        try:
            return _pg_report(h, category)
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres community report failed (%s) — falling back.", exc)

    _load_local()
    fc = _firestore()
    if fc is not None:
        try:
            from google.cloud.firestore_v1.transforms import Increment  # type: ignore
            doc = fc.collection("community_scams").document(str(h))
            now = time.time()
            doc.set(
                {
                    "simhash": h,
                    "category": category,
                    "count": Increment(1),
                    "last_seen": now,
                    "first_seen": now,  # set is merged so first_seen sticks if exists
                },
                merge=True,
            )
            snap = doc.get()
            return ScamRecord(
                simhash=h,
                category=str(snap.get("category") or category),
                count=int(snap.get("count") or 1),
                first_seen=float(snap.get("first_seen") or now),
                last_seen=float(snap.get("last_seen") or now),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Firestore community_db write failed: %s", exc)

    rec = _local_index.get(h)
    now = time.time()
    if rec is None:
        rec = ScamRecord(simhash=h, category=category, count=1, first_seen=now, last_seen=now)
    else:
        rec.count += 1
        rec.last_seen = now
    _local_index[h] = rec
    _persist_local(rec)
    return rec


@dataclass
class MatchResult:
    matched: bool
    record: Optional[ScamRecord]
    distance: int


def report_false_positive(text: str) -> bool:
    """
    Remove a previously-stored scam template (if any). Used when a user
    taps "خطأ — آمنة" on a result. Returns True if anything was removed.

    Removes the exact-hash AND any near-neighbor within HAMMING_THRESHOLD
    so the user's "this is fine" verdict applies to the entire fuzzy
    cluster, not just the precise wording they typed today.
    """
    template = strip_personal_data(text)
    h = simhash(template)
    if h == 0:
        return False

    if _pg() is not None:
        try:
            return _pg_remove(h)
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres false-positive removal failed (%s) — falling back.", exc)

    _load_local()
    removed = False

    fc = _firestore()
    if fc is not None:
        try:
            doc = fc.collection("community_scams").document(str(h))
            if doc.get().exists:
                doc.delete()
                removed = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Firestore delete failed: %s", exc)

    # Local index — drop anything within Hamming threshold.
    to_drop = [stored for stored in list(_local_index.keys()) if hamming(stored, h) <= HAMMING_THRESHOLD]
    for stored in to_drop:
        _local_index.pop(stored, None)
        removed = True

    if to_drop:
        # Rewrite the JSONL without the removed entries.
        try:
            with LOCAL_PATH.open("w", encoding="utf-8") as f:
                for rec in _local_index.values():
                    f.write(json.dumps(rec.to_dict()) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("community_db rewrite failed: %s", exc)

    return removed


def lookup(text: str) -> MatchResult:
    """Check whether the message looks like an already-reported scam."""
    template = strip_personal_data(text)
    h = simhash(template)
    if h == 0:
        return MatchResult(matched=False, record=None, distance=64)

    if _pg() is not None:
        try:
            return _pg_lookup(h)
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres community lookup failed (%s) — falling back.", exc)

    _load_local()
    # Firestore: do an exact-match lookup only (querying by Hamming
    # distance requires a geohash-style trick that's overkill for this
    # use case). Then fall back to in-memory near-match.
    fc = _firestore()
    if fc is not None:
        try:
            doc = fc.collection("community_scams").document(str(h)).get()
            if doc.exists:
                rec = ScamRecord(
                    simhash=h,
                    category=str(doc.get("category") or "scam"),
                    count=int(doc.get("count") or 1),
                    first_seen=float(doc.get("first_seen") or 0),
                    last_seen=float(doc.get("last_seen") or 0),
                )
                return MatchResult(matched=True, record=rec, distance=0)
        except Exception as exc:  # noqa: BLE001
            log.debug("Firestore community_db read failed: %s", exc)

    # In-memory near-match (Hamming ≤ HAMMING_THRESHOLD).
    if not _local_index:
        return MatchResult(matched=False, record=None, distance=64)
    best: Optional[ScamRecord] = None
    best_d = 65
    for stored, rec in _local_index.items():
        d = hamming(stored, h)
        if d < best_d:
            best_d = d
            best = rec
            if d == 0:
                break
    if best is not None and best_d <= HAMMING_THRESHOLD:
        return MatchResult(matched=True, record=best, distance=best_d)
    return MatchResult(matched=False, record=best, distance=best_d)


def stats() -> dict:
    """
    Surfaces the count from whatever backend is in use. The "local_count"
    name is preserved for client compatibility; in Postgres mode it's
    actually the count rows in the shared table.
    """
    if _pg() is not None:
        try:
            return {
                "local_count": _pg_count(),
                "backend": "postgres",
                "firestore_enabled": False,
                "hamming_threshold": HAMMING_THRESHOLD,
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres count failed (%s).", exc)
    _load_local()
    return {
        "local_count": len(_local_index),
        "backend": "firestore" if _firestore() is not None else "local",
        "firestore_enabled": _firestore() is not None,
        "hamming_threshold": HAMMING_THRESHOLD,
    }
