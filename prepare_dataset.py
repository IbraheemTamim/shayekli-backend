"""
Build the Shayekli training corpus (Section 2.1 of the upgrade spec).

Sources combined:
  1. ealvaradob/phishing-dataset on HuggingFace (5,971 SMS, 638 verified
     smishing). English; we translate the smishing samples to Palestinian
     Arabic via Claude.
  2. The legacy CSVs already shipping in the repo (palestinian_sms_dataset,
     large_hybrid_sms_dataset, arabic_hybrid_sms_dataset, expand_dataset).
  3. datasets/palestinian_templates.csv — manually curated local-dialect
     templates covering banks, telecoms, government, prizes, deliveries.

Output: datasets/training_corpus.csv (label, text, source, language, category)

This script is meant to be run BEFORE train_arabert.py — typically in a
Colab notebook, not on Railway. Claude usage costs roughly $0.10 / 1k
samples translated; pass --skip-translate to bypass it.

Usage:
    python prepare_dataset.py --out datasets/training_corpus.csv
    python prepare_dataset.py --skip-translate          # offline mode
    python prepare_dataset.py --max-translate 500       # cost cap
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("shayekli.prepare")

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

DEFAULT_OUT = DATASETS_DIR / "training_corpus.csv"

# Pre-existing CSVs that came with the repo. Each may use a different
# column layout — handle both ("label,text") and ("text,label") forms.
LEGACY_CSVS = [
    REPO_ROOT.parent / "palestinian_sms_dataset.csv",
    REPO_ROOT.parent / "arabic_hybrid_sms_dataset.csv",
    REPO_ROOT.parent / "large_hybrid_sms_dataset.csv",
]

PALESTINIAN_TEMPLATES = DATASETS_DIR / "palestinian_templates.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_legacy_csv(path: Path) -> list[tuple[str, int, str]]:
    if not path.exists():
        log.warning("Legacy CSV missing: %s", path)
        return []
    rows: list[tuple[str, int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalise column names.
        fieldnames = {fn.lower(): fn for fn in (reader.fieldnames or [])}
        text_col = fieldnames.get("text") or fieldnames.get("message") or fieldnames.get("body")
        label_col = fieldnames.get("label") or fieldnames.get("is_scam") or fieldnames.get("class")
        if not text_col or not label_col:
            log.warning("Couldn't infer columns for %s — fields: %s", path, fieldnames)
            return []
        for row in reader:
            text = (row[text_col] or "").strip()
            raw_label = (row[label_col] or "").strip().lower()
            if not text:
                continue
            label = 1 if raw_label in ("1", "scam", "spam", "phish", "smishing", "true") else 0
            rows.append((text, label, f"legacy:{path.name}"))
    log.info("Loaded %s rows from %s", len(rows), path.name)
    return rows


def _load_palestinian_templates() -> list[tuple[str, int, str]]:
    if not PALESTINIAN_TEMPLATES.exists():
        log.warning("Palestinian templates CSV missing: %s", PALESTINIAN_TEMPLATES)
        return []
    rows: list[tuple[str, int, str]] = []
    with PALESTINIAN_TEMPLATES.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            try:
                label = int(row.get("label") or 0)
            except Exception:
                label = 0
            if text:
                rows.append((text, label, f"palestinian:{row.get('category', 'general')}"))
    log.info("Loaded %s Palestinian-dialect templates.", len(rows))
    return rows


def _load_huggingface_smishing(max_rows: int | None = None) -> list[tuple[str, int, str]]:
    """
    Pull a public phishing-text dataset from HuggingFace.

    Tries multiple known-good repos / configs; logs failures at INFO so
    silent corpus shrinkage is impossible. Falls through on failure to
    the next candidate.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("`datasets` lib unavailable (%s) — skipping HuggingFace fetch.", exc)
        return []

    rows: list[tuple[str, int, str]] = []
    # Each entry: (repo, config_or_None, text_col_priority, label_col_priority)
    candidates = [
        ("ealvaradob/phishing-dataset", "combined_full"),
        ("ealvaradob/phishing-dataset", "combined_reduced"),
        ("ealvaradob/phishing-dataset", "texts"),
        ("Hellisotherpeople/sms-spam", None),
        ("ucirvine/sms_spam", None),
    ]
    for repo, config in candidates:
        cfg_label = f"/{config}" if config else ""
        try:
            if config:
                ds = load_dataset(repo, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(repo, split="train", trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("HF load failed for %s%s: %s", repo, cfg_label, exc)
            continue
        log.info("Loaded HF %s%s — %s rows", repo, cfg_label, len(ds))

        sample = ds[0] if len(ds) > 0 else {}
        text_col = next((c for c in ("text", "body", "message", "Message", "sms", "email") if c in sample), None)
        label_col = next((c for c in ("label", "Category", "class", "is_spam") if c in sample), None)
        if not text_col or not label_col:
            log.warning("  Couldn't infer text/label columns. Keys: %s", list(sample.keys()))
            continue

        kept = 0
        for i, ex in enumerate(ds):
            if max_rows and kept >= max_rows:
                break
            text = str(ex.get(text_col) or "").strip()
            raw_label = ex.get(label_col)
            if isinstance(raw_label, str):
                label = 1 if raw_label.lower() in ("spam", "phish", "phishing", "smishing", "1", "true") else 0
            else:
                try:
                    label = 1 if int(raw_label) == 1 else 0
                except Exception:
                    label = 0
            # Drop overlong emails — they kill SimHash + tokenizer budget.
            if not text or len(text) > 4000:
                continue
            rows.append((text, label, f"hf:{repo}{cfg_label}"))
            kept += 1
        log.info("  Kept %s usable rows from %s", kept, repo)
        if kept > 0:
            break  # first non-empty source is enough at our scale
    return rows


# ---------------------------------------------------------------------------
# Claude-based Arabic dialect translation
# ---------------------------------------------------------------------------
TRANSLATE_SYSTEM = (
    "You are an Arabic localization expert specializing in Palestinian dialect. "
    "Translate the given English SMS scam message into a NATURAL Palestinian Arabic "
    "scam message that a real attacker would actually send. Preserve the malicious "
    "intent and call-to-action; substitute local Palestinian context (Bank of "
    "Palestine, Jawwal, PalTel, Ministry of Health, Aramex Palestine, etc.) where "
    "the original references foreign brands. Keep URLs but you may swap the domain "
    "to a plausible-sounding scam domain. Output ONLY the Arabic message text — "
    "no explanation, no quotes, no JSON wrapping."
)


def _translate_to_palestinian_ar(messages: list[str]) -> list[str]:
    api_key = os.environ.get("CLAUDE_API_KEY", "").strip()
    if not api_key:
        log.warning("CLAUDE_API_KEY not set — skipping translation.")
        return []
    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:
        log.warning("anthropic SDK unavailable (%s) — skipping translation.", exc)
        return []
    client = Anthropic(api_key=api_key)
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    out: list[str] = []
    for i, en in enumerate(messages):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                system=TRANSLATE_SYSTEM,
                messages=[{"role": "user", "content": en}],
            )
            text = "".join(
                getattr(block, "text", "")
                for block in resp.content
                if getattr(block, "type", "") == "text"
            ).strip()
            if text:
                out.append(text)
            if (i + 1) % 25 == 0:
                log.info("  translated %s/%s", i + 1, len(messages))
            # Respect rate limits — cheap insurance.
            time.sleep(0.4)
        except Exception as exc:  # noqa: BLE001
            log.warning("translate failed at %s: %s", i, exc)
            time.sleep(1.5)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--skip-translate", action="store_true", help="Skip Claude augmentation.")
    p.add_argument("--max-translate", type=int, default=300, help="Cap on Claude translations.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    rows: list[tuple[str, int, str]] = []
    rows += _load_palestinian_templates()
    for path in LEGACY_CSVS:
        rows += _load_legacy_csv(path)
    rows += _load_huggingface_smishing(max_rows=6000)

    # Deduplicate while preserving label/source.
    seen: dict[str, tuple[int, str]] = {}
    for text, label, source in rows:
        key = text.strip().lower()
        if key and key not in seen:
            seen[key] = (label, source)
    base_rows = [(t, lab, src) for t, (lab, src) in zip(seen.keys(), seen.values())]
    log.info("After dedup: %s unique rows.", len(base_rows))

    augmented: list[tuple[str, int, str]] = []
    if not args.skip_translate:
        # Pick verified-scam English samples to translate into Palestinian Arabic.
        candidates = [t for t, lab, src in base_rows if lab == 1 and src.startswith("hf:")]
        random.shuffle(candidates)
        candidates = candidates[: max(0, args.max_translate)]
        if candidates:
            log.info("Translating %s scam samples to Palestinian Arabic via Claude…", len(candidates))
            arabic = _translate_to_palestinian_ar(candidates)
            augmented = [(t, 1, "claude_translated") for t in arabic]
            log.info("Got %s Arabic-translated scam samples.", len(augmented))

    final = base_rows + augmented
    random.shuffle(final)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "text", "source"])
        for text, label, source in final:
            w.writerow([label, text, source])

    scam = sum(1 for _, lab, _ in final if lab == 1)
    legit = len(final) - scam
    log.info("Wrote %s rows to %s (scam=%s, legit=%s).", len(final), out_path, scam, legit)


if __name__ == "__main__":
    main()
