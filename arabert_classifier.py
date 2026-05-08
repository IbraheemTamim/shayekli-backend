"""
AraBERT ONNX classifier (Phase 2b runtime).

Loads the fine-tuned AraBERT-v2 INT8 ONNX model + its tokenizer from
local disk, downloading both on first boot if URLs are configured via
env. Exposes a scikit-learn-compatible interface (`.predict()` and
`.predict_proba()`) so it slots into the existing detection pipeline
in main.py without any handler changes.

Dependencies are deliberately lightweight: `onnxruntime` for inference,
`tokenizers` (HF's pure-Rust tokenizer, no transformers/torch) for
pre-processing. About 20 MB of wheels total.

Env contract (read by `prepare_files`):
    MODEL_DOWNLOAD_URL      → URL to the .onnx file (~110 MB INT8).
    MODEL_TOKENIZER_URL     → URL to tokenizer.json.
    MODEL_LOCAL_DIR         → Local cache directory (default: ./models).
    MODEL_MAX_LENGTH        → Tokenizer truncation length (default: 128).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

log = logging.getLogger("shayekli.arabert")


# ---------------------------------------------------------------------------
# File acquisition (download once, cache to disk)
# ---------------------------------------------------------------------------
def _download_if_needed(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Model file already cached at %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)

    import httpx  # already a runtime dep

    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_in = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
                bytes_in += len(chunk)
    tmp.rename(dest)
    log.info("Downloaded %.1f MB -> %s", bytes_in / 1e6, dest)
    return dest


def prepare_files(onnx_url: str, tokenizer_url: str, local_dir: str | None = None) -> tuple[Path, Path]:
    """Ensure the ONNX model + tokenizer are on disk; download on first boot."""
    if not onnx_url or not tokenizer_url:
        raise RuntimeError("Both MODEL_DOWNLOAD_URL and MODEL_TOKENIZER_URL must be set.")
    base = Path(local_dir or os.environ.get("MODEL_LOCAL_DIR", "models")).resolve()
    onnx_path = base / "shayekli-arabert-mobile.onnx"
    tok_path = base / "tokenizer.json"
    _download_if_needed(onnx_url, onnx_path)
    _download_if_needed(tokenizer_url, tok_path)
    return onnx_path, tok_path


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class ArabertClassifier:
    """
    sklearn-compatible wrapper over an ONNX-exported text classifier.

    Methods:
        .predict(texts: Iterable[str]) -> np.ndarray of shape (N,) with class indices.
        .predict_proba(texts: Iterable[str]) -> np.ndarray of shape (N, 2).
            Column 0 = probability of legit, column 1 = probability of scam.
    """

    def __init__(self, onnx_path: Path | str, tokenizer_path: Path | str, max_length: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
            sess_options=self._build_session_options(),
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self._max_length = int(max_length or os.environ.get("MODEL_MAX_LENGTH", "128"))
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(length=self._max_length)

        self._input_names = {i.name for i in self._session.get_inputs()}
        self._uses_token_type_ids = "token_type_ids" in self._input_names
        log.info(
            "ArabertClassifier ready (inputs=%s, max_len=%s)",
            sorted(self._input_names),
            self._max_length,
        )

    @staticmethod
    def _build_session_options():
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def _encode(self, texts: List[str]):
        encs = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._uses_token_type_ids:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        # Drop any keys the model doesn't actually consume.
        return {k: v for k, v in feeds.items() if k in self._input_names}

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [str(t) if t is not None else "" for t in texts]
        if not text_list:
            return np.zeros((0, 2), dtype=np.float32)
        feeds = self._encode(text_list)
        outputs = self._session.run(None, feeds)
        logits = outputs[0]
        # Numerically-stable softmax.
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs.astype(np.float32)

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=-1)
