"""
Fine-tune AraBERT on the Shayekli training corpus, export ONNX, and
emit an INT8-quantized variant for on-device fallback (Section 2.2/2.4).

This is a heavyweight script — it pulls transformers + torch + onnx +
optimum (~3 GB of wheels) and benefits enormously from a GPU. Run it on
Google Colab (free T4) rather than on Railway.

Workflow:
    1. Run prepare_dataset.py first to produce datasets/training_corpus.csv.
    2. python train_arabert.py --data datasets/training_corpus.csv
    3. Outputs land in models/:
         - models/shayekli-arabert-v{N}/        (HF-format checkpoint)
         - models/shayekli-arabert-v{N}.onnx    (server-side ONNX, fp32)
         - models/shayekli-arabert-mobile-v{N}.onnx (INT8, mobile fallback)
    4. Bump MODEL_VERSION in Railway env, upload the .onnx as MODEL_DOWNLOAD_URL.

CLI:
    python train_arabert.py --data datasets/training_corpus.csv \\
        --base-model aubmindlab/bert-base-arabertv2 \\
        --version 2 --epochs 4 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("shayekli.train")

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to training_corpus.csv")
    p.add_argument("--base-model", default="aubmindlab/bert-base-arabertv2")
    p.add_argument("--version", type=int, default=2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-quantize", action="store_true")
    args = p.parse_args()

    # Heavy imports here so `--help` is fast and the import error message is
    # crystal-clear if the user forgot to install the training extras.
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
        import torch
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except Exception as exc:
        log.error("Training deps missing: %s", exc)
        log.error(
            "Install with:\n"
            "  pip install transformers[torch] datasets pandas scikit-learn onnx onnxruntime optimum"
        )
        sys.exit(1)

    torch.manual_seed(args.seed)

    # Local preprocess module so train + serve normalize identically.
    sys.path.insert(0, str(REPO_ROOT))
    from preprocess import preprocess_arabic  # type: ignore

    # ---------------------- Load + clean data ----------------------
    log.info("Loading %s", args.data)
    df = pd.read_csv(args.data)
    if "label" not in df.columns or "text" not in df.columns:
        raise SystemExit(f"CSV must have 'label' and 'text' columns; got {list(df.columns)}")
    df["text"] = df["text"].fillna("").astype(str).map(preprocess_arabic)
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    log.info("Corpus: %s rows  (scam=%s, legit=%s)", len(df), int((df.label == 1).sum()), int((df.label == 0).sum()))

    train_df, eval_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=args.seed)
    log.info("Split: train=%s eval=%s", len(train_df), len(eval_df))

    # ---------------------- Tokenize + model ----------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    def to_hf(d):
        ds = Dataset.from_pandas(d[["text", "label"]], preserve_index=False)
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length),
            batched=True,
        )

    train_ds = to_hf(train_df)
    eval_ds = to_hf(eval_df)

    out_dir = MODELS_DIR / f"shayekli-arabert-v{args.version}"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        report_to=[],
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary", pos_label=1),
        }

    collator = DataCollatorWithPadding(tokenizer)
    # Newer transformers removed `tokenizer=` from Trainer.__init__; the
    # collator already carries the tokenizer reference. Pass via the new
    # `processing_class=` kwarg when available, omit otherwise.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs)

    log.info("Training…")
    trainer.train()

    # Final eval + classification report.
    metrics = trainer.evaluate()
    log.info("Eval: %s", metrics)
    preds = trainer.predict(eval_ds).predictions.argmax(axis=-1)
    log.info("\n%s", classification_report(eval_df["label"].values, preds, target_names=["legit", "scam"]))

    # Save HF-format checkpoint.
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "shayekli_meta.json").write_text(
        json.dumps({"version": args.version, "base_model": args.base_model, **metrics}, indent=2),
        encoding="utf-8",
    )

    # ---------------------- ONNX export ----------------------
    onnx_path = MODELS_DIR / f"shayekli-arabert-v{args.version}.onnx"
    log.info("Exporting ONNX → %s", onnx_path)
    try:
        from optimum.exporters.onnx import main_export  # type: ignore

        main_export(
            model_name_or_path=str(out_dir),
            output=str(MODELS_DIR / f"_onnx-v{args.version}"),
            task="text-classification",
            opset=14,
        )
        # main_export writes model.onnx into the output dir; rename + tokenizer.
        src = MODELS_DIR / f"_onnx-v{args.version}" / "model.onnx"
        if src.exists():
            src.rename(onnx_path)
        else:
            log.warning("ONNX file not found at expected path: %s", src)
    except Exception as exc:
        log.error("ONNX export failed: %s", exc)
        return

    # ---------------------- INT8 quantization for mobile ----------------------
    if not args.no_quantize:
        mobile_path = MODELS_DIR / f"shayekli-arabert-mobile-v{args.version}.onnx"
        log.info("Quantizing INT8 → %s", mobile_path)
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore

            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(mobile_path),
                weight_type=QuantType.QInt8,
            )
            log.info("Mobile model: %.1f MB", mobile_path.stat().st_size / (1024 * 1024))
        except Exception as exc:
            log.warning("INT8 quantization failed: %s", exc)

    log.info("Done. Server ONNX: %s", onnx_path)
    log.info("Tokenizer dir:   %s", out_dir)
    log.info(
        "Next steps:\n"
        "  1. Upload %s to a public URL (HF Hub or S3).\n"
        "  2. On Railway, set MODEL_DOWNLOAD_URL=<that URL> and MODEL_VERSION=%s.\n"
        "  3. Redeploy. The mobile app will hot-swap on next launch.",
        onnx_path.name,
        args.version,
    )


if __name__ == "__main__":
    main()
