# Shayekli — Retraining Pipeline

The fine-tuning pipeline ships as standalone scripts so you can run it on
Google Colab (free T4 GPU) without bloating the Railway runtime image.

## Files

| File | Purpose |
|---|---|
| `datasets/palestinian_templates.csv` | Hand-curated Palestinian-dialect scam + legitimate templates |
| `prepare_dataset.py` | Pulls HuggingFace datasets, augments with Claude-translated Palestinian Arabic, writes `datasets/training_corpus.csv` |
| `train_arabert.py` | Fine-tunes AraBERT, exports ONNX (server fp32) + INT8 (mobile) |
| `preprocess.py` | Shared Arabic normalization — used by both training and inference |

## Step 1 — Prepare the corpus

Locally or on Colab:

```bash
pip install datasets anthropic httpx
export CLAUDE_API_KEY=sk-ant-...
python prepare_dataset.py --max-translate 300
```

Skip Claude augmentation (offline / cost-free):

```bash
python prepare_dataset.py --skip-translate
```

Output: `datasets/training_corpus.csv` with columns `label, text, source`.

## Step 2 — Fine-tune AraBERT (Colab recommended)

1. Open https://colab.research.google.com → New Notebook → Runtime → Change runtime type → GPU (T4 free).
2. Upload `prepare_dataset.py`, `train_arabert.py`, `preprocess.py`, `datasets/`.
3. In a Colab cell:

```python
!pip install -q transformers[torch] datasets pandas scikit-learn onnx onnxruntime optimum

# Optional — only if you want to re-translate from English smishing
%env CLAUDE_API_KEY=sk-ant-...

!python prepare_dataset.py --max-translate 300
!python train_arabert.py --data datasets/training_corpus.csv --version 2 --epochs 4
```

Fine-tuning AraBERT-base on ~10k SMS samples takes ~12 minutes on T4.

## Step 3 — Inspect outputs

```
models/
├── shayekli-arabert-v2/                    HF checkpoint (transformers Trainer.save_model output)
├── shayekli-arabert-v2.onnx                 server-side ONNX, fp32 (~430 MB)
└── shayekli-arabert-mobile-v2.onnx          INT8 quantized, mobile-ready (~110 MB)
```

Check `models/shayekli-arabert-v2/shayekli_meta.json` for accuracy / F1.

## Step 4 — Ship the model

Upload the ONNX files to a public URL (HuggingFace Hub or any object storage):

```bash
# HuggingFace Hub example
huggingface-cli login
huggingface-cli upload <your-username>/shayekli-arabert-v2 \
    models/shayekli-arabert-v2.onnx model.onnx
```

On Railway → service → Variables, set:

```
MODEL_VERSION=2
MODEL_DOWNLOAD_URL=https://huggingface.co/<your-username>/shayekli-arabert-v2/resolve/main/model.onnx
MODEL_CHANGELOG_AR=نسخة محسّنة مدرّبة على بيانات حقيقية للهجة الفلسطينية.
```

Redeploy. The mobile app polls `/model/version` on launch and pulls the
new model automatically (Phase 2c — frontend OTA loader).

## Cost notes

- **HuggingFace datasets**: free.
- **Colab GPU**: free for occasional use (~12 hours/week).
- **Claude translation**: ~$0.10 per 1k samples. With `--max-translate 300` ≈ $0.03.
- **Railway**: no extra cost — the trained ONNX is served by the existing service.

## Skipping retraining (until Phase 2c lands)

The Phase 2a runtime pipeline (preprocessing + URL reputation + Claude
fallback) already substantially improves detection quality even with the
existing scikit-learn model. You can defer retraining until you have a
weekend and Colab credits to spare.
