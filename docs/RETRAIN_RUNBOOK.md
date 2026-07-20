# Retrain runbook

End-to-end procedure for retraining the party classifier and promoting it to
the live model. The classifier is deliberately scoped to 2015 onward; the
pre-2015 backfill exists for the drift analytics, not the model.

## Where the data comes from

The source of truth is `data/parquet/speeches_full.parquet` (75,148 speeches,
2002-2026), rebuilt offline from the cached PDFs in `data/raw/` with the fixed
parser:

```bash
python scripts/rebuild_corpus.py --jobs 12    # offline, deterministic, no DB
python scripts/verify_rebuild.py              # 21 checks against source + old archive
```

The Colab notebook pulls this Parquet from the private HF dataset
`MartinBlomqvist/maktsprak-corpus`, so a training run needs only the repo plus an
`HF_TOKEN`. No Supabase credentials are required to train.

## Phase 1, retrain on Colab (GPU, ~$0 in compute credits, ~2-4 h on an L4)

Open `notebooks/retrain_colab.ipynb` on a GPU runtime and run the cells top to
bottom. They clone the repo, pull the corpus, and run:

```bash
python scripts/train_party_model_db.py \
    --parquet data/parquet/speeches_full.parquet \
    --year-min 2015 \
    --output-dir /content/drive/MyDrive/MaktsprakAI_Checkpoints
```

The split is speaker-independent: 15% of speakers are held out entirely, and the
exact held-out set is persisted to `val_speakers.json` next to the model. That
file is what makes evaluation reproducible; nothing re-derives the split.
Checkpoints land on Drive, so a disconnect just means re-running the cell.

### Cost / speed levers

| Flag | Effect |
| --- | --- |
| `--no-fgm` | skip FGM adversarial training (~2x faster, cheap baseline) |
| `--max-length 256` | ~2x faster, small accuracy cost |
| `--epochs N` | cap epochs (early stopping also applies) |
| `--batch-size N` | rescales the OneCycleLR `max_lr` automatically |

An L4 is plenty for a 110M-parameter BERT; an A100 is not needed.

## Phase 2, evaluate honestly

Score the checkpoint on its own held-out set, loading the persisted split so
there is no leakage:

```bash
python scripts/evaluate_model.py \
    --model /content/drive/MyDrive/MaktsprakAI_Checkpoints \
    --val-speakers /content/drive/MyDrive/MaktsprakAI_Checkpoints/val_speakers.json \
    --parquet data/parquet/speeches_full.parquet \
    --year-min 2015 \
    --limit 0
```

A clean split shows a large gap between held-out and training-speaker accuracy;
if they are close, the split is not holding and the number is not honest. Sanity
check the model on short distinctive input too (for example
`"Stoppa invandringen!"` should read SD, not the smallest class).

## Phase 3, promote

The notebook pushes the new model to a separate repo,
`MartinBlomqvist/maktsprak_classifier_reindexed`, so the live app keeps running
the old one during A/B. Promote to the live repo only once the new model is
verified locally (accuracy and the short-input behaviour):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
AutoModelForSequenceClassification.from_pretrained(_dir).push_to_hub(
    "MartinBlomqvist/maktsprak_classifier_clean")
AutoTokenizer.from_pretrained(_dir).push_to_hub(
    "MartinBlomqvist/maktsprak_classifier_clean")
```

The Cloud Run inference service bakes the model into its image, so it picks the
new weights up on its next build / redeploy. Update the stated metrics on the
site (`web/src/app/page.tsx`, `web/src/app/metod/page.tsx`) if the number moved.
