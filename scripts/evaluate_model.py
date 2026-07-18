"""Benchmark party classifiers on the speaker-independent held-out set.

Evaluates each model on the *same* validation set the speaker-independent
training run held out: 15 % of speakers are set aside entirely, so every
evaluated speech comes from a politician the (new) model never saw during
training.  A plain row-based split leaks speaker identity across the
train/test boundary and inflates macro-F1 dramatically (~59 % -> ~83 %).

The split here byte-for-byte replicates ``scripts/train_party_model_db.py``:
same year-by-year fetch, same tweet append, same party filter, same
``random.seed(42)`` shuffle of the speaker list, same 85/15 cut.  The
train/val speaker and row counts are logged so they can be cross-checked
against the training run's own printout.

Usage::

    python scripts/evaluate_model.py \
        --model MartinBlomqvist/maktsprak_classifier_clean \
        --model data/models/legacy/v1_2025-09_row-split \
        --model data/models/party_classifier \
        --limit 2000

Local model directories containing a raw ``best_model.pt`` state dict are
loaded onto the KB-BERT base with labels mapped from ``sorted(VALID_PARTIES)``
(matching training's ``LABELS = sorted(...)``); anything else is loaded as a
standard Hugging Face model via its own ``config.id2label``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.maktsprak_pipeline.config import TRAIN_BASE_MODEL, VALID_PARTIES
from src.maktsprak_pipeline.db import fetch_all_tweets
from src.maktsprak_pipeline.db.speeches import fetch_speeches_historical_v2
from src.maktsprak_pipeline.logger import get_logger

logger = get_logger()

# Must match label_party_from_account() in scripts/train_party_model_db.py,
# tweets are part of the speaker list, so they influence the split.
_PARTY_ACCOUNTS: dict[str, list[str]] = {
    "S": ["1587012835409788928"],
    "M": ["747426555417198592"],
    "V": ["282532238"],
    "L": ["455193032"],
    "KD": ["1407151866"],
    "C": ["232799403"],
    "MP": ["41214271", "370900852"],
    "SD": ["95972673"],
}
_ACCOUNT_TO_PARTY: dict[str, str] = {
    account: party for party, accounts in _PARTY_ACCOUNTS.items() for account in accounts
}


def _select_device() -> torch.device:
    """Return the best available torch device (cuda -> mps -> cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clean_text(text: object) -> str:
    """Replicate training's ``clean_text`` exactly."""
    return text.replace("\n", " ").strip() if isinstance(text, str) else ""


def build_test_set(
    limit: int | None, seed: int = 42, frozen_val_speakers: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """Recreate the speaker-independent validation set used during training.

    Fetches speeches year-by-year (2015-2026) with the speaker column,
    appends party-leader tweets, filters to the eight Riksdag parties.

    Args:
        limit: Max number of rows to evaluate (stratified downsample);
            ``None`` evaluates the full validation split.
        seed:  RNG seed for the shuffle fallback (see below).
        frozen_val_speakers: The exact held-out speaker set persisted at
            training time (``val_speakers.json`` next to the model). When
            given, rows are filtered directly against this set, no
            leakage risk regardless of how much the corpus has grown since
            training. When ``None``, falls back to re-deriving a split by
            shuffling the *current* corpus's speaker list, which does
            **not** reproduce the actual training-time partition once the
            database has changed (backfill, reindexing), the seed matches
            but the input doesn't, so this fallback should only be used for
            a model with no persisted speaker list (e.g. the legacy model).

    Returns:
        Tuple of ``(texts, parties)``.
    """
    # --- 1. Speeches, fetched exactly as training fetched them ---
    speech_dfs: list[pd.DataFrame] = []
    for year in range(2015, 2027):
        year_df = fetch_speeches_historical_v2(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
        logger.info(f"Fetched {len(year_df)} speeches for {year}.")
        speech_dfs.append(year_df)
    speeches_df = pd.concat(speech_dfs, ignore_index=True)
    speeches_df["text"] = speeches_df["text"].apply(_clean_text)
    speeches_df = speeches_df.rename(columns={"party": "label"})[["text", "label", "speaker"]]

    # --- 2. Tweets, appended exactly as training appended them ---
    tweets_df = fetch_all_tweets()
    tweets_df["text"] = tweets_df["text"].apply(_clean_text)
    tweets_df["label"] = tweets_df["username"].apply(lambda u: _ACCOUNT_TO_PARTY.get(u, "NA"))
    tweets_df = tweets_df.rename(columns={"username": "speaker"})[["text", "label", "speaker"]]

    # --- 3. Combine, filter, then select the held-out speakers ---
    df = pd.concat([speeches_df, tweets_df]).reset_index(drop=True)
    df = df[df["label"].isin(VALID_PARTIES)].reset_index(drop=True)
    df["speaker"] = df["speaker"].fillna("Okänd")

    if frozen_val_speakers is not None:
        val_speakers = frozen_val_speakers
        logger.info(f"Using frozen held-out speaker set: {len(val_speakers)} speakers.")
    else:
        logger.warning(
            "No frozen val_speakers.json given, re-deriving the split by shuffling the "
            "CURRENT corpus. This does not reproduce the actual training-time partition "
            "if the database has grown since training; treat results with caution."
        )
        unique_speakers = df["speaker"].unique().tolist()
        random.seed(seed)
        random.shuffle(unique_speakers)
        split_idx = int(len(unique_speakers) * 0.85)
        val_speakers = set(unique_speakers[split_idx:])

    val_df = df[df["speaker"].isin(val_speakers)].reset_index(drop=True)
    logger.info(
        f"{len(val_speakers)} held-out speakers -> "
        f"{len(df) - len(val_df)} train-side / {len(val_df)} val rows."
    )

    if limit is not None and limit < len(val_df):
        val_df, _ = train_test_split(
            val_df, train_size=limit, random_state=seed, stratify=val_df["label"]
        )
        logger.info(f"Downsampled to {len(val_df)} rows (stratified by party).")

    return val_df["text"].tolist(), val_df["label"].tolist()


def _load_model(
    model_path: str, device: torch.device
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load a model from a HF id/directory or a raw ``best_model.pt`` checkpoint.

    A directory containing ``best_model.pt`` (a bare ``state_dict`` saved by the
    training script) is loaded onto the KB-BERT base with the label mapping
    training used (``sorted(VALID_PARTIES)``).  Anything else goes through the
    standard ``from_pretrained`` path.
    """
    checkpoint = Path(model_path) / "best_model.pt"
    if checkpoint.is_file():
        labels = sorted(VALID_PARTIES)
        logger.info(f"  Loading raw checkpoint {checkpoint} onto {TRAIN_BASE_MODEL}.")
        tokenizer = AutoTokenizer.from_pretrained(TRAIN_BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            TRAIN_BASE_MODEL,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
        )
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def predict(
    model_path: str,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    """Run a model over *texts* and return predicted party labels.

    Args:
        model_path: HF hub id, HF model directory, or a directory containing a
            raw ``best_model.pt`` state dict.
        texts:      Input speeches.
        max_length: Truncation length.
        batch_size: Inference batch size.
        device:     Torch device.

    Returns:
        Predicted party strings (via the model's ``config.id2label``).
    """
    tokenizer, model = _load_model(model_path, device)
    id2label = model.config.id2label

    preds: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        logits = model(**enc).logits
        for idx in logits.argmax(dim=1).cpu().tolist():
            preds.append(id2label[idx])
    return preds


def evaluate(
    model_path: str,
    texts: list[str],
    truth: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Evaluate one model and persist its confusion matrix.

    Returns:
        Dict of headline metrics plus the per-class report.
    """
    logger.info(f"Evaluating: {model_path}")
    preds = predict(model_path, texts, max_length, batch_size, device)

    labels = sorted(VALID_PARTIES)
    acc = accuracy_score(truth, preds)
    f1 = f1_score(truth, preds, average="macro", labels=labels, zero_division=0)
    report = classification_report(truth, preds, labels=labels, zero_division=0, output_dict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_path.replace("/", "_").replace("\\", "_").strip("._")
    disp = ConfusionMatrixDisplay.from_predictions(
        truth,
        preds,
        labels=labels,
        normalize="true",
        xticks_rotation="vertical",
        values_format=".2f",
    )
    disp.ax_.set_title(f"{model_path}\nacc={acc:.3f}  macro-F1={f1:.3f}")
    plt.tight_layout()
    cm_path = out_dir / f"confusion_{safe_name}.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    logger.info(f"  accuracy={acc:.4f}  macro-F1={f1:.4f}  confusion matrix -> {cm_path}")

    return {"model": model_path, "accuracy": acc, "macro_f1": f1, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark party classifiers on the speaker-independent held-out set."
    )
    parser.add_argument(
        "--model", action="append", required=True, help="HF id or local path (repeatable)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max held-out rows (default 2000; 0 = full validation split).",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Token truncation length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument(
        "--val-speakers",
        type=Path,
        default=None,
        help="Path to a frozen val_speakers.json (written by train_party_model_db.py). "
        "Auto-discovered next to the first --model path if it's a local directory.",
    )
    args = parser.parse_args()

    device = _select_device()
    logger.info(f"Device: {device}")

    val_speakers_path = args.val_speakers
    if val_speakers_path is None:
        candidate = Path(args.model[0]) / "val_speakers.json"
        if candidate.is_file():
            val_speakers_path = candidate
            logger.info(f"Auto-discovered {candidate}")

    frozen_val_speakers = None
    if val_speakers_path is not None:
        frozen_val_speakers = set(json.loads(val_speakers_path.read_text(encoding="utf-8")))

    texts, truth = build_test_set(
        None if args.limit == 0 else args.limit, frozen_val_speakers=frozen_val_speakers
    )
    out_dir = Path(args.out)

    results = [
        evaluate(m, texts, truth, args.max_length, args.batch_size, device, out_dir)
        for m in args.model
    ]

    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n=== Benchmark summary (speaker-independent val set) ===")
    print(f"{'model':50} {'accuracy':>9} {'macro-F1':>9}")
    for r in results:
        print(f"{r['model']:50} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f}")
    # ASCII only, Windows consoles default to cp1252.
    for r in results[1:]:
        d_acc = r["accuracy"] - results[0]["accuracy"]
        d_f1 = r["macro_f1"] - results[0]["macro_f1"]
        print(
            f"\ndelta ({r['model']} vs {results[0]['model']}):  accuracy {d_acc:+.4f}   macro-F1 {d_f1:+.4f}"
        )
    print(f"\nWrote {out_dir / 'benchmark.json'} and confusion matrices.")
