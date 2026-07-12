"""Train the KB-BERT party classifier directly from the Supabase database.

Fine-tunes ``KB/bert-base-swedish-cased`` on Riksdag speeches (plus a small
set of party-leader tweets) to predict the speaker's party.  Training uses a
**speaker-independent split**: 15 % of unique speakers are held out entirely,
so validation measures generalisation to politicians the model has never seen.

Techniques used:

- AMP (mixed precision) on GPU
- OneCycleLR schedule with batch-scaled max_lr
- Weighted sampler + class-weighted loss for party imbalance
- Label smoothing, gradient clipping, weight decay
- FGM adversarial training on embeddings (disable with ``--no-fgm``)
- Encoder frozen for the first two epochs, then unfrozen
- Per-epoch checkpoints with resume, early stopping on macro-F1

Usage::

    # Full run (writes to config's TRAIN_MODEL_DIR):
    python scripts/train_party_model_db.py

    # Cheap baseline pass on Colab, saving checkpoints to Drive:
    python scripts/train_party_model_db.py --no-fgm --max-length 256 \
        --output-dir /content/drive/MyDrive/MaktsprakAI_Checkpoints

The exported model directory contains both ``best_model.pt`` (raw state dict,
best validation epoch) and a standard Hugging Face export of those same
weights, ready for ``push_to_hub``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Must be set before transformers is imported.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.maktsprak_pipeline.config import (
    TRAIN_BASE_MODEL,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOPPING_PATIENCE,
    TRAIN_LABEL_SMOOTHING,
    TRAIN_LEARNING_RATE,
    TRAIN_MAX_EPOCHS,
    TRAIN_MAX_LENGTH,
    TRAIN_MAX_LR,
    TRAIN_MODEL_DIR,
    TRAIN_WEIGHT_DECAY,
    VALID_PARTIES,
)
from src.maktsprak_pipeline.db import fetch_all_tweets
from src.maktsprak_pipeline.db.speeches import fetch_speeches_historical_v2
from src.maktsprak_pipeline.logger import get_logger

logger = get_logger()

GRAD_CLIP_NORM: float = 1.0
CLASSIFIER_DROPOUT: float = 0.2
SPLIT_SEED: int = 42
TRAIN_SPEAKER_FRACTION: float = 0.85
FREEZE_ENCODER_EPOCHS: int = 2

# Party-leader Twitter/X accounts, keyed by party.  Kept in sync with
# build_test_set() in scripts/evaluate_model.py — tweets are part of the
# speaker list, so they influence the speaker split.
PARTY_ACCOUNTS: dict[str, list[str]] = {
    "S": ["1587012835409788928"],
    "M": ["747426555417198592"],
    "V": ["282532238"],
    "L": ["455193032"],
    "KD": ["1407151866"],
    "C": ["232799403"],
    "MP": ["41214271", "370900852"],
    "SD": ["95972673"],
}
ACCOUNT_TO_PARTY: dict[str, str] = {
    account: party for party, accounts in PARTY_ACCOUNTS.items() for account in accounts
}


def clean_text(text: object) -> str:
    """Normalise a raw text value to a single-line stripped string."""
    return text.replace("\n", " ").strip() if isinstance(text, str) else ""


def get_training_data() -> pd.DataFrame:
    """Fetch and combine speeches and tweets into one labelled DataFrame.

    Speeches are fetched year by year (2015-2026) to keep individual Supabase
    queries small.  Party-leader tweets are appended with the account id as
    ``speaker``.  Rows outside the eight Riksdag parties are dropped.

    Note:
        The fetch order is load-bearing: ``scripts/evaluate_model.py``
        recreates the speaker split by replicating this exact pipeline, so the
        year loop, column selection, and concat order must not change without
        updating the evaluation script in lockstep.

    Returns:
        DataFrame with columns ``text``, ``label`` (party), ``speaker``.
    """
    speech_dfs: list[pd.DataFrame] = []
    for year in range(2015, 2027):
        year_df = fetch_speeches_historical_v2(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
        logger.info(f"Fetched {len(year_df)} speeches for {year}.")
        speech_dfs.append(year_df)
    speeches_df = pd.concat(speech_dfs, ignore_index=True)
    speeches_df["text"] = speeches_df["text"].apply(clean_text)
    speeches_df = speeches_df.rename(columns={"party": "label"})[["text", "label", "speaker"]]

    tweets_df = fetch_all_tweets()
    tweets_df["text"] = tweets_df["text"].apply(clean_text)
    tweets_df["label"] = tweets_df["username"].apply(lambda u: ACCOUNT_TO_PARTY.get(u, "NA"))
    tweets_df = tweets_df.rename(columns={"username": "speaker"})[["text", "label", "speaker"]]

    df = pd.concat([speeches_df, tweets_df]).reset_index(drop=True)
    return df[df["label"].isin(VALID_PARTIES)].reset_index(drop=True)


def speaker_split(
    df: pd.DataFrame,
    seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_SPEAKER_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so that no speaker appears in both train and validation.

    The speaker list is deliberately **not** sorted before shuffling: the
    evaluation script reproduces this exact partition by shuffling the raw
    ``unique()`` order with the same seed.  Change one, change both.

    Args:
        df:             Combined dataset with a ``speaker`` column.
        seed:           RNG seed shared with the evaluation script.
        train_fraction: Fraction of unique speakers assigned to training.

    Returns:
        Tuple of ``(train_df, val_df)``.
    """
    df["speaker"] = df["speaker"].fillna("Okänd")

    unique_speakers = df["speaker"].unique().tolist()
    random.seed(seed)
    random.shuffle(unique_speakers)

    split_idx = int(len(unique_speakers) * train_fraction)
    train_speakers = set(unique_speakers[:split_idx])
    val_speakers = set(unique_speakers[split_idx:])

    train_df = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
    val_df = df[df["speaker"].isin(val_speakers)].reset_index(drop=True)

    logger.info(
        f"Speaker split: {len(train_speakers)} train / {len(val_speakers)} val speakers "
        f"-> {len(train_df)} train / {len(val_df)} val rows."
    )
    return train_df, val_df


class PartyDataset(Dataset):
    """Tokenised (text, label) pairs for party classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class FGM:
    """Fast Gradient Method: adversarial perturbation of embedding weights.

    After the normal backward pass, ``attack()`` nudges the embedding weights
    along the gradient direction; a second forward/backward on the perturbed
    model adds an adversarial loss term, then ``restore()`` puts the original
    weights back.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1.0) -> None:
        self.model = model
        self.epsilon = epsilon
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embedding" in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def validate(
    model: nn.Module, val_loader: DataLoader, device: torch.device, label_names: list[str]
) -> dict[str, float]:
    """Run validation and return headline metrics (also logs a per-class report)."""
    model.eval()
    preds_all: list[int] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels_all.extend(batch["labels"].cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(labels_all, preds_all),
        "f1_macro": f1_score(labels_all, preds_all, average="macro"),
        "precision_macro": precision_score(labels_all, preds_all, average="macro", zero_division=0),
        "recall_macro": recall_score(labels_all, preds_all, average="macro", zero_division=0),
    }
    logger.info(
        "Validation: " + " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )
    logger.info(
        "\n"
        + classification_report(labels_all, preds_all, target_names=label_names, zero_division=0)
    )
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the party classifier from the DB.")
    parser.add_argument(
        "--no-fgm",
        action="store_true",
        help="Disable FGM adversarial training (~2x faster; use for a cheap baseline).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=TRAIN_MAX_LENGTH,
        help=f"Token length (default {TRAIN_MAX_LENGTH}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_MAX_EPOCHS,
        help=f"Max epochs (default {TRAIN_MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help=f"Batch size (default {TRAIN_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(TRAIN_MODEL_DIR),
        help=f"Where to write checkpoints and the final model (default {TRAIN_MODEL_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    """Train, validate, and export the party classifier."""
    args = parse_args()
    use_fgm: bool = not args.no_fgm
    output_dir: Path = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_lr = TRAIN_MAX_LR * (args.batch_size / TRAIN_BATCH_SIZE)
    logger.info(
        f"Device: {device} | FGM={use_fgm} | max_length={args.max_length} | "
        f"epochs={args.epochs} | batch_size={args.batch_size} | max_lr={max_lr:.2e} | "
        f"output={output_dir}"
    )

    # ------------------------------------------------------------------ data
    df = get_training_data()

    counts = df["label"].value_counts().to_dict()
    for party in sorted(VALID_PARTIES):
        logger.info(f"  {party}: {counts.get(party, 0)} samples")
        if counts.get(party, 0) < 10:
            logger.warning(f"  {party} has very few samples ({counts.get(party, 0)}).")
    logger.info(f"Total samples: {len(df)}")

    label_names = sorted(df["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    train_df, val_df = speaker_split(df)
    train_texts, train_labels = train_df["text"].tolist(), train_df["label_id"].tolist()
    val_texts, val_labels = val_df["text"].tolist(), val_df["label_id"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(TRAIN_BASE_MODEL)
    train_dataset = PartyDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = PartyDataset(val_texts, val_labels, tokenizer, args.max_length)

    # Oversample rare parties so every batch stays roughly balanced.
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    sample_weights = [1.0 / label_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ----------------------------------------------------------------- model
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAIN_BASE_MODEL,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=CLASSIFIER_DROPOUT,
    ).to(device)

    # Weight decay on everything except bias/LayerNorm.
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer = AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": TRAIN_WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=TRAIN_LEARNING_RATE,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device),
        label_smoothing=TRAIN_LABEL_SMOOTHING,
    )
    fgm = FGM(model)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------------------------------------------------------------- resume
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1])
    )
    start_epoch = 0
    if checkpoints:
        latest = checkpoints[-1]
        logger.info(f"Resuming from checkpoint {latest}.")
        model.load_state_dict(torch.load(latest, map_location=device))
        start_epoch = int(latest.stem.split("_")[-1])
    # Note: optimizer/scheduler state is not checkpointed, so a resumed run
    # restarts the LR schedule.  Acceptable for occasional Colab disconnects.

    # ----------------------------------------------------------------- train
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_path = output_dir / "best_model.pt"

    for epoch in range(start_epoch, args.epochs):
        # Freeze the encoder for the first epochs so the fresh classifier head
        # settles before full fine-tuning ('>=' keeps this correct on resume).
        unfreeze = epoch >= FREEZE_ENCODER_EPOCHS
        for name, param in model.named_parameters():
            param.requires_grad = unfreeze or "encoder" not in name

        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != "labels"}

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(**inputs).logits, batch["labels"])
            scaler.scale(loss).backward()

            if use_fgm:
                fgm.attack()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss_adv = criterion(model(**inputs).logits, batch["labels"])
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=total_loss / (progress.n + 1))

        logger.info(f"Epoch {epoch + 1} finished. Avg loss: {total_loss / len(train_loader):.4f}")

        metrics = validate(model, val_loader, device, label_names)

        if metrics["f1_macro"] > best_val_f1:
            best_val_f1 = metrics["f1_macro"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved (macro-F1 {best_val_f1:.4f}) -> {best_path}")
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement for {epochs_no_improve} epoch(s). Best macro-F1: {best_val_f1:.4f}"
            )
            if epochs_no_improve >= TRAIN_EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch + 1}.pt")

    # ---------------------------------------------------------------- export
    # Reload the best epoch before exporting: after early stopping the live
    # model holds the *last* (worse) weights, not the best ones.
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Best model (macro-F1 {best_val_f1:.4f}) exported to {output_dir}")


if __name__ == "__main__":
    main()
