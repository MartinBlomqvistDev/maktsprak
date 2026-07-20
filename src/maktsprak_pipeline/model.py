"""Inference module, load the fine-tuned party classifier and run predictions.

The model is a KB-BERT (KB/bert-base-swedish-cased) sequence classifier
fine-tuned on Riksdag speeches and tweets.  It is hosted on Hugging Face Hub
at :data:`~config.MODEL_NAME_OR_PATH`.
"""

from __future__ import annotations

import torch
from torch.nn.functional import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .config import MODEL_NAME_OR_PATH
from .logger import get_logger

logger = get_logger()


def load_model_and_tokenizer(
    device: str | None = None,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Download (or load from cache) the fine-tuned classifier from Hugging Face Hub.

    Args:
        device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
                Auto-detected from CUDA availability when ``None``.

    Returns:
        A ``(model, tokenizer)`` tuple ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model '{MODEL_NAME_OR_PATH}' on device '{device}'.")
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, config=config)
    model.to(device)
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


def predict_party(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
) -> list[dict[str, float]]:
    """Return per-party softmax probabilities for each input text.

    Args:
        model:     Fine-tuned classifier in eval mode.
        tokenizer: Matching tokenizer.
        texts:     List of raw speech strings to classify.

    Returns:
        One dict per input text mapping party abbreviation → probability (0-1).
        Party order matches :data:`~config.PARTY_ORDER`.
    """
    device = next(model.parameters()).device
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    results: list[dict[str, float]] = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs: list[float] = softmax(logits, dim=1).squeeze().cpu().tolist()

        results.append({id2label[i]: p for i, p in enumerate(probs)})

    return results
