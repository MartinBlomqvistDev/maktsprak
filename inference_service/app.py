"""FastAPI inference service for the party classifier.

Deployed to Cloud Run, deliberately with no dependency on the
``src.maktsprak_pipeline`` package: importing that package's ``__init__.py``
pulls in ``pipeline.orchestrate`` -> ``db.client``, which requires Supabase
credentials this container has no reason to hold.

``MODEL_NAME_OR_PATH`` points at ``/opt/model``, baked into the image at build
time (see the Dockerfile).  It used to be a Hub id resolved at startup, which
meant every cold start depended on Hugging Face answering, until HF returned
429 and the service simply stopped starting.  The weights ship with the image
now; nothing in the request path touches the network.
"""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME_OR_PATH = os.environ.get(
    "MODEL_NAME_OR_PATH", "MartinBlomqvist/maktsprak_classifier_clean"
)
MAX_LENGTH = 512

app = FastAPI(title="Maktspråk inference service")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
_model.to(_device)
_model.eval()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
_id2label: dict[int, str] = {int(k): v for k, v in _model.config.id2label.items()}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class PredictResponse(BaseModel):
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME_OR_PATH}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    inputs = _tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs: list[float] = softmax(logits, dim=1).squeeze().cpu().tolist()

    return PredictResponse(probabilities={_id2label[i]: p for i, p in enumerate(probs)})
