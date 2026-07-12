---
title: Maktsprak Inference
emoji: 🗳️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Maktspråk — inference service

FastAPI wrapper around `MartinBlomqvist/maktsprak_classifier_clean`
(KB-BERT fine-tuned for Swedish parliamentary party classification).
Serves the live "Testa modellen" demo at maktsprak.streamlit.app /
the Vercel site's `/api/predict` proxy. Not a general-purpose API.

## Endpoints

- `POST /predict` — `{"text": "..."}` -> `{"probabilities": {"S": 0.31, ...}}`
- `GET /health` — liveness check

Text sent here is not stored.
