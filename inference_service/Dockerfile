FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image instead of fetching them at startup.
#
# The service previously called from_pretrained() against the Hub on every cold
# start. With scale-to-zero that is every idle period, so a live demo depended on
# Hugging Face being reachable AND un-rate-limited at the exact moment a visitor
# arrived. It eventually wasn't: HF returned 429 Too Many Requests, the container
# could not start, and Cloud Run served 503 until someone noticed.
#
# Downloading at build time makes the image self-contained: no network in the
# request path, a much faster cold start (no 500 MB transfer), and a deployment
# pinned to exactly the weights it was built with rather than to whatever the Hub
# happens to serve later.
#
# A separate script rather than an inline heredoc: heredocs in RUN need BuildKit
# (`# syntax=docker/dockerfile:1.4`), which Cloud Build's default builder does
# not guarantee.
ENV MODEL_ID=MartinBlomqvist/maktsprak_classifier_clean
ENV MODEL_NAME_OR_PATH=/opt/model
COPY bake_model.py .
RUN python bake_model.py && rm bake_model.py

# Belt and braces: even if some code path passed a Hub id, this forbids a
# network call rather than letting it silently reintroduce the failure above.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY app.py .

ENV PORT=7860
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
