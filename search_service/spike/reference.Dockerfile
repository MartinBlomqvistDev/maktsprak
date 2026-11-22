# The inference service's image, rebuilt as the storage spike's reference for
# a cold start on Cloud Run (docs/search/adr-001-index-store.md). The base,
# torch, requirements, app, environment and command are those of
# inference_service/Dockerfile. Only the weights come from elsewhere: that
# image downloads the classifier from Hugging Face while it is built, and this
# one saves it with the same calls and the same transformers from the copy of
# the Hugging Face cache that run.py stages as model-src/, so that nothing is
# downloaded. A separate stage keeps that copy out of the image.
FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY inference_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS bake
COPY model-src /tmp/model-src
RUN python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; M.from_pretrained('/tmp/model-src').save_pretrained('/opt/model'); T.from_pretrained('/tmp/model-src').save_pretrained('/opt/model')"

FROM base
ENV MODEL_ID=MartinBlomqvist/maktsprak_classifier_clean
ENV MODEL_NAME_OR_PATH=/opt/model
COPY --from=bake /opt/model /opt/model

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY inference_service/app.py .

ENV PORT=7860
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
