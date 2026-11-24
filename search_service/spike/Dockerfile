# The images every measurement of the storage spike runs in
# (docs/search/adr-001-index-store.md). run.py builds both targets from a
# staged copy of search/ and search_service/, since the search code is outside
# this folder:
#
#   slim  without torch: every store, and a query model run by ONNX Runtime
#   full  slim's libraries and torch: a query model run by torch, and the
#         building of the stores and the models
#
# The stores are built once into a Docker volume and mounted read-only, so one
# image serves every store. The service of phase 4 bakes its one store into its
# image instead; both read the files from the same kind of disk.
#
# slim comes before torch, since Docker's builder without BuildKit builds every
# stage up to the target it is asked for.
FROM python:3.11-slim AS base

WORKDIR /app

COPY search_service/spike/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# LanceDB logs a warning on every search that selects columns without its
# score column; only errors are logged. No model is ever fetched at run time.
ENV PYTHONUNBUFFERED=1 LANCEDB_LOG=error HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "search_service.spike.server:create_app_from_environment", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--log-level", "warning"]

# The code goes last in each target, so that a change to it does not reinstall
# the libraries. It is compiled now, so that a cold start does not compile the
# code it imports.
FROM base AS slim
COPY search /app/search
COPY search_service /app/search_service
RUN python -m compileall -q /app/search /app/search_service

# torch for the query models, from PyTorch's index of builds without CUDA, as
# inference_service installs it. The version is the one phase 3 embeds with.
FROM base AS torch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY search_service/spike/requirements-torch.txt /app/requirements-torch.txt
RUN pip install --no-cache-dir -r /app/requirements-torch.txt

FROM torch AS full
COPY search /app/search
COPY search_service /app/search_service
RUN python -m compileall -q /app/search /app/search_service
