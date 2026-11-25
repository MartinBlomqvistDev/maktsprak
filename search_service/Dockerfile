# The search service (search_service/app.py), built from the repository root:
#
#   docker build -f search_service/Dockerfile --target service .
#
# The package is baked into the image instead of fetched at start, as
# inference_service bakes its model: with scale-to-zero every cold start would
# otherwise depend on the Hub answering, and a deployment would serve whatever
# the dataset holds that day. The fetch stage downloads the package at one
# pinned revision and installs it only if its manifest has the pinned hash and
# every file its hash (bake_index.py); the service checks it again at start.
#
# Targets:
#
#   service  the image Cloud Run runs: the libraries, the package, the code.
#   mounted  the same without the package, which a local run mounts at
#            /opt/index; for measuring the container before the package is
#            published.
#
# Cloud Run runs it with one vCPU and 4 GiB in the second generation of its
# execution environment, as docs/search/adr-001-index-store.md has it.

# The published package. A new package is a new revision and a new hash here,
# and a new image.
ARG INDEX_REPO=MartinBlomqvist/fraga-protokollet-index
ARG INDEX_REVISION=168672620ca36e69a0f032fd48afed9f605324a4
ARG INDEX_MANIFEST_SHA256=61af3617ff834bb148a8d81906e9c0e2e27413c5ea32915888561b58dbb862b4

FROM python:3.11-slim AS deps

WORKDIR /app

COPY search_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# One thread per compute pool, as the store was measured; the package's place,
# how it is checked at start, and the month's spend on a mounted bucket. The
# image owns these values so that no shell rewrites the paths on the way in.
#
# At start the service checks the manifest's pinned hash, every file's size
# and every count, but does not hash the files again: the build has hashed
# each one, and an image's layers do not change. Hashing them at start took
# 2.4 to 3.8 s in a local container with one CPU, on every cold start.
# INDEX_VERIFY=hashes does it anyway, as a local run of the mounted target may.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    INDEX_DIR=/opt/index \
    INDEX_VERIFY=sizes \
    BUDGET_LEDGER=/ledger/budget.json \
    PORT=8080

EXPOSE 8080

CMD exec uvicorn search_service.app:create_app_from_environment --factory --host 0.0.0.0 --port ${PORT} --no-access-log

FROM deps AS mounted
COPY search /app/search
COPY search_service /app/search_service
RUN python -m compileall -q /app/search /app/search_service

FROM python:3.11-slim AS fetch
RUN pip install --no-cache-dir huggingface_hub==2.4.0
COPY search_service/bake_index.py /bake_index.py
ARG INDEX_REPO
ARG INDEX_REVISION
ARG INDEX_MANIFEST_SHA256
RUN python /bake_index.py --repo "${INDEX_REPO}" --revision "${INDEX_REVISION}" \
    --manifest-sha256 "${INDEX_MANIFEST_SHA256}" --dest /opt/index

FROM deps AS service
ARG INDEX_MANIFEST_SHA256
ENV INDEX_MANIFEST_SHA256=${INDEX_MANIFEST_SHA256} HF_HUB_OFFLINE=1
COPY --from=fetch /opt/index /opt/index
COPY search /app/search
COPY search_service /app/search_service
RUN python -m compileall -q /app/search /app/search_service
