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
# Every library is installed from a lock with the hash of each file
# (requirements.lock, and bake-requirements.lock for the fetch stage), and the
# base image is pinned by its digest, so the image can be built again from its
# commit. The service runs as a user of its own, who can read the package and
# the code but change neither.
#
# Targets:
#
#   service      the image Cloud Run runs: the libraries, the package, the code.
#   mounted      the same without the package, which a local run mounts at
#                /opt/index.
#   fetch-tools  the fetch stage's libraries alone, to check its lock without
#                downloading the package.
#
# Cloud Run runs it with one vCPU and 4 GiB in the second generation of its
# execution environment, as docs/search/adr-001-index-store.md has it.

# The published package. A new package is a new revision and a new hash here,
# and a new image.
ARG INDEX_REPO=MartinBlomqvist/fraga-protokollet-index
ARG INDEX_REVISION=168672620ca36e69a0f032fd48afed9f605324a4
ARG INDEX_MANIFEST_SHA256=61af3617ff834bb148a8d81906e9c0e2e27413c5ea32915888561b58dbb862b4

# python:3.11-slim, as its index on Docker Hub stood on 2026-11-27.
ARG PYTHON_IMAGE=python:3.11-slim@sha256:61e01f72680f86a5279ad510818ab7b2ee9f049822589d4e34e33e044360356c

FROM ${PYTHON_IMAGE} AS deps

WORKDIR /app

COPY search_service/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /app/requirements.lock \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin search

# One thread per compute pool, as the store was measured; the package's place
# and how it is checked at start; and the test run's report, which decides
# whether the answer step may run. The image owns these values so that no
# shell rewrites the paths on the way in.
#
# At start the service checks the manifest's pinned hash, every file's size
# and every count, but does not hash the files again: the build has hashed
# each one, the service's user cannot change them, and hashing them at start
# took 2.4 to 3.8 s in a local container with one CPU, on every cold start.
# INDEX_VERIFY=hashes does it anyway, as a local run of the mounted target may.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    INDEX_DIR=/opt/index \
    INDEX_VERIFY=sizes \
    ANSWER_REPORT=/app/reports/answer_test.json \
    PORT=8080

EXPOSE 8080

CMD exec uvicorn search_service.app:create_app_from_environment --factory --host 0.0.0.0 --port ${PORT} --no-access-log

# The test run's committed files. The pattern matches the files of the run
# that are committed already, so the copy holds before the launch report,
# answer_test.json, exists; the service reads only that one.
FROM deps AS mounted
COPY search /app/search
COPY search_service /app/search_service
COPY data/reports/search/answer_test*.json /app/reports/
RUN python -m compileall -q /app/search /app/search_service
USER 10001

FROM ${PYTHON_IMAGE} AS fetch-tools
COPY search_service/bake-requirements.lock /bake-requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /bake-requirements.lock

FROM fetch-tools AS fetch
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
COPY data/reports/search/answer_test*.json /app/reports/
RUN python -m compileall -q /app/search /app/search_service
USER 10001
