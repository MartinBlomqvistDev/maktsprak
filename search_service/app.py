"""The search service on Cloud Run: FastAPI over the package, for the page's proxy alone.

``/health`` says what the instance holds and needs no key. ``/search``,
``/answer`` and ``/parties`` take a question and its filters, need the shared
secret the proxy sends in ``X-Search-Key``, and answer in lines of JSON, one per
part, so that the page can show the list of results while the selector is still
choosing. The proxy names its caller in ``X-Caller`` for the rate limit.

The service refuses to start unless the package is the one its manifest
describes, the manifest is the one the image pins, and the prompt, the question's
model and the store are the ones the manifest records.

Environment:

- ``INDEX_DIR``: the package, baked into the image.
- ``INDEX_MANIFEST_SHA256``: the hash the package's manifest must have.
- ``INDEX_VERIFY``: ``hashes`` to hash every file at start, ``sizes`` to check
  only that every file is there with its size.
- ``SEARCH_SHARED_SECRET``: the secret the proxy holds.
- ``OPENROUTER_API_KEY``: the key of the embedding API.
- ``ANTHROPIC_API_KEY``: the selector's key; without it the baseline answers.
- ``BUDGET_LEDGER``: the file the month's spend on the model is kept in, on a
  mounted bucket so that a new instance does not start the month again;
  without it the spend is counted in memory.

Usage::

    uvicorn search_service.app:create_app_from_environment --factory
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from search.answer.budget import BudgetGuard, GuardedSelector
from search.answer.selector import BaselineSelector, ClaudeSelector, load_selector_prompt
from search.config import Prices, load_settings
from search.index.embed import OPENROUTER_KEY, EmbedderSpec, OpenRouterEmbedder
from search.index.lexical import Bm25Params
from search_service.package import (
    STORE_DIR,
    PackageManifest,
    SpeechTable,
    read_manifest,
    verify_package,
)
from search_service.service import (
    InvalidRequest,
    Kind,
    Question,
    RateLimiter,
    SearchService,
    Selector,
    ServiceStore,
)
from search_service.spike.contracts import SearchFilter
from search_service.spike.settings import SearchSettings

logger = logging.getLogger("search_service")

SECRET_HEADER = "x-search-key"
CALLER_HEADER = "x-caller"
NDJSON = "application/x-ndjson"
DEFAULT_INDEX_DIR = "/opt/index"
VERIFY_MODES = ("hashes", "sizes")

#: The selector's client: one retry by the SDK, and a minute per request, so a
#: question whose model does not answer falls back to the baseline in time.
ANTHROPIC_RETRIES = 1
ANTHROPIC_TIMEOUT_S = 60.0


def _line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _error(status: int, code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status, headers=headers)


def create_app(
    service: SearchService,
    secret: str,
    health: dict[str, Any],
    on_close: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """The service's routes over one :class:`SearchService`.

    Args:
        service: The service.
        secret: The shared secret every question must carry.
        health: What ``/health`` reports beside the counters.
        on_close: Run when the application shuts down.

    Returns:
        The application.

    Raises:
        ValueError: If the secret is empty.
    """
    if not secret:
        raise ValueError("The service needs a shared secret.")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if on_close is not None:
            await on_close()

    app = FastAPI(
        title="Maktspråk search service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error(400, "invalid_request")

    def admit(
        request: Request, body: Question, kind: Kind, parties: bool = False
    ) -> tuple[str, SearchFilter] | JSONResponse:
        given = request.headers.get(SECRET_HEADER, "").encode("utf-8")
        if not hmac.compare_digest(given, secret.encode("utf-8")):
            return _error(401, "unauthorized")
        try:
            text, search_filter = service.read(body, parties=parties)
        except (InvalidRequest, ValueError):
            return _error(400, "invalid_request")
        caller = request.headers.get(CALLER_HEADER) or (
            request.client.host if request.client else "unknown"
        )
        wait = service.rates.check(caller, kind)
        if wait is not None:
            service.counters["rate_limited"] += 1
            return _error(429, "rate_limited", {"Retry-After": str(math.ceil(wait))})
        return text, search_filter

    async def find(text: str, search_filter: SearchFilter):
        vector, waited = await service.vector(text)
        return await asyncio.to_thread(service.find, text, search_filter, vector, waited)

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        return {**health, "counters": dict(service.counters)}

    @app.post("/search", response_model=None)
    async def search(body: Question, request: Request) -> StreamingResponse | JSONResponse:
        admitted = admit(request, body, "search")
        if isinstance(admitted, JSONResponse):
            return admitted
        try:
            found = await find(*admitted)
        except Exception as error:
            logger.error("A search failed: %s", type(error).__name__)
            return _error(502, "search_failed")
        service.counters["searches"] += 1
        return StreamingResponse(iter([_line(service.search_line(found))]), media_type=NDJSON)

    @app.post("/answer", response_model=None)
    async def answer(body: Question, request: Request) -> StreamingResponse | JSONResponse:
        admitted = admit(request, body, "answer")
        if isinstance(admitted, JSONResponse):
            return admitted
        text = admitted[0]
        try:
            found = await find(*admitted)
        except Exception as error:
            logger.error("A search failed: %s", type(error).__name__)
            return _error(502, "search_failed")
        service.counters["answers"] += 1

        async def lines():
            yield _line(service.search_line(found))
            try:
                answered = await asyncio.to_thread(service.answer, text, found)
                yield _line(service.answer_line(answered, found))
            except Exception as error:
                logger.error("An answer failed: %s", type(error).__name__)
                yield _line({"type": "error", "error": "answer_failed"})

        return StreamingResponse(lines(), media_type=NDJSON)

    @app.post("/parties", response_model=None)
    async def parties(body: Question, request: Request) -> StreamingResponse | JSONResponse:
        admitted = admit(request, body, "search", parties=True)
        if isinstance(admitted, JSONResponse):
            return admitted
        text, search_filter = admitted
        try:
            vector, waited = await service.vector(text)
            result = await asyncio.to_thread(service.parties, text, search_filter, vector)
        except Exception as error:
            logger.error("The view of the parties failed: %s", type(error).__name__)
            return _error(502, "search_failed")
        service.counters["parties"] += 1
        retrieval = "hybrid" if vector is not None else "bm25"
        line = service.parties_line(result, retrieval, waited)
        return StreamingResponse(iter([_line(line)]), media_type=NDJSON)

    return app


def _refuse(problem: str) -> None:
    raise SystemExit(f"The search service will not start: {problem}.")


def make_selector(manifest: PackageManifest, store: ServiceStore) -> tuple[Selector, str, str]:
    """The selector the page runs, from the manifest and the environment.

    Returns:
        The selector, its name and where it keeps the month's spend.
    """
    prompt = load_selector_prompt()
    if prompt.sha256 != manifest.answer.prompt_sha256:
        _refuse("the selector's prompt is not the one the test split measured")
    params = Bm25Params(k1=manifest.retrieval.k1, b=manifest.retrieval.b)
    baseline = BaselineSelector(store.lexical, params, manifest.answer.threshold)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return baseline, "baseline", "none"
    import anthropic

    client = anthropic.Anthropic(max_retries=ANTHROPIC_RETRIES, timeout=ANTHROPIC_TIMEOUT_S)
    answer = manifest.answer
    claude = ClaudeSelector(
        prompt,
        answer.model,
        answer.effort,
        answer.max_tokens,
        Prices(input=answer.input_price, output=answer.output_price),
    )
    ledger = os.environ.get("BUDGET_LEDGER")
    path = Path(ledger) if ledger and Path(ledger).parent.is_dir() else None
    guard = BudgetGuard(answer.monthly_budget_usd, path)
    return GuardedSelector(claude, baseline, guard, client), "claude", "file" if path else "memory"


def create_app_from_environment() -> FastAPI:
    """Check the package, open it, and build the application from the environment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    directory = Path(os.environ.get("INDEX_DIR", DEFAULT_INDEX_DIR))
    verify = os.environ.get("INDEX_VERIFY", "hashes")
    if verify not in VERIFY_MODES:
        _refuse(f"INDEX_VERIFY must be one of {VERIFY_MODES}")
    pinned = os.environ.get("INDEX_MANIFEST_SHA256") or None
    manifest = verify_package(directory, manifest_sha256=pinned, hashes=verify == "hashes")
    _, digest = read_manifest(directory)
    verified_s = time.perf_counter() - started

    settings = load_settings()
    candidate = settings.vectors.candidate(manifest.embedding.name)
    spec = EmbedderSpec.from_candidate(candidate)
    recorded = manifest.embedding
    if (spec.model, spec.revision, spec.provider, spec.quantization, spec.dimension) != (
        recorded.model,
        recorded.revision,
        recorded.provider,
        recorded.quantization,
        recorded.dimension,
    ) or spec.query_prompt != recorded.query_prompt:
        _refuse("the question's model in the settings is not the one the package was built for")
    secret = os.environ.get("SEARCH_SHARED_SECRET", "")
    key = os.environ.get(OPENROUTER_KEY, "")
    if not secret or not key:
        _refuse(f"SEARCH_SHARED_SECRET and {OPENROUTER_KEY} must both be set")
    if candidate.price_per_million_usd is None:
        _refuse("the question's model has no price")

    retrieval = manifest.retrieval
    store = ServiceStore(
        directory / STORE_DIR,
        recorded.dimension,
        "float32",
        SearchSettings(
            depth=retrieval.depth,
            hits=retrieval.hits,
            rrf_k=retrieval.rrf_k,
            gather_below=retrieval.gather_below,
        ),
    )
    if (
        store.manifest.analyzer.name != retrieval.analyzer
        or (store.params.k1, store.params.b) != (retrieval.k1, retrieval.b)
        or store.manifest.chunk_rows != retrieval.chunk_rows
        or store.passages != manifest.passages
    ):
        _refuse("the store is not the one the manifest records")
    speeches = SpeechTable(directory, store.texts)
    if speeches.rows != manifest.speeches:
        _refuse("the speeches' table is not the one the manifest records")

    embedder = OpenRouterEmbedder(
        spec,
        key,
        float(candidate.price_per_million_usd),
        settings.service.embedding_cap_usd,
        batch_inputs=1,
        concurrency=1,
    )
    client = embedder.open_client()

    async def embed(text: str):
        return await embedder.embed_one(client, text, "query")

    selector, selector_name, ledger = make_selector(manifest, store)
    service_settings = settings.service
    service = SearchService(
        manifest,
        store,
        speeches,
        embed,
        selector,
        RateLimiter(
            service_settings.rate_window_s,
            {
                "search": service_settings.searches_per_window,
                "answer": service_settings.answers_per_window,
            },
        ),
        service_settings.question_chars,
    )
    health = {
        "status": "ok",
        "package": manifest.name,
        "manifest_sha256": digest,
        "verified": verify,
        "passages": manifest.passages,
        "speeches": manifest.speeches,
        "last_date": manifest.last_date.isoformat(),
        "embedding": recorded.name,
        "selector": selector_name,
        "ledger": ledger,
        "verify_s": round(verified_s, 2),
        "start_s": round(time.perf_counter() - started, 2),
    }
    logger.info(
        "Package %s verified by %s in %.1f s; ready in %.1f s",
        manifest.name,
        verify,
        verified_s,
        time.perf_counter() - started,
    )
    return create_app(service, secret, health, on_close=client.aclose)
