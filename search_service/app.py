"""The search service on Cloud Run: FastAPI over the package, for the page's proxy alone.

``/health`` needs no key and says only that the instance is up and which
package it holds; with the key it also says how the instance is set up and
what it has counted. ``/search``, ``/answer`` and ``/parties`` take a question
and its filters, need the shared secret the proxy sends in ``X-Search-Key``,
and answer in lines of JSON, one per part, so that the page can show the list
of results while the selector is still choosing. The proxy names its caller in
``X-Caller`` for the rate limit.

The secret is checked before any of a request's body is read, and a body holds
at most ``MAX_BODY_BYTES``, so a caller without the key costs the instance its
headers and nothing more.

``/answer`` and ``/parties`` choose quotations. They answer only while
``FRAGA_ANSWER_STEP`` is ``on``, the switch the page's proxy reads, and the
image holds the test run's report with a launch rule that holds; the report
also decides which selector runs. Otherwise they are refused with 403
``answer_step_off``, and the service reads neither the selector's key nor the
budget's ledger, so the search starts without them.

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
- ``FRAGA_ANSWER_STEP``: ``on`` to answer ``/answer`` and ``/parties``; any
  other value, or none, leaves the answer step off.
- ``ANSWER_REPORT``: the test run's report (``answer_test.json``), which the
  image carries once it is committed. Without it, or when its launch rule
  fails, the answer step stays off whatever the switch says.
- ``ANTHROPIC_API_KEY``: the selector's key, read only when the answer step is
  on and the report has the language model select; without it the baseline
  answers.
- ``BUDGET_LEDGER``: where the month's spend on the model is kept, an object
  ``gs://bucket/name`` that every instance shares, or a file for a local run.
  Without it the month counts as spent and the baseline answers; a ledger that
  cannot be read refuses the model until it can. The service starts all the
  same.

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
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from search.answer.budget import BudgetGuard, GuardedSelector, LedgerError, ledger_from
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

#: The switch of the answer step, named as the page's proxy names it.
ANSWER_STEP_ENV = "FRAGA_ANSWER_STEP"

#: The most bytes a request's body may hold. The longest question a request
#: allows takes at most 4 000 bytes in UTF-8, which leaves room for its filters
#: and for escapes.
MAX_BODY_BYTES = 8 * 1024

#: The paths a request reaches without the shared secret.
OPEN_PATHS = frozenset({"/health"})

#: The test run's report, where the image carries it.
DEFAULT_ANSWER_REPORT = "/app/reports/answer_test.json"

#: The selector's client: no retries of the SDK's own, since the selector sends
#: a failed request once more itself and the budget counts each request, and
#: 20 s per request, so that both requests fit within the answer step's
#: deadline (ANSWER_DEADLINE_S in service.py) and the baseline answers in time.
ANTHROPIC_RETRIES = 0
ANTHROPIC_TIMEOUT_S = 20.0


def _line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _error(status: int, code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status, headers=headers)


def _header(scope: Scope, name: bytes) -> bytes | None:
    """The first value of a header, as the connection sent it."""
    for key, value in scope["headers"]:
        if key == name:
            return value
    return None


def _replay(body: bytes, receive: Receive) -> Receive:
    """A receive that gives the body already read, then what the connection sends."""
    given = False

    async def replay() -> Message:
        nonlocal given
        if not given:
            given = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


class RequestGate:
    """Refuses a request before its body is read: without the shared secret, or too large.

    A pure ASGI middleware, outside FastAPI's parsing, so that the body of a
    request without the key is never read or decoded, and the service tells
    nobody without the key what shape a request takes. A request with the key
    has its body read here, up to the limit, whether it gives a length or
    streams, and handed on whole.

    Attributes:
        app: The application behind the gate.
        max_body: The most bytes a body may hold.
    """

    def __init__(self, app: ASGIApp, secret: str, max_body: int = MAX_BODY_BYTES) -> None:
        if not secret:
            raise ValueError("The service needs a shared secret.")
        self.app = app
        self.max_body = max_body
        self._secret = secret.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or (scope["type"] == "http" and scope["path"] in OPEN_PATHS):
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        given = _header(scope, SECRET_HEADER.encode("ascii")) or b""
        if not hmac.compare_digest(given, self._secret):
            await _error(401, "unauthorized")(scope, receive, send)
            return
        length = _header(scope, b"content-length")
        if length is not None and not length.isdigit():
            await _error(400, "invalid_request")(scope, receive, send)
            return
        if length is not None and int(length) > self.max_body:
            await _error(413, "invalid_request")(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return  # the caller went away before the body was in
            body += message.get("body", b"")
            if len(body) > self.max_body:
                await _error(413, "invalid_request")(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        await self.app(scope, _replay(bytes(body), receive), send)


def create_app(
    service: SearchService,
    secret: str,
    health: dict[str, Any],
    on_close: Callable[[], Awaitable[None]] | None = None,
    *,
    answer_step: bool = False,
    max_body: int = MAX_BODY_BYTES,
) -> FastAPI:
    """The service's routes over one :class:`SearchService`, behind the gate.

    Args:
        service: The service.
        secret: The shared secret every question must carry.
        health: What ``/health`` reports beside the counters.
        on_close: Run when the application shuts down.
        answer_step: Whether ``/answer`` and ``/parties`` answer; off unless
            switched on.
        max_body: The most bytes a request's body may hold.

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
        service.close()
        if on_close is not None:
            await on_close()

    app = FastAPI(
        title="Maktspråk search service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestGate, secret=secret, max_body=max_body)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error(400, "invalid_request")

    def admit(
        request: Request, body: Question, kind: Kind, parties: bool = False
    ) -> tuple[str, SearchFilter] | JSONResponse:
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
    def health_check(request: Request) -> dict[str, Any]:
        service.rates.tidy()
        given = (request.headers.get(SECRET_HEADER) or "").encode("utf-8")
        if not hmac.compare_digest(given, secret.encode("utf-8")):
            return {"status": "ok", "manifest_sha256": health.get("manifest_sha256")}
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
        if not answer_step:
            return _error(403, "answer_step_off")
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
                answered = await service.answer_in_time(text, found)
                yield _line(service.answer_line(answered, found))
            except Exception as error:
                logger.error("An answer failed: %s", type(error).__name__)
                yield _line({"type": "error", "error": "answer_failed"})

        return StreamingResponse(lines(), media_type=NDJSON)

    @app.post("/parties", response_model=None)
    async def parties(body: Question, request: Request) -> StreamingResponse | JSONResponse:
        if not answer_step:
            return _error(403, "answer_step_off")
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


class SpentGuard(BudgetGuard):
    """A month counted as spent, for a service with no ledger to count in.

    The language model is never asked, so nothing is spent that no ledger
    would record, and the baseline answers with the fallback ``budget``.
    """

    def __init__(self, monthly_usd: float) -> None:
        super().__init__(monthly_usd, None)

    def allows(self) -> bool:
        """Never: the month counts as spent."""
        return False


def answer_step_on(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the environment switches the answer step on; off unless it says ``on``."""
    value = environ.get(ANSWER_STEP_ENV, "")
    if value not in ("", "on", "off"):
        logger.warning("%s is neither on nor off, so the answer step is off.", ANSWER_STEP_ENV)
    return value == "on"


def launch_verdict(path: Path) -> tuple[str | None, str]:
    """What the test run's report lets the answer step run.

    Args:
        path: The report, ``answer_test.json``.

    Returns:
        The selector that runs (``selector`` or ``baseline``), or ``None`` when
        the answer step may not run, and the report's state: ``launch``,
        ``no_launch``, ``missing`` or ``unreadable``.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError):
        return None, "unreadable"
    if not isinstance(report, dict) or report.get("runs") not in ("selector", "baseline"):
        return None, "unreadable"
    if report.get("launch") is not True:
        return None, "no_launch"
    return report["runs"], "launch"


def make_selector(
    manifest: PackageManifest, store: ServiceStore, runs: str | None
) -> tuple[Selector, str, str]:
    """The selector the page runs, from the manifest, the test run's verdict and the environment.

    The prompt is checked whether or not the answer step runs, since it is part
    of the system the test split measured. With the answer step off nothing
    answers, and neither the key nor the ledger is read; so too when the report
    has the baseline run. When it has the language model run, the model
    answers within the month's budget if the key is set; without a ledger the
    month counts as spent, and a ledger that cannot be read refuses the model
    until it can, so the baseline answers.

    Args:
        manifest: The package's manifest.
        store: The store, whose BM25 the baseline scores with.
        runs: What the report lets run, ``selector`` or ``baseline``; ``None``
            with the answer step off.

    Returns:
        The selector, what answers (``none``, ``baseline`` or ``claude``), and
        the ledger's state (``none``, ``gcs``, ``file``, ``missing`` or
        ``unreadable``).
    """
    prompt = load_selector_prompt()
    if prompt.sha256 != manifest.answer.prompt_sha256:
        _refuse("the selector's prompt is not the one the test split measured")
    params = Bm25Params(k1=manifest.retrieval.k1, b=manifest.retrieval.b)
    baseline = BaselineSelector(store.lexical, params, manifest.answer.threshold)
    if runs is None:
        return baseline, "none", "none"
    if runs == "baseline" or not os.environ.get("ANTHROPIC_API_KEY"):
        return baseline, "baseline", "none"
    answer = manifest.answer
    claude = ClaudeSelector(
        prompt,
        answer.model,
        answer.effort,
        answer.max_tokens,
        Prices(input=answer.input_price, output=answer.output_price),
    )
    spent = GuardedSelector(claude, baseline, SpentGuard(answer.monthly_budget_usd))
    ledger = os.environ.get("BUDGET_LEDGER", "")
    shared = ledger.startswith("gs://")
    if not ledger or (not shared and not Path(ledger).parent.is_dir()):
        logger.error("The budget has no ledger to count in, so the month counts as spent.")
        return spent, "baseline", "missing"
    try:
        guard = BudgetGuard(answer.monthly_budget_usd, ledger_from(ledger))
    except ValueError:
        logger.error("BUDGET_LEDGER is not gs://bucket/object, so the month counts as spent.")
        return spent, "baseline", "missing"
    state = "gcs" if shared else "file"
    try:
        guard.spent()
    except LedgerError as error:
        logger.error("The budget's ledger cannot be read, so the model waits for it: %s", error)
        state = "unreadable"
    import anthropic

    client = anthropic.Anthropic(max_retries=ANTHROPIC_RETRIES, timeout=ANTHROPIC_TIMEOUT_S)
    return GuardedSelector(claude, baseline, guard, client), "claude", state


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

    switched_on = answer_step_on()
    runs, report = launch_verdict(Path(os.environ.get("ANSWER_REPORT", DEFAULT_ANSWER_REPORT)))
    if switched_on and runs is None:
        logger.warning(
            "%s is on, but the test run's report is %s, so the answer step stays off.",
            ANSWER_STEP_ENV,
            report.replace("_", " "),
        )
    answer_step = switched_on and runs is not None
    selector, selector_name, ledger = make_selector(manifest, store, runs if answer_step else None)
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
        "answer_step": "on" if answer_step else "off",
        "answer_report": report,
        "selector": selector_name,
        "ledger": ledger,
        "verify_s": round(verified_s, 2),
        "start_s": round(time.perf_counter() - started, 2),
    }
    logger.info(
        "Package %s verified by %s in %.1f s; answer step %s; ready in %.1f s",
        manifest.name,
        verify,
        verified_s,
        health["answer_step"],
        time.perf_counter() - started,
    )
    return create_app(service, secret, health, on_close=client.aclose, answer_step=answer_step)
