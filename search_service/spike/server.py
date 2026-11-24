"""A minimal HTTP service around one store, one query model or both, measured the same way.

It is not the search service of phase 4. It puts every store and every query
model behind the same calls: ``/health``, which answers once everything is
loaded; ``/search`` for one search; ``/parties`` for the view of the eight
parties; ``/embed`` for the query model alone; and ``/usage`` for the
container's memory and CPU counters.

``SPIKE_STORE`` names the store, read from the directory in ``SPIKE_STORES``.
``SPIKE_MODEL``, ``SPIKE_RUNTIME`` and ``SPIKE_PRECISION`` name a query model,
read from the directory in ``SPIKE_MODELS``. With a model, a search embeds its
question first and the vector leg searches with that vector; without one, it
uses the question's synthetic vector. Everything is loaded before the server
accepts a connection, so a cold start covers it. Run as::

    uvicorn search_service.spike.server:create_app_from_environment --factory --port 8000
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol, cast, get_args

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from search_service.spike.cgroup import Usage, read_usage
from search_service.spike.contracts import (
    EmbedRequest,
    EmbedResponse,
    PartyViewRequest,
    PartyViewResponse,
    SearchRequest,
    SearchResponse,
)
from search_service.spike.numpy_store import STORE_DIR_NAME, NumpyStore
from search_service.spike.query_model import MODELS_DIR_NAME, model_directory
from search_service.spike.settings import (
    RUNTIME_PRECISIONS,
    Precision,
    Runtime,
    SpikeSettings,
    StoreSpec,
    load_spike_settings,
    variant_name,
)

STORE_ENV = "SPIKE_STORE"
STORES_ENV = "SPIKE_STORES"
MODEL_ENV = "SPIKE_MODEL"
RUNTIME_ENV = "SPIKE_RUNTIME"
PRECISION_ENV = "SPIKE_PRECISION"
MODELS_ENV = "SPIKE_MODELS"
DEFAULT_STORES = Path("/stores")
RUNTIMES: tuple[str, ...] = get_args(Runtime)


class Store(Protocol):
    """What the service needs of a store."""

    @property
    def passages(self) -> int:
        """How many passages the store holds."""
        ...

    def search(self, request: SearchRequest, vector: np.ndarray | None = None) -> SearchResponse:
        """One search."""
        ...

    def party_view(
        self, request: PartyViewRequest, vector: np.ndarray | None = None
    ) -> PartyViewResponse:
        """The best passage of each party."""
        ...


class Embedder(Protocol):
    """What the service needs of a query model."""

    @property
    def dim(self) -> int:
        """The dimension of its vectors."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed one question."""
        ...


class Startup(BaseModel):
    """How the service started.

    Attributes:
        store: The store's name, or ``None`` for a query model alone.
        kind: ``numpy`` or ``lance``, or ``None``.
        passages: How many passages the store holds.
        model: The query model with its runtime and precision, or ``None``.
        process_age_s: How long the process had run when loading began, the
            interpreter and the server's imports included; ``None`` where the
            kernel does not say.
        model_import_s: How long importing the query model's runtime took.
        model_load_s: How long loading the query model and its tokenizer took.
        model_first_s: How long embedding its first text took.
        model_s: The three together.
        open_s: How long opening the store took.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    store: str | None
    kind: str | None
    passages: int | None
    model: str | None = None
    process_age_s: float | None
    model_import_s: float | None = None
    model_load_s: float | None = None
    model_first_s: float | None = None
    model_s: float | None = None
    open_s: float | None


def process_age() -> float | None:
    """How long this process has run, from Linux's ``/proc``; ``None`` elsewhere.

    The start time in ``/proc/self/stat`` counts in clock ticks, a hundredth of
    a second on most systems.
    """
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError, AttributeError):
        return None
    return round(uptime - started, 3)


def store_directory(stores_dir: Path, spec: StoreSpec) -> Path:
    """Where a store lives: the numpy stores of synthetic vectors share one directory, and a
    store of real vectors or a LanceDB store has its own."""
    if spec.kind == "numpy" and not spec.real:
        return stores_dir / STORE_DIR_NAME
    return stores_dir / spec.name


def open_store(stores_dir: Path, spec: StoreSpec, settings: SpikeSettings) -> Store:
    """Open one store for search.

    Args:
        stores_dir: The directory the stores were built into.
        spec: The store.
        settings: The spike's settings.

    Returns:
        The store.
    """
    directory = store_directory(stores_dir, spec)
    if spec.kind == "numpy":
        return NumpyStore(directory, spec.dim, spec.format, settings.search)
    from search_service.spike.lance_store import LanceStore

    return LanceStore(directory, spec, settings)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def create_app(store: Store | None, embedder: Embedder | None, startup: Startup) -> FastAPI:
    """The service around an open store, a loaded query model, or both.

    Args:
        store: The store, or ``None``.
        embedder: The query model, or ``None``.
        startup: How the service started, returned by ``/health``.

    Returns:
        The application.
    """
    app = FastAPI(title="Storage spike", docs_url=None, redoc_url=None, openapi_url=None)

    def need_store() -> Store:
        if store is None:
            raise HTTPException(status_code=404, detail="This service holds no store.")
        return store

    @app.get("/health")
    def health() -> Startup:
        return startup

    @app.post("/search")
    def search(request: SearchRequest) -> SearchResponse:
        target = need_store()
        started = time.perf_counter()
        vector = embedder.embed(request.text) if embedder and request.mode != "lexical" else None
        embedded = _ms(started) if vector is not None else None
        response = target.search(request, vector)
        if embedded is None:
            return response
        timings = response.timings.model_copy(
            update={"embed_ms": embedded, "total_ms": _ms(started)}
        )
        return response.model_copy(update={"timings": timings})

    @app.post("/parties")
    def parties(request: PartyViewRequest) -> PartyViewResponse:
        target = need_store()
        started = time.perf_counter()
        vector = embedder.embed(request.text) if embedder else None
        embedded = _ms(started) if vector is not None else None
        response = target.party_view(request, vector)
        if embedded is None:
            return response
        timings = response.timings.model_copy(
            update={"embed_ms": embedded, "total_ms": _ms(started)}
        )
        return response.model_copy(update={"timings": timings})

    @app.post("/embed")
    def embed(request: EmbedRequest) -> EmbedResponse:
        if embedder is None:
            raise HTTPException(status_code=404, detail="This service holds no query model.")
        started = time.perf_counter()
        vector = embedder.embed(request.text)
        return EmbedResponse(dim=int(vector.shape[0]), embed_ms=_ms(started))

    @app.get("/usage")
    def usage() -> Usage:
        return read_usage()

    return app


def create_app_from_environment() -> FastAPI:
    """Load the query model and open the store the environment names, and build the service.

    Raises:
        ValueError: If it names neither, or a model whose dimension is not the store's.
    """
    settings = load_spike_settings()
    store_name = os.environ.get(STORE_ENV) or None
    model_name = os.environ.get(MODEL_ENV) or None
    if store_name is None and model_name is None:
        raise ValueError(f"Name a store in {STORE_ENV}, a query model in {MODEL_ENV}, or both.")
    stores_dir = Path(os.environ.get(STORES_ENV, str(DEFAULT_STORES)))
    age = process_age()

    embedder = None
    times = None
    label = None
    if model_name is not None:
        from search_service.spike.query_model import load_query_model

        runtime = os.environ.get(RUNTIME_ENV, "torch")
        precision = os.environ.get(PRECISION_ENV, "fp32")
        if runtime not in RUNTIMES:
            raise ValueError(f"{RUNTIME_ENV} must be one of {RUNTIMES}, not {runtime!r}.")
        if precision not in RUNTIME_PRECISIONS[runtime]:
            allowed = RUNTIME_PRECISIONS[runtime]
            raise ValueError(f"{runtime} runs a model in one of {allowed}, not {precision!r}.")
        spec = settings.query_models.model(model_name)
        models_dir = Path(os.environ.get(MODELS_ENV, str(stores_dir / MODELS_DIR_NAME)))
        chosen_runtime, chosen = cast(Runtime, runtime), cast(Precision, precision)
        embedder = load_query_model(
            model_directory(models_dir, spec, chosen_runtime, chosen),
            spec,
            chosen_runtime,
            chosen,
            settings.query_models.onnx,
        )
        times = embedder.times
        label = variant_name(spec.name, chosen_runtime, chosen)

    store = None
    open_s = None
    store_spec = None
    if store_name is not None:
        store_spec = settings.store(store_name)
        if embedder is not None and embedder.dim != store_spec.dim:
            raise ValueError(
                f"{label} gives {embedder.dim} dimensions; {store_name} holds {store_spec.dim}."
            )
        started = time.perf_counter()
        store = open_store(stores_dir, store_spec, settings)
        open_s = round(time.perf_counter() - started, 3)

    startup = Startup(
        store=store_name,
        kind=store_spec.kind if store_spec else None,
        passages=store.passages if store else None,
        model=label,
        process_age_s=age,
        model_import_s=times.import_s if times else None,
        model_load_s=times.load_s if times else None,
        model_first_s=times.first_s if times else None,
        model_s=times.total_s if times else None,
        open_s=open_s,
    )
    return create_app(store, embedder, startup)
