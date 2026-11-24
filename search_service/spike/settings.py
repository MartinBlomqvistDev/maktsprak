"""Settings of the storage spike, read from ``spike.yaml`` or a file beside it."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPIKE_FILE: Path = Path(__file__).resolve().with_name("spike.yaml")

#: Names another settings file beside ``spike.yaml``, which every part of the
#: spike then reads instead: the builder, the server and the runner.
SETTINGS_ENV = "SPIKE_SETTINGS"

#: A path below the search-data directory: no part of it starts with a dot, so none is ``..``.
DATA_PATH_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*$"

VectorFormat = Literal["float32", "float16", "int8"]
StoreKind = Literal["numpy", "lance"]
VectorIndex = Literal["none", "IVF_PQ", "IVF_HNSW_SQ"]
FilterKind = Literal["party", "period", "speaker", "party_period"]
Runtime = Literal["torch", "onnx"]
Precision = Literal["fp32", "bf16", "int8"]

#: The precisions each runtime runs a query model in: torch in float32 or
#: bfloat16, ONNX Runtime in float32 or with its weights quantized to int8.
RUNTIME_PRECISIONS: dict[str, tuple[str, ...]] = {
    "torch": ("fp32", "bf16"),
    "onnx": ("fp32", "int8"),
}


def variant_name(name: str, runtime: Runtime, precision: Precision) -> str:
    """The name of a query model in one runtime and one precision, as runs are named."""
    return f"{name}-{runtime}-{precision}"


class _Frozen(BaseModel):
    """Base for settings: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Budget(_Frozen):
    """The bounds of the pre-registration's storage rule.

    Attributes:
        p95_ms: The highest warm search p95, in milliseconds.
        cold_start_s: The longest cold start, in seconds.
        memory_gib: The most memory, in GiB.
    """

    p95_ms: float = Field(gt=0)
    cold_start_s: float = Field(gt=0)
    memory_gib: float = Field(gt=0)


class ContainerSettings(_Frozen):
    """The limits every measured container runs under.

    Attributes:
        cpus: The value of ``docker run --cpus``.
        memory: The value of ``docker run --memory``; ``g`` means GiB.
        threads: Environment variables that size the thread pools to the CPUs.
    """

    cpus: str
    memory: str
    threads: dict[str, str]


class VectorSettings(_Frozen):
    """How the synthetic vectors are drawn.

    Attributes:
        seed: The seed every chunk's generator starts from.
        chunk_rows: How many vectors one chunk holds.
    """

    seed: int
    chunk_rows: int = Field(gt=0)


class SearchSettings(_Frozen):
    """The shape of one search.

    Attributes:
        depth: How many passages each leg ranks before fusion.
        hits: How many passages come back, with their text.
        rrf_k: The constant of reciprocal rank fusion.
        gather_below: Below this share of all passages allowed by a filter, a
            numpy store scores only those passages.
    """

    depth: int = Field(gt=0)
    hits: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    gather_below: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _hits_within_depth(self) -> SearchSettings:
        if self.hits > self.depth:
            raise ValueError("A search cannot return more hits than its legs rank.")
        return self


class FtsSettings(_Frozen):
    """LanceDB's full-text index, as :class:`lancedb.index.FTS` takes it.

    Attributes:
        language: The language of the stemmer and of the stop words.
        stem: Whether tokens are stemmed.
        remove_stop_words: Whether the language's stop words are dropped.
        ascii_folding: Whether letters are folded to ASCII, which makes å and a one letter.
        lower_case: Whether tokens are lower-cased.
        max_token_length: Tokens longer than this many bytes are dropped; ``None`` keeps all.
        with_position: Whether positions are stored, which phrase queries need.
    """

    language: str
    stem: bool
    remove_stop_words: bool
    ascii_folding: bool
    lower_case: bool
    max_token_length: int | None
    with_position: bool


class LanceSettings(_Frozen):
    """How a LanceDB store is indexed.

    Attributes:
        fts: The full-text index.
        scalar_indexes: The scalar index of each filter column, by column.
        metric: The distance every vector search and index uses.
    """

    fts: FtsSettings
    scalar_indexes: dict[Literal["party", "day", "person_id"], Literal["BTREE", "BITMAP"]]
    metric: Literal["dot", "cosine", "l2"]


class ProbeSettings(_Frozen):
    """The check of LanceDB's full-text search against its documentation.

    Attributes:
        seed: Decides the words drawn for the check of the stemmer.
        common_tokens: How many of the corpus's commonest words are checked.
        drawn_tokens: How many more are drawn at random from its vocabulary.
        passage_step: Every this many passages is indexed for the check of BM25.
        one_word_queries: The words whose one-word queries fit ``k1`` and ``b``.
        hits_per_query: How many hits of each one-word query are compared.
        k1_grid: The values of ``k1`` tried: first, last and step.
        b_grid: The values of ``b`` tried: first, last and step.
        questions: How many natural questions of the development split are scored both ways.
    """

    seed: int
    common_tokens: int = Field(ge=0)
    drawn_tokens: int = Field(ge=0)
    passage_step: int = Field(gt=0)
    one_word_queries: tuple[str, ...]
    hits_per_query: int = Field(gt=0)
    k1_grid: tuple[float, float, float]
    b_grid: tuple[float, float, float]
    questions: int = Field(ge=0)

    @field_validator("k1_grid", "b_grid")
    @classmethod
    def _grid_is_usable(cls, grid: tuple[float, float, float]) -> tuple[float, float, float]:
        first, last, step = grid
        if step <= 0 or last < first:
            raise ValueError("A grid needs a positive step and a last value not below its first.")
        return grid

    @staticmethod
    def values(grid: tuple[float, float, float]) -> list[float]:
        """The values of a grid, from its first to its last, rounded to six decimals."""
        first, last, step = grid
        count = int(round((last - first) / step)) + 1
        return [round(first + number * step, 6) for number in range(count)]


class WorkloadSettings(_Frozen):
    """Which questions are sent, and under which filters.

    Attributes:
        seed: Decides the filters drawn and the questions of the warm-up and the party view.
        sets: The question sets whose development split is sent.
        warmup_set: The set the warm-up questions are drawn from.
        warmup_queries: How many warm-up questions go first, not counted.
        filters: The weight of each kind of filter drawn for a question without one.
        party_view_questions: How many natural questions the view of the eight parties
            is measured on.
        draw_filters: Whether a question without a filter of its own gets one
            drawn; otherwise it is sent without a filter.
    """

    seed: int
    sets: tuple[Literal["natural", "keyword", "filtered", "stance", "interpellation"], ...]
    warmup_set: Literal["natural", "keyword", "filtered", "stance", "interpellation"]
    warmup_queries: int = Field(ge=0)
    filters: dict[FilterKind, float]
    party_view_questions: int = Field(ge=0)
    draw_filters: bool = True

    @field_validator("filters")
    @classmethod
    def _weights_are_usable(cls, weights: dict[FilterKind, float]) -> dict[FilterKind, float]:
        if not weights or any(weight < 0 for weight in weights.values()):
            raise ValueError("The filter weights must be at least one, none below 0.")
        if sum(weights.values()) <= 0:
            raise ValueError("The filter weights cannot all be 0.")
        return weights


class Images(_Frozen):
    """The two images of the spike, built from one Dockerfile.

    Attributes:
        slim: Without torch: every store, and a query model run with ONNX Runtime.
        full: With torch: a query model run with torch, and the building of the
            stores and the models.
    """

    slim: str
    full: str


class RunSettings(_Frozen):
    """How the measurements are run.

    Attributes:
        cold_starts: How many cold starts each store is measured with.
        port: The host port the measured container answers on.
        images: The names of the images the measurements run in.
        volume: The Docker volume the stores are built into.
        leg_queries: How many searches each leg alone is measured on.
        sweep_queries: How many searches each point of a sweep is measured on.
        phase_limit_s: A phase stops once it has run this long.
        ready_timeout_s: How long a container may take to answer ``/health``.
        request_timeout_s: How long one request may take.
    """

    cold_starts: int = Field(gt=0)
    port: int = Field(gt=0, lt=65536)
    images: Images
    volume: str
    leg_queries: int = Field(ge=0)
    sweep_queries: int = Field(ge=0)
    phase_limit_s: float = Field(gt=0)
    ready_timeout_s: float = Field(gt=0)
    request_timeout_s: float = Field(gt=0)


class AnnParams(_Frozen):
    """Search parameters of an approximate vector index; ``None`` keeps LanceDB's default.

    Attributes:
        nprobes: How many partitions of an IVF index are searched.
        refine_factor: Re-rank this many times the hits asked for with the full vectors.
        ef: The size of the candidate list of an HNSW search.
    """

    nprobes: int | None = Field(default=None, gt=0)
    refine_factor: int | None = Field(default=None, gt=0)
    ef: int | None = Field(default=None, gt=0)

    @property
    def label(self) -> str:
        """A short name, ``default`` when nothing is set."""
        parts = [f"{name}={value}" for name, value in self.model_dump().items() if value]
        return ",".join(parts) or "default"


class StoreSpec(_Frozen):
    """One store that is built and measured.

    Attributes:
        name: The store's name, unique in the settings.
        kind: ``numpy`` for option B, ``lance`` for option A.
        dim: The vectors' dimension.
        format: How each vector is stored.
        index: LanceDB's vector index, or ``none`` for exact search.
        sweep: Further parameters an approximate index is measured with.
        vectors_from: A vector index of :mod:`search.index.vector`, as a path
            below the search-data directory, whose vectors the store holds
            instead of synthetic ones; for a numpy store only.
        queries_from: The questions' vectors as the service asks for them, as a
            path below the search-data directory; given with ``vectors_from``,
            and the store then searches each question with its own vector.
    """

    name: str = Field(pattern=r"^[a-z0-9-]+$")
    kind: StoreKind
    dim: int = Field(gt=0)
    format: VectorFormat
    index: VectorIndex = "none"
    sweep: tuple[AnnParams, ...] = ()
    vectors_from: str | None = Field(default=None, pattern=DATA_PATH_PATTERN)
    queries_from: str | None = Field(default=None, pattern=DATA_PATH_PATTERN)

    @model_validator(mode="after")
    def _check_combination(self) -> StoreSpec:
        if self.kind == "numpy" and self.index != "none":
            raise ValueError(f"{self.name}: a numpy store searches exactly and has no index.")
        if self.kind == "lance" and self.format == "int8":
            raise ValueError(f"{self.name}: LanceDB stores vectors as float16, float32 or float64.")
        if self.index == "none" and self.sweep:
            raise ValueError(f"{self.name}: only an approximate index has parameters to sweep.")
        if (self.vectors_from is None) != (self.queries_from is None):
            raise ValueError(
                f"{self.name}: real vectors are searched with real questions, so "
                "vectors_from and queries_from come together."
            )
        if self.vectors_from is not None and self.kind != "numpy":
            raise ValueError(f"{self.name}: only a numpy store is built from real vectors.")
        return self

    @property
    def real(self) -> bool:
        """Whether the store holds real vectors rather than synthetic ones."""
        return self.vectors_from is not None


class QueryModelSpec(_Frozen):
    """One local model a service could embed its questions with.

    Attributes:
        name: The model's short name, unique in the settings.
        model: Its id on Hugging Face.
        revision: The commit its weights are read at.
        dim: The dimension of its vectors.
        query_prompt: The text put before a question, as its model card gives it.
        max_tokens: The most tokens it reads of a text, or its own limit where that is lower.
    """

    name: str = Field(pattern=r"^[a-z0-9-]+$")
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dim: int = Field(gt=0)
    query_prompt: str
    max_tokens: int = Field(gt=0)


class OnnxSettings(_Frozen):
    """How a query model is exported to ONNX and run with ONNX Runtime.

    Attributes:
        opset: The ONNX opset the model is exported with.
        threads: The threads of ONNX Runtime's pools, within one operator and between operators.
    """

    opset: int = Field(gt=0)
    threads: int = Field(gt=0)


class QueryModelSettings(_Frozen):
    """The local query models, and the runtimes and precisions each is measured in.

    Attributes:
        runtimes: The precisions each runtime is measured in, by runtime.
        onnx: How a model is exported to ONNX and run.
        check_texts: How many questions of the workload each prepared model is
            checked on against sentence-transformers in float32.
        models: The models.
    """

    runtimes: dict[Runtime, tuple[Precision, ...]]
    onnx: OnnxSettings
    check_texts: int = Field(gt=0)
    models: tuple[QueryModelSpec, ...]

    @field_validator("runtimes")
    @classmethod
    def _precisions_fit_their_runtime(
        cls, runtimes: dict[Runtime, tuple[Precision, ...]]
    ) -> dict[Runtime, tuple[Precision, ...]]:
        for runtime, precisions in runtimes.items():
            wrong = [p for p in precisions if p not in RUNTIME_PRECISIONS[runtime]]
            if wrong:
                allowed = ", ".join(RUNTIME_PRECISIONS[runtime])
                raise ValueError(f"{runtime} runs a model in {allowed}, not in {', '.join(wrong)}.")
        onnx = runtimes.get("onnx", ())
        if "int8" in onnx and onnx[0] != "fp32":
            raise ValueError(
                "onnx int8 is quantized from the float32 export, so onnx lists fp32 first."
            )
        return runtimes

    @field_validator("models")
    @classmethod
    def _names_are_unique(cls, models: tuple[QueryModelSpec, ...]) -> tuple[QueryModelSpec, ...]:
        names = [model.name for model in models]
        if len(set(names)) != len(names):
            raise ValueError("Two query models share a name.")
        return models

    def variants(self) -> list[tuple[Runtime, Precision]]:
        """Every runtime with every precision it is measured in, in the order of the settings."""
        return [(runtime, p) for runtime, precisions in self.runtimes.items() for p in precisions]

    def model(self, name: str) -> QueryModelSpec:
        """The query model called *name*.

        Raises:
            KeyError: If the settings name no such model.
        """
        for spec in self.models:
            if spec.name == name:
                return spec
        raise KeyError(f"The settings name no query model called {name!r}.")


class ReferenceSettings(_Frozen):
    """The inference service's image, started cold as a reference for Cloud Run.

    Attributes:
        image: The name the reference image is built under.
        model: The classifier's id on Hugging Face.
        revision: The revision of the classifier read from the Hugging Face cache.
        port: The port the service listens on inside its container.
        text: The text of every request, as the keep-warm workflow sends it.
        cold_starts: How many cold starts are measured.
        warm_requests: How many requests follow the last cold start.
        cloud_run_cold_s: The cold start on Cloud Run the repository records.
        cloud_run_warm_s: The warm answer it records beside it.
        cloud_run_source: The file of the repository that records both.
        cloud_run_cpu: The CPUs the service has on Cloud Run.
        cloud_run_memory: Its memory on Cloud Run, as Cloud Run writes it.
        cloud_run_startup_cpu_boost: Whether Cloud Run gives it more CPU while it starts.
        pinned_cpu: The CPU of Docker's virtual machine that a second set of
            cold starts pins the container to, beside the limits, so that its
            processes see one CPU; ``None`` measures under the limits alone.
    """

    image: str
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    port: int = Field(gt=0, lt=65536)
    text: str = Field(min_length=1)
    cold_starts: int = Field(gt=0)
    warm_requests: int = Field(ge=0)
    cloud_run_cold_s: float = Field(gt=0)
    cloud_run_warm_s: float = Field(gt=0)
    cloud_run_source: str
    cloud_run_cpu: str
    cloud_run_memory: str
    cloud_run_startup_cpu_boost: bool
    pinned_cpu: str | None = Field(default=None, pattern=r"^\d+$")


class SpikeSettings(_Frozen):
    """Everything ``spike.yaml`` holds.

    Attributes:
        budget: The bounds of the storage rule.
        container: The limits of the measured containers.
        vectors: How the synthetic vectors are drawn.
        search: The shape of one search.
        lance: How a LanceDB store is indexed.
        probe: The check of LanceDB's full-text search.
        workload: Which questions are sent.
        runs: How the measurements are run.
        stores: The stores built and measured.
        query_models: The local models a service could embed its questions with.
        reference: The inference service's image, measured as a reference.
    """

    budget: Budget
    container: ContainerSettings
    vectors: VectorSettings
    search: SearchSettings
    lance: LanceSettings
    probe: ProbeSettings
    workload: WorkloadSettings
    runs: RunSettings
    stores: tuple[StoreSpec, ...]
    query_models: QueryModelSettings
    reference: ReferenceSettings

    @field_validator("stores")
    @classmethod
    def _names_are_unique(cls, stores: tuple[StoreSpec, ...]) -> tuple[StoreSpec, ...]:
        names = [store.name for store in stores]
        if len(set(names)) != len(names):
            raise ValueError("Two stores share a name.")
        return stores

    @model_validator(mode="after")
    def _ef_reaches_the_depth(self) -> SpikeSettings:
        for spec in self.stores:
            for ann in spec.sweep:
                if ann.ef is not None and ann.ef < self.search.depth:
                    raise ValueError(
                        f"{spec.name} sweeps an ef of {ann.ef}, below the depth of "
                        f"{self.search.depth}: LanceDB refuses an HNSW search for more "
                        "rows than ef."
                    )
        return self

    def store(self, name: str) -> StoreSpec:
        """The store called *name*.

        Raises:
            KeyError: If the settings name no such store.
        """
        for spec in self.stores:
            if spec.name == name:
                return spec
        raise KeyError(f"The settings name no store called {name!r}.")


def settings_file(name: str | None = None) -> Path:
    """The settings file called *name* beside ``spike.yaml``; ``spike.yaml`` itself without one.

    Raises:
        ValueError: If *name* is not the plain name of a YAML file.
    """
    if not name:
        return SPIKE_FILE
    if Path(name).name != name or not name.endswith(".yaml"):
        raise ValueError(f"A settings file is named as it lies beside spike.yaml, not {name!r}.")
    return SPIKE_FILE.with_name(name)


def load_spike_settings(path: Path | None = None) -> SpikeSettings:
    """Read and check the spike's settings.

    Args:
        path: The YAML file; without one, the file ``SPIKE_SETTINGS`` names
            beside ``spike.yaml``, or ``spike.yaml`` itself.

    Returns:
        The settings.
    """
    chosen = path or settings_file(os.environ.get(SETTINGS_ENV))
    return SpikeSettings.model_validate(yaml.safe_load(chosen.read_text(encoding="utf-8")))
