"""The spike's report: what was measured in each container, held against the budget.

A container serves a store alone, a query model alone, or a store with a query
model, the way the service would hold both. It is within the budget when three
things hold: the p95 of its measured requests, timed from outside the
container, is at most ``budget.p95_ms``; the slowest cold start is at most
``budget.cold_start_s``; and it served every request under its memory limit
without the kernel killing it. The memory limit is the budget's own figure, and
the cgroup's peak is reported beside it.

For a store alone, the measured requests are filtered hybrid searches with the
questions' synthetic vectors. For a query model alone, they are the same
questions embedded one at a time. For a store with a model, they are the same
searches, each embedding its question first.

The reference is the inference service's image started cold under the same
limits. Its factor is the cold start the repository records for that service on
Cloud Run divided by the median cold start measured here: one recorded figure,
so a rough scale rather than a measurement. It is a net figure as well. Cloud
Run runs the service under the same limits but with more CPU while it starts,
which makes its start faster than here, and adds what a container here never
waits for, such as placing the instance and fetching the image. The image sets
no thread settings, so the CPUs a process sees size its thread pools: under the
limits alone that is every CPU of Docker's virtual machine. The reference is
therefore also started pinned to one CPU, and each condition gets a factor.

Percentiles are numpy's default, linear between the two nearest requests.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from search_service.spike.cgroup import Usage
from search_service.spike.contracts import Timings
from search_service.spike.query_model import PreparedModel
from search_service.spike.settings import (
    Budget,
    ContainerSettings,
    Precision,
    ReferenceSettings,
    Runtime,
    SearchSettings,
    StoreSpec,
    VectorSettings,
    variant_name,
)

GIB = 1 << 30

#: The phase whose searches the budget's p95 is read from, with a store.
BUDGET_PHASE = "hybrid"

#: The phase whose embeddings the budget's p95 is read from, for a query model alone.
EMBED_PHASE = "embed"

_PARTS = ("embed_ms", "filter_ms", "lexical_ms", "vector_ms", "fusion_ms", "fetch_ms")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchRecord(_Frozen):
    """One request as it was measured.

    Attributes:
        run: The name of the run: a store, a query model, or both.
        phase: The phase it belongs to, such as ``hybrid`` or ``sweep:nprobes=20``.
        query_id: The question's id.
        question_set: The question's set.
        filter_kind: Where its filter came from.
        client_ms: The time from sending the request to the whole answer, outside the container.
        timings: The service's own times.
        allowed: How many passages the filter allowed, when the store counts them.
        hit_ids: The passages returned, best first.
        usage: The container's counters read right after.
    """

    run: str
    phase: str
    query_id: str
    question_set: str
    filter_kind: str
    client_ms: float
    timings: Timings
    allowed: int | None
    hit_ids: tuple[str, ...]
    usage: Usage


class Distribution(_Frozen):
    """Summary of a set of times, in milliseconds.

    Attributes:
        count: How many.
        p50: The median.
        p95: The 95th percentile.
        p99: The 99th percentile.
        max: The slowest.
        mean: The mean.
    """

    count: int
    p50: float
    p95: float
    p99: float
    max: float
    mean: float


def distribution(values: Sequence[float]) -> Distribution | None:
    """Summarise *values*, or return ``None`` when there are none."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return Distribution(
        count=len(array),
        p50=round(float(np.percentile(array, 50)), 2),
        p95=round(float(np.percentile(array, 95)), 2),
        p99=round(float(np.percentile(array, 99)), 2),
        max=round(float(array.max()), 2),
        mean=round(float(array.mean()), 2),
    )


class PhaseSummary(_Frozen):
    """The requests of one phase.

    Attributes:
        phase: Its name.
        searches: How many requests it holds.
        truncated: Whether it stopped at the time limit before every request was sent.
        client_ms: The times measured outside the container.
        server_ms: The service's own total times.
        parts_ms: The mean time of each part the service timed apart.
        by_filter: The times outside, by where the filter came from.
        by_set: The times outside, by question set.
        latencies_ms: Every time outside, in the order sent, to a tenth of a millisecond.
    """

    phase: str
    searches: int
    truncated: bool
    client_ms: Distribution | None
    server_ms: Distribution | None
    parts_ms: dict[str, float]
    by_filter: dict[str, Distribution]
    by_set: dict[str, Distribution]
    latencies_ms: tuple[float, ...]


def summarize_phase(phase: str, records: Sequence[SearchRecord], truncated: bool) -> PhaseSummary:
    """Summarise the records of one phase."""
    by_filter: dict[str, list[float]] = defaultdict(list)
    by_set: dict[str, list[float]] = defaultdict(list)
    parts: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_filter[record.filter_kind].append(record.client_ms)
        by_set[record.question_set].append(record.client_ms)
        for part in _PARTS:
            value = getattr(record.timings, part)
            if value is not None:
                parts[part].append(value)
    return PhaseSummary(
        phase=phase,
        searches=len(records),
        truncated=truncated,
        client_ms=distribution([record.client_ms for record in records]),
        server_ms=distribution([record.timings.total_ms for record in records]),
        parts_ms={part: round(float(np.mean(values)), 2) for part, values in parts.items()},
        by_filter={
            kind: summary
            for kind, values in sorted(by_filter.items())
            if (summary := distribution(values))
        },
        by_set={
            name: summary
            for name, values in sorted(by_set.items())
            if (summary := distribution(values))
        },
        latencies_ms=tuple(round(record.client_ms, 1) for record in records),
    )


class ColdStart(_Frozen):
    """One cold start, timed from just before the container was started.

    Attributes:
        start_s: How long ``docker start`` took to return.
        ready_s: Until ``/health`` first answered.
        first_search_ms: How long the first request took: a search, or an
            embedding for a query model alone.
        total_s: Until the first request was answered.
        process_age_s: How long the process had run when loading began.
        model_import_s: How long importing the query model's runtime took.
        model_load_s: How long loading the query model and its tokenizer took.
        model_first_s: How long embedding its first text took.
        model_s: The three together.
        open_s: How long opening the store took.
        throttled_periods: CPU periods in which the quota ran out, until the
            first request was answered.
        throttled_s: The time spent waiting for the next period, until then.
    """

    start_s: float
    ready_s: float
    first_search_ms: float
    total_s: float
    process_age_s: float | None
    model_import_s: float | None = None
    model_load_s: float | None = None
    model_first_s: float | None = None
    model_s: float | None = None
    open_s: float | None
    throttled_periods: int | None = None
    throttled_s: float | None = None


class ColdSummary(_Frozen):
    """The cold starts of one run.

    Attributes:
        runs: Each cold start, in order.
        median_s: The median time to the first request answered.
        max_s: The slowest.
    """

    runs: tuple[ColdStart, ...]
    median_s: float
    max_s: float


def summarize_cold(runs: Sequence[ColdStart]) -> ColdSummary | None:
    """Summarise the cold starts, or return ``None`` when there are none."""
    if not runs:
        return None
    totals = [run.total_s for run in runs]
    return ColdSummary(
        runs=tuple(runs),
        median_s=round(float(np.median(totals)), 3),
        max_s=round(max(totals), 3),
    )


class MemorySummary(_Frozen):
    """What the warm container used.

    Attributes:
        peak_bytes: The cgroup's peak memory, page cache included.
        anon_max_bytes: The most of the processes' own memory read after a request.
        file_max_bytes: The most page cache read after a request.
        rss_peak_bytes: The process's highest resident memory.
        refaults: Pages of files read again after eviction, during the warm phases.
        major_faults: Page faults that read from disk, during the warm phases.
        oom_kills: Processes the kernel killed for want of memory.
        throttled_periods: CPU periods in which the quota ran out, during the warm phases.
        throttled_s: The time spent waiting for the next period, during the warm phases.
    """

    peak_bytes: int | None
    anon_max_bytes: int | None
    file_max_bytes: int | None
    rss_peak_bytes: int | None
    refaults: int | None
    major_faults: int | None
    oom_kills: int | None
    throttled_periods: int | None
    throttled_s: float | None


def _most(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return max(known) if known else None


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def summarize_memory(before: Usage, readings: Sequence[Usage], after: Usage) -> MemorySummary:
    """Summarise the readings taken during the warm phases.

    Args:
        before: The reading taken when the warm phases began.
        readings: The readings after each request.
        after: The reading taken when they ended.

    Returns:
        The summary.
    """
    every = [before, *readings, after]
    throttled_us = _delta(before.throttled_us, after.throttled_us)
    return MemorySummary(
        peak_bytes=_most(usage.memory_peak for usage in every),
        anon_max_bytes=_most(usage.anon for usage in every),
        file_max_bytes=_most(usage.file for usage in every),
        rss_peak_bytes=_most(usage.rss_peak for usage in every),
        refaults=_delta(before.refaults, after.refaults),
        major_faults=_delta(before.major_faults, after.major_faults),
        oom_kills=after.oom_kills,
        throttled_periods=_delta(before.throttled, after.throttled),
        throttled_s=None if throttled_us is None else round(throttled_us / 1e6, 3),
    )


class Verdict(_Frozen):
    """A run held against the budget.

    Attributes:
        p95_ms: The p95 of the budget's phase, outside the container.
        cold_start_s: The slowest cold start.
        memory_peak_gib: The cgroup's peak memory, in GiB.
        survived: Whether the container served every request without being killed.
        within_p95: Whether the p95 is within the budget.
        within_cold_start: Whether the slowest cold start is.
        within_memory: Whether the run stayed within the memory limit.
        within_budget: Whether all three hold.
    """

    p95_ms: float | None
    cold_start_s: float | None
    memory_peak_gib: float | None
    survived: bool
    within_p95: bool
    within_cold_start: bool
    within_memory: bool
    within_budget: bool


def verdict(
    budget: Budget,
    phase: PhaseSummary | None,
    cold: ColdSummary | None,
    memory: MemorySummary | None,
    survived: bool,
) -> Verdict:
    """Hold one run, or one point of a sweep, against the budget.

    A phase that stopped at its time limit counts as over the budget, since the
    requests it did not reach could not be timed.
    """
    p95 = phase.client_ms.p95 if phase is not None and phase.client_ms is not None else None
    peak = memory.peak_bytes if memory is not None else None
    killed = bool(memory is not None and memory.oom_kills)
    within_p95 = bool(
        survived and p95 is not None and p95 <= budget.p95_ms and not (phase and phase.truncated)
    )
    within_cold = bool(cold is not None and cold.max_s <= budget.cold_start_s)
    within_memory = bool(
        survived and not killed and peak is not None and peak <= budget.memory_gib * GIB
    )
    return Verdict(
        p95_ms=p95,
        cold_start_s=cold.max_s if cold is not None else None,
        memory_peak_gib=None if peak is None else round(peak / GIB, 3),
        survived=survived,
        within_p95=within_p95,
        within_cold_start=within_cold,
        within_memory=within_memory,
        within_budget=within_p95 and within_cold and within_memory,
    )


class ModelLabel(_Frozen):
    """A query model in one runtime and one precision.

    Attributes:
        name: The model's short name.
        runtime: The runtime that runs it.
        precision: The precision of its weights.
        dim: The dimension of its vectors.
    """

    name: str
    runtime: Runtime
    precision: Precision
    dim: int

    @property
    def label(self) -> str:
        """The model with its runtime and precision, as runs are named."""
        return variant_name(self.name, self.runtime, self.precision)


class RunResult(_Frozen):
    """Everything measured in one kind of container.

    Attributes:
        name: The run's name: the store's, the model's with its runtime and
            precision, or both joined by ``+``.
        image: The image the container ran.
        store: The store, or ``None`` for a query model alone.
        model: The query model, or ``None`` for a store alone.
        store_bytes: The size on disk of what the store reads.
        prepared: The prepared query model: its size on disk, and how its
            vectors agreed with sentence-transformers in float32.
        cold: Its cold starts.
        phases: Its phases, in the order run.
        memory: What the warm container used.
        failure: What went wrong, when the measurement could not finish.
        verdict: The run held against the budget, at LanceDB's default search parameters.
        sweep_verdicts: Each point of a sweep held against the budget, by its label.
        measured_at: When its first cold start began, as an ISO timestamp.
    """

    name: str
    image: str
    store: StoreSpec | None
    model: ModelLabel | None
    store_bytes: int | None
    prepared: PreparedModel | None
    cold: ColdSummary | None
    phases: tuple[PhaseSummary, ...]
    memory: MemorySummary | None
    failure: str | None
    verdict: Verdict
    sweep_verdicts: dict[str, Verdict]
    measured_at: str | None = None


class SkippedPair(_Frozen):
    """A store and a query model of the same dimension that were not measured together.

    Attributes:
        store: The store.
        model: The query model with its precision.
        reason: Why: which of them was over the budget alone, or was not there.
    """

    store: str
    model: str
    reason: str


class Overlap(_Frozen):
    """How far one run's passages agree with a reference run's, query by query.

    Attributes:
        store: The run.
        reference: The run it is compared with.
        phase: The phase both ran.
        queries: How many queries both answered.
        mean_shared_at_10: The mean share of the reference's ten passages the run also returned.
    """

    store: str
    reference: str
    phase: str
    queries: int
    mean_shared_at_10: float


def overlap(
    store: Sequence[SearchRecord], reference: Sequence[SearchRecord], phase: str
) -> Overlap | None:
    """Compare two runs' answers to the same queries in one phase."""
    theirs = {record.query_id: record.hit_ids[:10] for record in reference if record.phase == phase}
    shares = []
    for record in store:
        if record.phase != phase or record.query_id not in theirs:
            continue
        expected = theirs[record.query_id]
        if expected:
            shares.append(len(set(record.hit_ids[:10]) & set(expected)) / len(expected))
    if not shares or not store:
        return None
    return Overlap(
        store=store[0].run,
        reference=reference[0].run,
        phase=phase,
        queries=len(shares),
        mean_shared_at_10=round(float(np.mean(shares)), 4),
    )


class ReferenceColdStart(_Frozen):
    """One cold start of the inference service's image, timed from just before it was started.

    Attributes:
        start_s: How long ``docker start`` took to return.
        ready_s: Until ``/health`` first answered.
        first_predict_ms: How long the first prediction took.
        total_s: Until the first prediction was answered.
        throttled_periods: CPU periods in which the quota ran out, until then.
        throttled_s: The time spent waiting for the next period, until then.
    """

    start_s: float
    ready_s: float
    first_predict_ms: float
    total_s: float
    throttled_periods: int | None = None
    throttled_s: float | None = None


class ReferenceResult(_Frozen):
    """The inference service's image, started cold under the limits of the spike's containers.

    Attributes:
        image: The image.
        image_bytes: Its size.
        model: The classifier it serves.
        revision: The revision its weights were saved from.
        pinned_cpu: The CPU the container was pinned to beside the limits, or
            ``None`` under the limits alone.
        cpus_seen: How many CPUs a process in the container sees.
        threads: The most threads one process in the container ran, after the
            warm predictions.
        cold: Each cold start, in order.
        median_s: The median time to the first prediction answered.
        max_s: The slowest.
        warm_ms: The predictions after the last cold start, timed from outside.
        warm_throttled_periods: CPU periods in which the quota ran out, during them.
        warm_throttled_s: The time spent waiting for the next period, during them.
        memory_peak_bytes: The cgroup's peak memory after them.
        cloud_run_cold_s: The cold start on Cloud Run the repository records.
        cloud_run_warm_s: The warm answer it records beside it.
        cloud_run_source: The file that records both.
        cloud_run_cpu: The CPUs the service has on Cloud Run.
        cloud_run_memory: Its memory there.
        cloud_run_startup_cpu_boost: Whether Cloud Run gives it more CPU while it starts.
        factor: ``cloud_run_cold_s`` divided by ``median_s``.
        measured_at: When its first cold start began, as an ISO timestamp.
        failure: What went wrong, when the measurement could not finish.
    """

    image: str
    image_bytes: int | None
    model: str
    revision: str
    pinned_cpu: str | None = None
    cpus_seen: int | None
    threads: int | None = None
    cold: tuple[ReferenceColdStart, ...]
    median_s: float | None
    max_s: float | None
    warm_ms: Distribution | None
    warm_throttled_periods: int | None = None
    warm_throttled_s: float | None = None
    memory_peak_bytes: int | None
    cloud_run_cold_s: float
    cloud_run_warm_s: float
    cloud_run_source: str
    cloud_run_cpu: str
    cloud_run_memory: str
    cloud_run_startup_cpu_boost: bool
    factor: float | None
    measured_at: str | None = None
    failure: str | None


def summarize_reference(
    settings: ReferenceSettings,
    cold: Sequence[ReferenceColdStart],
    warm_ms: Sequence[float],
    *,
    pinned_cpu: str | None = None,
    image_bytes: int | None,
    cpus_seen: int | None,
    threads: int | None = None,
    warm_throttled_periods: int | None = None,
    warm_throttled_s: float | None = None,
    memory_peak_bytes: int | None,
    measured_at: str | None = None,
    failure: str | None,
) -> ReferenceResult:
    """Summarise the reference's cold starts and put the recorded Cloud Run figure beside them."""
    totals = [run.total_s for run in cold]
    median = round(float(np.median(totals)), 3) if totals else None
    return ReferenceResult(
        image=settings.image,
        image_bytes=image_bytes,
        model=settings.model,
        revision=settings.revision,
        pinned_cpu=pinned_cpu,
        cpus_seen=cpus_seen,
        threads=threads,
        cold=tuple(cold),
        median_s=median,
        max_s=round(max(totals), 3) if totals else None,
        warm_ms=distribution(warm_ms),
        warm_throttled_periods=warm_throttled_periods,
        warm_throttled_s=warm_throttled_s,
        memory_peak_bytes=memory_peak_bytes,
        cloud_run_cold_s=settings.cloud_run_cold_s,
        cloud_run_warm_s=settings.cloud_run_warm_s,
        cloud_run_source=settings.cloud_run_source,
        cloud_run_cpu=settings.cloud_run_cpu,
        cloud_run_memory=settings.cloud_run_memory,
        cloud_run_startup_cpu_boost=settings.cloud_run_startup_cpu_boost,
        factor=round(settings.cloud_run_cold_s / median, 2) if median else None,
        measured_at=measured_at,
        failure=failure,
    )


class SupersededRun(_Frozen):
    """A run that was measured again, kept with the container it first ran in.

    Attributes:
        run: The first measurement.
        container: The container's settings when it was made.
        reason: Why the run was measured again.
    """

    run: RunResult
    container: ContainerSettings
    reason: str


class CloudRunEstimate(_Frozen):
    """One run's slowest cold start here, times one factor of the reference.

    The budget holds the cold start measured here, as the pre-registration
    words the rule, so an estimate decides nothing. It marks a run whose cold
    start could pass the budget on Cloud Run.

    Attributes:
        run: The run's name.
        pinned_cpu: The CPU of the reference's condition whose factor is used,
            or ``None`` for the limits alone.
        factor: That condition's factor.
        cold_start_s: The run's slowest cold start here.
        scaled_s: ``cold_start_s`` times ``factor``.
        over_budget: Whether ``scaled_s`` passes the budget's cold start.
    """

    run: str
    pinned_cpu: str | None
    factor: float
    cold_start_s: float
    scaled_s: float
    over_budget: bool


def cloud_run_estimates(
    budget: Budget, runs: Sequence[RunResult], references: Sequence[ReferenceResult]
) -> list[CloudRunEstimate]:
    """Scale every run's slowest cold start by every factor of the reference."""
    estimates = []
    for reference in references:
        if reference.factor is None:
            continue
        for run in runs:
            if run.cold is None:
                continue
            scaled = round(run.cold.max_s * reference.factor, 2)
            estimates.append(
                CloudRunEstimate(
                    run=run.name,
                    pinned_cpu=reference.pinned_cpu,
                    factor=reference.factor,
                    cold_start_s=run.cold.max_s,
                    scaled_s=scaled,
                    over_budget=scaled > budget.cold_start_s,
                )
            )
    return estimates


class SpikeReport(_Frozen):
    """The spike's report.

    The report is written after every run, so that a spike that stops still
    leaves what it measured; ``finished`` says whether every run was measured.

    Attributes:
        created: When the report was written, as an ISO timestamp.
        finished: Whether every run and the reference were measured.
        budget: The bounds of the storage rule.
        container: The limits every measured container ran under.
        search: The shape of one search.
        vectors: How the synthetic vectors were drawn.
        workload: How many searches of each kind were sent.
        environment: The machine, Docker and the libraries in each image.
        images: The size of each image, without any store or model, by its name.
        runs: Every run: the stores alone, the query models alone, and the pairs.
        unprepared: The query models that could not be prepared, and why, by variant.
        skipped: The pairs not measured, and why.
        overlaps: How far the stores' answers agree with each dimension's exact float32 numpy store.
        references: The inference service's image under the limits alone and,
            when it was measured so, pinned to one CPU as well.
        cloud_run_estimates: Every run's slowest cold start scaled by every
            factor of the reference: a risk for Cloud Run, never a verdict.
        superseded: Runs measured again, each kept with its first container and why.
    """

    created: str
    finished: bool
    budget: Budget
    container: ContainerSettings
    search: SearchSettings
    vectors: VectorSettings
    workload: dict[str, dict[str, int]]
    environment: dict[str, str]
    images: dict[str, int | None]
    runs: tuple[RunResult, ...]
    unprepared: dict[str, str]
    skipped: tuple[SkippedPair, ...]
    overlaps: tuple[Overlap, ...]
    references: tuple[ReferenceResult, ...]
    cloud_run_estimates: tuple[CloudRunEstimate, ...] = ()
    superseded: tuple[SupersededRun, ...] = ()
