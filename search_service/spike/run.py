"""Measure the stores and query models of the storage spike, each in a container with one CPU.

Runs on the host and drives Docker:

1. ``--image`` builds the spike's two images from a staged copy of ``search/``
   and ``search_service/``: the slim one without torch, the full one with it.
   With ``--reference`` it also builds the reference image, the inference
   service's image with its weights from the Hugging Face cache.
2. ``--build`` builds the stores into the Docker volume, and prepares the query
   models found in the Hugging Face cache, in the full image and a container
   without limits. Each model is checked there on questions of the workload.
3. Three kinds of run follow, each in containers with the limits of
   ``spike.yaml``: every store alone; every prepared query model alone, in each
   runtime and precision; and every store with every query model of its
   dimension, where both were within the budget alone. A store with a model is
   what a service with a local model would hold: the model embeds each
   question, and the store searches with that vector. A run with a model run by
   torch uses the full image; every other run uses the slim one.
4. A run gets ``runs.cold_starts`` cold starts, each in a new container after
   the page cache of Docker's virtual machine is dropped, timed from just before
   the container is started to the first request answered. The last container
   then gets the warm-up and the measured phases, and the container's counters
   are read after every request. A store alone also gets each leg alone, each
   point of a sweep and the view of the eight parties; a store with a model gets
   the searches and the view of the eight parties; a model alone embeds the
   same questions.
5. With ``--reference``, the reference image is started cold the same way and
   sent the keep-warm workflow's request, which puts a factor between the cold
   starts here and the one the repository records on Cloud Run. It is started
   under the limits alone, then pinned to one CPU as well when the settings name one.
6. The summary goes to ``storage_spike.json`` among the tracked reports, and
   every request's record to ``spike/searches.jsonl`` in the search-data
   directory. Both are written after every run, so a spike that stops leaves
   what it measured, and ``--resume`` measures only what that report lacks.
   ``--remeasure`` measures runs of that report again, and the report keeps
   their first measurement with the reason. ``--memory`` measures under
   another memory limit, into a report and records named after it, and
   ``--rewrite`` adds to a report what a later version derives from it.
   ``--settings`` reads another settings file beside ``spike.yaml``, here and in
   every container, into a report and records named after it.

The containers run without swap, so memory beyond the limit cannot be hidden on disk.

Usage::

    python -m search_service.spike.run [--data-root DIR] [--settings FILE] [--image] [--build]
        [--reference] [--only NAME ...] [--skip-models] [--memory 4g]
        [--hf-cache DIR] [--resume [--remeasure NAME ... --reason TEXT]]
        [--rewrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import httpx
from pydantic import TypeAdapter

from search.config import load_settings
from search.corpus.store import write_report
from search_service.spike.cgroup import Usage, parse_flat_keyed
from search_service.spike.contracts import (
    EmbedRequest,
    EmbedResponse,
    Mode,
    PartyViewRequest,
    PartyViewResponse,
    SearchRequest,
    SearchResponse,
    Timings,
)
from search_service.spike.inputs import TUNING_NAME, read_passage_columns
from search_service.spike.query_model import (
    MODELS_DIR_NAME,
    PREPARATION_NAME,
    Preparation,
    PreparedModel,
)
from search_service.spike.report import (
    BUDGET_PHASE,
    EMBED_PHASE,
    ColdStart,
    ModelLabel,
    Overlap,
    PhaseSummary,
    ReferenceColdStart,
    ReferenceResult,
    RunResult,
    SearchRecord,
    SkippedPair,
    SpikeReport,
    SupersededRun,
    cloud_run_estimates,
    overlap,
    summarize_cold,
    summarize_memory,
    summarize_phase,
    summarize_reference,
    verdict,
)
from search_service.spike.settings import (
    SETTINGS_ENV,
    AnnParams,
    ReferenceSettings,
    SpikeSettings,
    StoreSpec,
    load_spike_settings,
    settings_file,
    variant_name,
)
from search_service.spike.workload import Workload, WorkloadQuery, build_workload

logger = logging.getLogger("spike.run")

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
REPORT_NAME = "storage_spike.json"
SEARCHES_NAME = "searches.jsonl"
SPIKE_DATA_DIR = "spike"
CONTAINER_PORT = 8000
HF_MOUNT = "/hf"
CHECK_MOUNT = "/check"
CHECK_TEXTS_NAME = "check_texts.json"
REFERENCE_DOCKERFILE = "reference.Dockerfile"
REFERENCE_MODEL_DIR = "model-src"
_PREPARATIONS = TypeAdapter(list[Preparation])


class DockerError(RuntimeError):
    """A Docker command failed."""


def _now() -> str:
    """The time now, in UTC, as an ISO timestamp to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seconds(microseconds: int | None) -> float | None:
    """Microseconds as seconds, to the millisecond, or ``None``."""
    return None if microseconds is None else round(microseconds / 1e6, 3)


@dataclass(frozen=True)
class DockerCli:
    """How Docker is reached from this machine.

    Attributes:
        command: The command that runs Docker's client, such as ``docker``, or
            ``wsl -d Ubuntu -u root -- docker`` for an engine inside WSL.
        wsl: Whether the engine runs inside WSL, which names the drive ``C:`` as ``/mnt/c``.
    """

    command: tuple[str, ...] = ("docker",)
    wsl: bool = False

    def path(self, path: Path | str) -> str:
        """A path on this machine as the engine names it."""
        if not self.wsl:
            return str(path)
        windows = PureWindowsPath(str(path))
        drive = windows.drive.rstrip(":").lower()
        return f"/mnt/{drive}" + windows.as_posix()[len(windows.drive) :]


#: The Docker client in use; ``main`` replaces it when the command line names another.
CLI = DockerCli()


def docker(*args: str, check: bool = True) -> str:
    """Run one Docker command and return what it printed.

    Raises:
        DockerError: If the command fails and *check* is true.
    """
    completed = subprocess.run(
        [*CLI.command, *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and completed.returncode != 0:
        raise DockerError(f"docker {' '.join(args[:2])} failed: {completed.stderr.strip()[-2000:]}")
    return completed.stdout.strip()


def stage_context(target: Path) -> Path:
    """Copy what the spike's images need into *target*, the build context, and return it."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO_ROOT / "search", target / "search", ignore=ignore)
    shutil.copytree(REPO_ROOT / "search_service", target / "search_service", ignore=ignore)
    shutil.copyfile(REPO_ROOT / "search_service" / "spike" / "Dockerfile", target / "Dockerfile")
    return target


def build_images(settings: SpikeSettings) -> None:
    """Build the spike's two images, each a target of the same Dockerfile."""
    with tempfile.TemporaryDirectory() as staging:
        context = stage_context(Path(staging))
        for target, image in (
            ("slim", settings.runs.images.slim),
            ("full", settings.runs.images.full),
        ):
            logger.info("Building the image %s", image)
            docker("build", "--target", target, "-t", image, CLI.path(context))


def hub_snapshot(hf_cache: Path, model: str, revision: str) -> Path:
    """The directory of one revision of a model in the Hugging Face hub cache."""
    return hf_cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision


def stage_reference_context(target: Path, settings: SpikeSettings, hf_cache: Path) -> Path:
    """Copy what the reference image needs into *target*, and return it.

    That is the inference service's requirements and app, the classifier's
    files at the pinned revision, copied out of the cache with every link
    followed, and the reference Dockerfile.

    Raises:
        FileNotFoundError: If the cache does not hold the classifier at that revision.
    """
    reference = settings.reference
    snapshot = hub_snapshot(hf_cache, reference.model, reference.revision)
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"The Hugging Face cache holds no {reference.model} at {reference.revision}."
        )
    shutil.copytree(snapshot, target / REFERENCE_MODEL_DIR, symlinks=False)
    service = target / "inference_service"
    service.mkdir(parents=True)
    for name in ("requirements.txt", "app.py"):
        shutil.copyfile(REPO_ROOT / "inference_service" / name, service / name)
    dockerfile = REPO_ROOT / "search_service" / "spike" / REFERENCE_DOCKERFILE
    shutil.copyfile(dockerfile, target / "Dockerfile")
    return target


def build_reference_image(settings: SpikeSettings, hf_cache: Path) -> None:
    """Build the reference image from the inference service and the cached classifier."""
    with tempfile.TemporaryDirectory() as staging:
        context = stage_reference_context(Path(staging), settings, hf_cache)
        logger.info("Building the image %s", settings.reference.image)
        docker("build", "-t", settings.reference.image, CLI.path(context))


def default_hf_cache() -> Path:
    """The Hugging Face hub cache of this machine, as the library finds it."""
    home = os.environ.get("HF_HOME")
    root = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return root / "hub"


def check_texts(workload: Workload, count: int) -> list[str]:
    """The questions each query model is checked on: the warm-up's first, then the measured ones."""
    texts: list[str] = []
    for query in [*workload.warmup, *workload.queries]:
        if query.text not in texts:
            texts.append(query.text)
        if len(texts) == count:
            break
    return texts


def build_stores(
    settings: SpikeSettings,
    search_dir: Path,
    reports_dir: Path,
    hf_cache: Path,
    only: Sequence[str] | None,
    texts: Sequence[str],
) -> None:
    """Build the stores and prepare the query models in the volume, in a container without limits.

    The Hugging Face cache is mounted read-only and the container is told to
    stay offline, so a model the cache lacks is reported, never downloaded.
    *texts* are the questions each model is checked on.
    """
    docker("volume", "create", settings.runs.volume)
    with tempfile.TemporaryDirectory() as staging:
        (Path(staging) / CHECK_TEXTS_NAME).write_text(
            json.dumps(list(texts), ensure_ascii=False), encoding="utf-8"
        )
        command = [
            "run", "--rm",
            "-v", f"{CLI.path(search_dir)}:/data/search:ro",
            "-v", f"{CLI.path(reports_dir)}:/data/reports:ro",
            "-v", f"{CLI.path(hf_cache)}:{HF_MOUNT}:ro",
            "-v", f"{CLI.path(staging)}:{CHECK_MOUNT}:ro",
            "-v", f"{settings.runs.volume}:/stores",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1",
            *settings_env(),
            settings.runs.images.full,
            "python", "-m", "search_service.spike.build",
            "--search-dir", "/data/search",
            "--tuning", f"/data/reports/{TUNING_NAME}",
            "--out", "/stores",
            "--models", "--hf-cache", HF_MOUNT,
            "--check-texts", f"{CHECK_MOUNT}/{CHECK_TEXTS_NAME}",
        ]  # fmt: skip
        if only:
            command += ["--only", *only]
        logger.info("Building the stores into the volume %s", settings.runs.volume)
        subprocess.run([*CLI.command, *command], check=True)


def drop_page_cache(settings: SpikeSettings) -> None:
    """Drop the page cache of Docker's virtual machine, so that the next start reads from disk."""
    docker(
        "run", "--rm", "--privileged", settings.runs.images.slim,
        "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches",
    )  # fmt: skip


@dataclass(frozen=True)
class Subject:
    """What one kind of container serves: a store, a query model, or both.

    Attributes:
        store: The store, or ``None``.
        model: The query model in one runtime and precision, or ``None``.
    """

    store: StoreSpec | None
    model: ModelLabel | None

    @property
    def name(self) -> str:
        """The run's name: the store's, the model's, or both joined by ``+``."""
        parts = [self.store.name if self.store else "", self.model.label if self.model else ""]
        return "+".join(part for part in parts if part)


def image_for(settings: SpikeSettings, subject: Subject) -> str:
    """The image a run uses: the full one for a model run by torch, the slim one otherwise."""
    if subject.model is not None and subject.model.runtime == "torch":
        return settings.runs.images.full
    return settings.runs.images.slim


def settings_env() -> list[str]:
    """The flag that gives a container this run's settings file, when it is not ``spike.yaml``."""
    name = os.environ.get(SETTINGS_ENV)
    return ["-e", f"{SETTINGS_ENV}={name}"] if name else []


def create_container(settings: SpikeSettings, subject: Subject) -> str:
    """Create, without starting, the container that serves *subject*; return its id."""
    limits = settings.container
    command = [
        "create",
        "--cpus", limits.cpus,
        "--memory", limits.memory,
        "--memory-swap", limits.memory,
        "-p", f"127.0.0.1:{settings.runs.port}:{CONTAINER_PORT}",
        "-v", f"{settings.runs.volume}:/stores:ro",
        "-e", "SPIKE_STORES=/stores",
        "-e", "TOKENIZERS_PARALLELISM=false",
        *settings_env(),
    ]  # fmt: skip
    if subject.store is not None:
        command += ["-e", f"SPIKE_STORE={subject.store.name}"]
    if subject.model is not None:
        command += ["-e", f"SPIKE_MODEL={subject.model.name}"]
        command += ["-e", f"SPIKE_RUNTIME={subject.model.runtime}"]
        command += ["-e", f"SPIKE_PRECISION={subject.model.precision}"]
    for name, value in sorted(limits.threads.items()):
        command += ["-e", f"{name}={value}"]
    return docker(*command, image_for(settings, subject))


def container_state(container: str) -> str:
    """The container's status, whether it was killed for memory, and its exit code."""
    return docker(
        "inspect", "-f", "{{.State.Status}} oom={{.State.OOMKilled}} exit={{.State.ExitCode}}",
        container, check=False,
    )  # fmt: skip


class Client:
    """Requests to the measured container, timed from outside it."""

    def __init__(self, port: int, timeout: float) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def wait_ready(self, timeout: float) -> dict[str, Any]:
        """Poll ``/health`` until it answers.

        Raises:
            TimeoutError: If it does not answer within *timeout* seconds.
        """
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                response = self.http.get(f"{self.base}/health", timeout=1.0)
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                pass
            time.sleep(0.02)
        raise TimeoutError(f"The service did not answer within {timeout} s.")

    def post(self, path: str, payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """One request, and the time it took from outside, in milliseconds."""
        started = time.perf_counter()
        response = self.http.post(f"{self.base}{path}", json=payload)
        elapsed = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return elapsed, response.json()

    def search(self, request: SearchRequest) -> tuple[float, SearchResponse]:
        """One search, and the time it took from outside."""
        elapsed, body = self.post("/search", request.model_dump(mode="json"))
        return elapsed, SearchResponse.model_validate(body)

    def parties(self, request: PartyViewRequest) -> tuple[float, PartyViewResponse]:
        """The view of the eight parties, and the time it took from outside."""
        elapsed, body = self.post("/parties", request.model_dump(mode="json"))
        return elapsed, PartyViewResponse.model_validate(body)

    def embed(self, request: EmbedRequest) -> tuple[float, EmbedResponse]:
        """One question embedded by the query model alone, and the time it took from outside."""
        elapsed, body = self.post("/embed", request.model_dump(mode="json"))
        return elapsed, EmbedResponse.model_validate(body)

    def usage(self) -> Usage:
        """The container's counters."""
        return Usage.model_validate(self.http.get(f"{self.base}/usage").json())


def search_request(
    query: WorkloadQuery, mode: Mode = "hybrid", ann: AnnParams | None = None
) -> SearchRequest:
    """The request of one workload query."""
    return SearchRequest(
        text=query.text,
        vector_seed=query.vector_seed,
        filter=query.filter,
        mode=mode,
        ann=ann or AnnParams(),
    )


def _embedded(response: EmbedResponse) -> SearchResponse:
    """An embedding's time, in the shape a record holds."""
    timings = Timings(total_ms=response.embed_ms, embed_ms=response.embed_ms)
    return SearchResponse(hits=(), allowed=None, timings=timings)


class ServiceRun:
    """The measurement of one kind of container."""

    def __init__(self, settings: SpikeSettings, subject: Subject, workload: Workload) -> None:
        self.settings = settings
        self.subject = subject
        self.workload = workload
        self.records: list[SearchRecord] = []
        self.client = Client(settings.runs.port, settings.runs.request_timeout_s)

    def _record(
        self, phase: str, query: WorkloadQuery, elapsed: float, response: SearchResponse
    ) -> SearchRecord:
        record = SearchRecord(
            run=self.subject.name,
            phase=phase,
            query_id=query.query_id,
            question_set=query.question_set,
            filter_kind=query.filter_kind,
            client_ms=round(elapsed, 3),
            timings=response.timings,
            allowed=response.allowed,
            hit_ids=tuple(hit.passage_id for hit in response.hits),
            usage=self.client.usage(),
        )
        self.records.append(record)
        return record

    def _request(self, phase: str, query: WorkloadQuery, mode: Mode, ann: AnnParams | None) -> None:
        """Send one query the way this kind of container takes it, and record it."""
        if self.subject.store is None:
            elapsed, embedded = self.client.embed(EmbedRequest(text=query.text))
            self._record(phase, query, elapsed, _embedded(embedded))
        else:
            elapsed, response = self.client.search(search_request(query, mode, ann))
            self._record(phase, query, elapsed, response)

    def cold_start(self) -> tuple[str, ColdStart]:
        """Start a new container after dropping the page cache, and time it to the first request."""
        drop_page_cache(self.settings)
        container = create_container(self.settings, self.subject)
        started = time.perf_counter()
        docker("start", container)
        start_s = time.perf_counter() - started
        health = self.client.wait_ready(self.settings.runs.ready_timeout_s)
        ready_s = time.perf_counter() - started
        first = self.workload.queries[0]
        self._request("cold", first, "hybrid", None)
        total_s = time.perf_counter() - started
        usage = self.records[-1].usage
        return container, ColdStart(
            start_s=round(start_s, 3),
            ready_s=round(ready_s, 3),
            first_search_ms=round(self.records[-1].client_ms, 1),
            total_s=round(total_s, 3),
            process_age_s=health.get("process_age_s"),
            model_import_s=health.get("model_import_s"),
            model_load_s=health.get("model_load_s"),
            model_first_s=health.get("model_first_s"),
            model_s=health.get("model_s"),
            open_s=health.get("open_s"),
            throttled_periods=usage.throttled,
            throttled_s=_seconds(usage.throttled_us),
        )

    def phase(
        self, name: str, queries: Sequence[WorkloadQuery], mode: Mode, ann: AnnParams | None
    ) -> PhaseSummary:
        """Send *queries* one at a time and summarise them, stopping at the phase's time limit."""
        started = time.perf_counter()
        count = len(self.records)
        truncated = False
        for query in queries:
            if time.perf_counter() - started > self.settings.runs.phase_limit_s:
                truncated = True
                break
            self._request(name, query, mode, ann)
        logger.info("  %s: %d requests", name, len(self.records) - count)
        return summarize_phase(name, self.records[count:], truncated)

    def party_phase(self) -> PhaseSummary:
        """The view of the eight parties, one request per question."""
        records = []
        started = time.perf_counter()
        truncated = False
        for query in self.workload.party_view:
            if time.perf_counter() - started > self.settings.runs.phase_limit_s:
                truncated = True
                break
            request = PartyViewRequest(text=query.text, vector_seed=query.vector_seed)
            elapsed, response = self.client.parties(request)
            hits = tuple(hit for hit in response.hits.values() if hit is not None)
            answer = SearchResponse(hits=hits, allowed=None, timings=response.timings)
            records.append(self._record("parties", query, elapsed, answer))
        logger.info("  parties: %d views", len(records))
        return summarize_phase("parties", records, truncated)

    def _phases(self) -> list[PhaseSummary]:
        """The measured phases, by what the container serves."""
        runs = self.settings.runs
        queries = self.workload.queries
        if self.subject.store is None:
            return [self.phase(EMBED_PHASE, queries, "hybrid", None)]
        phases = [self.phase(BUDGET_PHASE, queries, "hybrid", None)]
        if self.subject.model is None:
            legs = queries[: runs.leg_queries]
            phases.append(self.phase("lexical", legs, "lexical", None))
            phases.append(self.phase("vector", legs, "vector", None))
            for ann in self.subject.store.sweep:
                sweep = queries[: runs.sweep_queries]
                phases.append(self.phase(f"sweep:{ann.label}", sweep, "hybrid", ann))
        phases.append(self.party_phase())
        return phases

    def run(self) -> RunResult:
        """Measure the container; a failure ends the measurement and is reported."""
        runs = self.settings.runs
        measured_at = _now()
        cold_runs: list[ColdStart] = []
        phases: list[PhaseSummary] = []
        memory = None
        failure = None
        survived = False
        container = None
        try:
            for number in range(runs.cold_starts):
                container, cold = self.cold_start()
                cold_runs.append(cold)
                logger.info("  cold start %d: %.2f s", number + 1, cold.total_s)
                if number < runs.cold_starts - 1:
                    docker("rm", "-f", container)
            before = self.client.usage()
            for query in self.workload.warmup:
                self._request("warmup", query, "hybrid", None)
            phases = self._phases()
            after = self.client.usage()
            readings = [record.usage for record in self.records if record.phase != "cold"]
            memory = summarize_memory(before, readings, after)
            survived = True
        except (httpx.HTTPError, DockerError, TimeoutError) as error:
            state = container_state(container) if container else "no container"
            failure = f"{type(error).__name__}: {error}; container: {state}"
            logger.error("  failed: %s", failure)
        finally:
            self.client.close()
            if container:
                docker("rm", "-f", container, check=False)
        by_phase = {phase.phase: phase for phase in phases}
        cold = summarize_cold(cold_runs)
        main_phase = EMBED_PHASE if self.subject.store is None else BUDGET_PHASE
        budget = self.settings.budget
        sweeps = {}
        if self.subject.store is not None and self.subject.model is None:
            sweeps = {
                ann.label: verdict(
                    budget, by_phase.get(f"sweep:{ann.label}"), cold, memory, survived
                )
                for ann in self.subject.store.sweep
            }
        return RunResult(
            name=self.subject.name,
            image=image_for(self.settings, self.subject),
            store=self.subject.store,
            model=self.subject.model,
            store_bytes=None,
            prepared=None,
            cold=cold,
            phases=tuple(phases),
            memory=memory,
            failure=failure,
            verdict=verdict(budget, by_phase.get(main_phase), cold, memory, survived),
            sweep_verdicts=sweeps,
            measured_at=measured_at,
        )


def reference_container(settings: SpikeSettings, pinned_cpu: str | None = None) -> str:
    """Create, without starting, a container of the reference image; return its id.

    It gets the spike's limits but none of its thread settings, since the
    inference service's image sets none. With *pinned_cpu* it is also pinned
    to that one CPU of Docker's virtual machine.
    """
    limits = settings.container
    pinned = [] if pinned_cpu is None else ["--cpuset-cpus", pinned_cpu]
    return docker(
        "create",
        "--cpus", limits.cpus,
        *pinned,
        "--memory", limits.memory,
        "--memory-swap", limits.memory,
        "-p", f"127.0.0.1:{settings.runs.port}:{settings.reference.port}",
        settings.reference.image,
    )  # fmt: skip


def reference_conditions(reference: ReferenceSettings) -> list[str | None]:
    """The CPU of each of the reference's conditions: none, then the one the settings name."""
    return [None] if reference.pinned_cpu is None else [None, reference.pinned_cpu]


def _container_number(container: str, *command: str) -> int | None:
    """A number printed by a command run inside *container*, or ``None``."""
    printed = docker("exec", container, *command, check=False)
    return int(printed) if printed.isdigit() else None


def container_throttling(container: str) -> tuple[int | None, int | None]:
    """The CPU periods in which *container*'s quota ran out, and the microseconds it waited."""
    printed = docker("exec", container, "cat", "/sys/fs/cgroup/cpu.stat", check=False)
    stat = parse_flat_keyed(printed)
    return stat.get("nr_throttled"), stat.get("throttled_usec")


def most_threads(container: str) -> int | None:
    """The most threads one process in *container* runs, or ``None``."""
    printed = docker(
        "exec", container, "sh", "-c", "cat /proc/[0-9]*/status 2>/dev/null", check=False
    )  # fmt: skip
    counts = [
        int(parts[1])
        for parts in (line.split() for line in printed.splitlines())
        if len(parts) == 2 and parts[0] == "Threads:" and parts[1].isdigit()
    ]
    return max(counts) if counts else None


def _delta(before: int | None, after: int | None) -> int | None:
    return None if before is None or after is None else after - before


def measure_reference(settings: SpikeSettings, pinned_cpu: str | None = None) -> ReferenceResult:
    """Start the reference image cold, as the spike's runs start, and time its first prediction.

    With *pinned_cpu* the container is also pinned to that one CPU.
    """
    reference = settings.reference
    client = Client(settings.runs.port, settings.runs.request_timeout_s)
    payload = {"text": reference.text}
    measured_at = _now()
    cold: list[ReferenceColdStart] = []
    warm: list[float] = []
    container = None
    failure = None
    cpus_seen = peak = threads = None
    warm_periods = warm_us = None
    try:
        for number in range(reference.cold_starts):
            drop_page_cache(settings)
            container = reference_container(settings, pinned_cpu)
            started = time.perf_counter()
            docker("start", container)
            start_s = time.perf_counter() - started
            client.wait_ready(settings.runs.ready_timeout_s)
            ready_s = time.perf_counter() - started
            first_ms, _ = client.post("/predict", payload)
            total_s = time.perf_counter() - started
            periods, waited_us = container_throttling(container)
            cold.append(
                ReferenceColdStart(
                    start_s=round(start_s, 3),
                    ready_s=round(ready_s, 3),
                    first_predict_ms=round(first_ms, 1),
                    total_s=round(total_s, 3),
                    throttled_periods=periods,
                    throttled_s=_seconds(waited_us),
                )
            )
            logger.info("  cold start %d: %.2f s", number + 1, total_s)
            if number < reference.cold_starts - 1:
                docker("rm", "-f", container)
        before = container_throttling(container)
        warm = [client.post("/predict", payload)[0] for _ in range(reference.warm_requests)]
        after = container_throttling(container)
        warm_periods, warm_us = _delta(before[0], after[0]), _delta(before[1], after[1])
        peak = _container_number(container, "cat", "/sys/fs/cgroup/memory.peak")
        cpus_seen = _container_number(container, "nproc")
        threads = most_threads(container)
    except (httpx.HTTPError, DockerError, TimeoutError) as error:
        state = container_state(container) if container else "no container"
        failure = f"{type(error).__name__}: {error}; container: {state}"
        logger.error("  failed: %s", failure)
    finally:
        client.close()
        if container:
            docker("rm", "-f", container, check=False)
    return summarize_reference(
        reference,
        cold,
        warm,
        pinned_cpu=pinned_cpu,
        image_bytes=image_size(reference.image),
        cpus_seen=cpus_seen,
        threads=threads,
        warm_throttled_periods=warm_periods,
        warm_throttled_s=_seconds(warm_us),
        memory_peak_bytes=peak,
        measured_at=measured_at,
        failure=failure,
    )


def _volume_file(settings: SpikeSettings, path: str) -> str:
    """The text of one file in the volume, or an empty string when it is not there."""
    return docker(
        "run", "--rm", "-v", f"{settings.runs.volume}:/stores:ro", settings.runs.images.slim,
        "sh", "-c", f"cat {shlex.quote(path)} 2>/dev/null || true", check=False,
    )  # fmt: skip


def store_bytes(settings: SpikeSettings, spec: StoreSpec) -> int | None:
    """The size on disk of what one store reads, from its manifest in the volume."""
    directory = "numpy" if spec.kind == "numpy" else spec.name
    text = _volume_file(settings, f"/stores/{directory}/store.json")
    if not text:
        return None
    manifest = json.loads(text)
    if spec.kind == "lance":
        return int(manifest["bytes"])
    vectors = [
        v for v in manifest["vectors"] if v["dim"] == spec.dim and v["format"] == spec.format
    ]
    return sum(manifest["bytes"].values()) + sum(v["bytes"] for v in vectors)


def read_preparations(text: str) -> tuple[dict[str, PreparedModel], dict[str, str]]:
    """The prepared query models by variant, and the reason each other variant was not prepared."""
    prepared: dict[str, PreparedModel] = {}
    unprepared: dict[str, str] = {}
    for entry in _PREPARATIONS.validate_json(text) if text else []:
        if entry.prepared is not None:
            prepared[entry.label] = entry.prepared
        else:
            unprepared[entry.label] = entry.error or "not prepared"
    return prepared, unprepared


def prepared_models(settings: SpikeSettings) -> tuple[dict[str, PreparedModel], dict[str, str]]:
    """What the build wrote of the query models into the volume."""
    return read_preparations(
        _volume_file(settings, f"/stores/{MODELS_DIR_NAME}/{PREPARATION_NAME}")
    )


def _libraries(image: str, packages: Sequence[str]) -> str:
    """Python's version and each package's in one image, ``-`` for one it lacks."""
    code = (
        "import importlib.metadata as m, platform\n"
        "def version(name):\n"
        "    try:\n"
        "        return m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        return '-'\n"
        f"print(platform.python_version(), *(p + '==' + version(p) for p in {list(packages)!r}))"
    )
    return docker("run", "--rm", image, "python", "-c", code)


def environment(settings: SpikeSettings) -> dict[str, str]:
    """The machine, Docker and the libraries in each image."""
    info = docker(
        "info",
        "--format",
        "{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}|{{.KernelVersion}}|{{.OperatingSystem}}|{{.CgroupVersion}}",
    )
    server, cpus, memory, kernel, system, cgroup = (info.split("|") + [""] * 6)[:6]
    packages = [
        "lancedb",
        "numpy",
        "pyarrow",
        "fastapi",
        "uvicorn",
        "onnxruntime",
        "tokenizers",
        "torch",
        "transformers",
        "sentence-transformers",
        "onnx",
    ]
    processor = docker(
        "run", "--rm", settings.runs.images.slim, "sh", "-c",
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2; grep -m1 flags /proc/cpuinfo "
        "| tr ' ' '\\n' | grep -E '^(avx2|avx512f|avx512_bf16|amx_bf16|avx_vnni|avx512_vnni)$' "
        "| tr '\\n' ' '",
        check=False,
    )  # fmt: skip
    return {
        "host": f"{platform.system()} {platform.release()}",
        "processor_and_flags": " ".join(processor.split()),
        "docker_server": server,
        "docker_vm_cpus": cpus,
        "docker_vm_memory_bytes": memory,
        "docker_vm_kernel": kernel,
        "docker_vm_system": system,
        "cgroup_version": cgroup,
        "slim_python_and_libraries": _libraries(settings.runs.images.slim, packages),
        "full_python_and_libraries": _libraries(settings.runs.images.full, packages),
    }


def workload_counts(workload: Workload) -> dict[str, dict[str, int]]:
    """How many measured searches there are of each set and each kind of filter."""
    sets: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for query in workload.queries:
        sets[query.question_set] = sets.get(query.question_set, 0) + 1
        kinds[query.filter_kind] = kinds.get(query.filter_kind, 0) + 1
    return {
        "sets": dict(sorted(sets.items())),
        "filters": dict(sorted(kinds.items())),
        "phases": {"warmup": len(workload.warmup), "parties": len(workload.party_view)},
    }


def overlaps(settings: SpikeSettings, records: dict[str, list[SearchRecord]]) -> list[Overlap]:
    """Compare each store alone with the exact float32 numpy store of its dimension, by phase."""
    found = []
    for spec in settings.stores:
        reference = f"numpy-{spec.dim}-float32"
        if spec.name == reference or spec.name not in records or reference not in records:
            continue
        for phase in (BUDGET_PHASE, "lexical", "vector"):
            result = overlap(records[spec.name], records[reference], phase)
            if result is not None:
                found.append(result)
    return found


def model_labels(settings: SpikeSettings, prepared: dict[str, PreparedModel]) -> list[ModelLabel]:
    """Every prepared query model in each runtime and precision, in the order of the settings."""
    return [
        ModelLabel(name=spec.name, runtime=runtime, precision=precision, dim=spec.dim)
        for spec in settings.query_models.models
        for runtime, precision in settings.query_models.variants()
        if variant_name(spec.name, runtime, precision) in prepared
    ]


def plan_pairs(
    settings: SpikeSettings,
    stores: Sequence[RunResult],
    models: Sequence[RunResult],
    prepared: dict[str, PreparedModel],
) -> tuple[list[Subject], list[SkippedPair]]:
    """The stores and query models to measure together, and those left out with the reason.

    A pair is measured when the store and the model have the same dimension and
    each was within the budget alone: adding a model to a store only adds memory
    and time, so a pair cannot pass where either part failed alone.
    """
    by_name = {result.name: result for result in [*stores, *models]}
    pairs, skipped = [], []
    for store in settings.stores:
        for spec in settings.query_models.models:
            if spec.dim != store.dim:
                continue
            for runtime, precision in settings.query_models.variants():
                label = ModelLabel(
                    name=spec.name, runtime=runtime, precision=precision, dim=spec.dim
                )
                alone_store = by_name.get(store.name)
                alone_model = by_name.get(label.label)
                if label.label not in prepared:
                    reason = "the model was not prepared"
                elif alone_store is None or alone_model is None:
                    reason = "one of them was not measured alone"
                elif not alone_store.verdict.within_budget:
                    reason = "the store alone is over the budget"
                elif not alone_model.verdict.within_budget:
                    reason = "the model alone is over the budget"
                else:
                    pairs.append(Subject(store=store, model=label))
                    continue
                skipped.append(SkippedPair(store=store.name, model=label.label, reason=reason))
    return pairs, skipped


def image_size(image: str) -> int | None:
    """The size of an image, or ``None`` when it is not there."""
    size = docker("image", "inspect", "-f", "{{.Size}}", image, check=False)
    return int(size) if size.isdigit() else None


def read_report(path: Path) -> SpikeReport | None:
    """The report a previous run of the spike wrote at *path*, or ``None`` when there is none."""
    if not path.exists():
        return None
    return SpikeReport.model_validate_json(path.read_text(encoding="utf-8"))


def finished_runs(report: SpikeReport | None) -> dict[str, RunResult]:
    """The runs of a previous report that ended without a failure, by name."""
    if report is None:
        return {}
    return {result.name: result for result in report.runs if result.failure is None}


def split_kept(
    previous: SpikeReport | None, remeasure: Sequence[str], reason: str | None
) -> tuple[dict[str, RunResult], list[SupersededRun]]:
    """The runs of a previous report to keep, and those set aside to be measured again.

    A run set aside keeps the container settings of the report it came from,
    so that what changed stays in the report.

    Raises:
        ValueError: If a run to measure again has no reason, or the report holds no such run.
    """
    kept = finished_runs(previous)
    superseded = list(previous.superseded) if previous is not None else []
    if not remeasure:
        return kept, superseded
    if not reason:
        raise ValueError("A run measured again needs a reason.")
    by_name = {result.name: result for result in previous.runs} if previous is not None else {}
    for name in remeasure:
        if name not in by_name or previous is None:
            raise ValueError(f"The report holds no run called {name!r}.")
        kept.pop(name, None)
        superseded.append(
            SupersededRun(run=by_name[name], container=previous.container, reason=reason)
        )
    return kept, superseded


def finished_references(report: SpikeReport | None) -> dict[str | None, ReferenceResult]:
    """The reference's conditions in a previous report that ended without a failure."""
    if report is None:
        return {}
    return {result.pinned_cpu: result for result in report.references if result.failure is None}


def kept_records(path: Path, names: set[str]) -> dict[str, list[SearchRecord]]:
    """The records in *path* of the runs named, by run, in the order they were written."""
    kept: dict[str, list[SearchRecord]] = {}
    if not path.exists():
        return kept
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                record = SearchRecord.model_validate_json(line)
                if record.run in names:
                    kept.setdefault(record.run, []).append(record)
    return kept


def write_atomically(report: SpikeReport, path: Path) -> None:
    """Write the report beside *path*, then put it in place, so that a stop never halves it."""
    partial = path.with_name(path.name + ".partial")
    write_report(report, partial)
    os.replace(partial, path)


def with_memory(settings: SpikeSettings, memory: str) -> SpikeSettings:
    """The settings with every measured container's memory, and the budget's, set to *memory*.

    Raises:
        ValueError: If *memory* is not a whole number of GiB as docker writes it, such as ``4g``.
    """
    digits = memory.removesuffix("g")
    if not (memory.endswith("g") and digits.isdigit() and int(digits) > 0):
        raise ValueError(f"A memory limit is a whole number of GiB, such as 4g, not {memory!r}.")
    return settings.model_copy(
        update={
            "container": settings.container.model_copy(update={"memory": memory}),
            "budget": settings.budget.model_copy(update={"memory_gib": float(digits)}),
        }
    )


def output_names(memory: str | None, settings_name: str | None = None) -> tuple[str, str]:
    """The names of the report and of the records.

    They are the main ones, or ones after the settings file when it is not
    ``spike.yaml``, each with the memory limit after it when one is given.
    """
    if settings_name:
        stem = Path(settings_name).stem + (f"_memory_{memory}" if memory else "")
        return f"{stem}.json", f"{stem}.jsonl"
    if memory is None:
        return REPORT_NAME, SEARCHES_NAME
    return f"storage_spike_memory_{memory}.json", f"searches_memory_{memory}.jsonl"


def rewrite_report(path: Path) -> int:
    """Read the report at *path*, add its estimates for Cloud Run, and write it again.

    Returns:
        The exit code: 1 when there is no report to rewrite.
    """
    report = read_report(path)
    if report is None:
        logger.error("There is no report at %s to rewrite.", path)
        return 1
    estimates = cloud_run_estimates(report.budget, report.runs, report.references)
    write_atomically(report.model_copy(update={"cloud_run_estimates": tuple(estimates)}), path)
    logger.info("Rewrote %s", path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Measure the stores of the storage spike.")
    parser.add_argument("--data-root", type=Path, default=None, help="Where the bulk data lives.")
    parser.add_argument(
        "--settings",
        default=None,
        help=(
            "Read this settings file beside spike.yaml instead, here and in every container; "
            "the report and the records are named after it."
        ),
    )
    parser.add_argument("--image", action="store_true", help="Build the images first.")
    parser.add_argument(
        "--build", action="store_true", help="Build the stores and prepare the models first."
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also measure the inference service's image; with --image, build it first.",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, help="Build and measure only these stores."
    )
    parser.add_argument("--no-measure", action="store_true", help="Stop after building.")
    parser.add_argument(
        "--memory",
        default=None,
        help=(
            "Run the measured containers with this memory limit instead, such as 4g; the "
            "budget's memory becomes the same, and the report and the records are written "
            "beside the main ones under names that carry the limit."
        ),
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="Measure no query model alone or in a pair."
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help=(
            "Measure nothing: read the report already written, derive what this "
            "version of the report adds, and write it again."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep the runs of the report already written that ended without a failure, "
            "with their records, and measure the rest. The settings must be the same."
        ),
    )
    parser.add_argument(
        "--remeasure",
        nargs="+",
        default=(),
        help=(
            "With --resume, measure these runs again; the report keeps their first "
            "measurement among its superseded runs."
        ),
    )
    parser.add_argument(
        "--reason", default=None, help="Why the runs of --remeasure are measured again."
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="The Hugging Face hub cache the query models and the classifier are read from.",
    )
    parser.add_argument(
        "--docker-command",
        default="docker",
        help="The command that runs Docker's client, such as 'wsl -d Ubuntu -u root -- docker'.",
    )
    parser.add_argument(
        "--wsl-paths",
        action="store_true",
        help="Name this machine's paths to the engine as WSL does, the drive C: as /mnt/c.",
    )
    args = parser.parse_args(argv)
    if args.remeasure and not args.resume:
        parser.error("--remeasure needs --resume.")
    if args.remeasure and not args.reason:
        parser.error("--remeasure needs --reason.")
    return args


def main(argv: list[str] | None = None) -> int:
    """Build what the command line asks for, then measure every run and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    global CLI
    CLI = DockerCli(command=tuple(shlex.split(args.docker_command)), wsl=args.wsl_paths)
    search_settings = load_settings(data_root=args.data_root)
    if args.settings:
        settings_file(args.settings)
        os.environ[SETTINGS_ENV] = args.settings
    settings = load_spike_settings()
    if args.memory is not None:
        settings = with_memory(settings, args.memory)
    report_name, searches_name = output_names(args.memory, args.settings)
    if args.rewrite:
        return rewrite_report(search_settings.tracked.reports / report_name)
    hf_cache = args.hf_cache or default_hf_cache()
    if args.image:
        build_images(settings)
        if args.reference:
            build_reference_image(settings, hf_cache)

    columns = read_passage_columns(search_settings.data.search / "passages.parquet")
    workload = build_workload(settings.workload, search_settings.tracked.questions, columns)
    if args.build:
        build_stores(
            settings,
            search_settings.data.search,
            search_settings.tracked.reports,
            hf_cache,
            args.only,
            check_texts(workload, settings.query_models.check_texts),
        )
    if args.no_measure:
        return 0

    report_path = search_settings.tracked.reports / report_name
    previous = read_report(report_path) if args.resume else None
    kept_runs, superseded = split_kept(previous, args.remeasure, args.reason)
    kept_references = finished_references(previous)
    stores = [spec for spec in settings.stores if args.only is None or spec.name in args.only]
    prepared, unprepared = prepared_models(settings)
    searches_path = search_settings.data.search / SPIKE_DATA_DIR / searches_name
    searches_path.parent.mkdir(parents=True, exist_ok=True)
    records = kept_records(searches_path, set(kept_runs))
    images = settings.runs.images
    fixed: dict[str, Any] = {
        "budget": settings.budget,
        "container": settings.container,
        "search": settings.search,
        "vectors": settings.vectors,
        "workload": workload_counts(workload),
        "environment": environment(settings),
        "images": {image: image_size(image) for image in (images.slim, images.full)},
        "unprepared": unprepared,
    }
    results: list[RunResult] = []
    skipped: list[SkippedPair] = []
    references: list[ReferenceResult] = []

    def write(finished: bool) -> None:
        report = SpikeReport(
            created=_now(),
            finished=finished,
            runs=tuple(results),
            skipped=tuple(skipped),
            overlaps=tuple(overlaps(settings, records)),
            references=tuple(references),
            cloud_run_estimates=tuple(cloud_run_estimates(settings.budget, results, references)),
            superseded=tuple(superseded),
            **fixed,
        )
        write_atomically(report, report_path)

    with searches_path.open("w", encoding="utf-8", newline="\n") as out:
        for kept in records.values():
            out.writelines(record.model_dump_json() + "\n" for record in kept)
        out.flush()

        def measure(subject: Subject) -> RunResult:
            if subject.name in kept_runs:
                logger.info("%s: kept from the report", subject.name)
                result = kept_runs[subject.name]
            else:
                logger.info("%s", subject.name)
                run = ServiceRun(settings, subject, workload)
                result = run.run()
                records[subject.name] = run.records
                out.writelines(record.model_dump_json() + "\n" for record in run.records)
                out.flush()
                update: dict[str, Any] = {}
                if subject.store is not None:
                    update["store_bytes"] = store_bytes(settings, subject.store)
                if subject.model is not None:
                    update["prepared"] = prepared.get(subject.model.label)
                logger.info("  verdict: %s", result.verdict.model_dump())
                result = result.model_copy(update=update)
            results.append(result)
            write(finished=False)
            return result

        alone = [measure(Subject(store=spec, model=None)) for spec in stores]
        labels = [] if args.skip_models else model_labels(settings, prepared)
        models = [measure(Subject(store=None, model=label)) for label in labels]
        pairs, left_out = plan_pairs(
            settings.model_copy(update={"stores": tuple(stores)}), alone, models, prepared
        )
        skipped.extend(left_out)
        for subject in pairs:
            measure(subject)
        planned = {result.name for result in results}
        results.extend(result for name, result in kept_runs.items() if name not in planned)

    if args.reference:
        image = settings.reference.image
        for pinned_cpu in reference_conditions(settings.reference):
            if pinned_cpu in kept_references:
                logger.info("%s, pinned to CPU %s: kept from the report", image, pinned_cpu)
                references.append(kept_references[pinned_cpu])
                continue
            logger.info("%s, pinned to CPU %s", image, pinned_cpu)
            references.append(measure_reference(settings, pinned_cpu))
            write(finished=False)
    write(finished=True)
    logger.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
