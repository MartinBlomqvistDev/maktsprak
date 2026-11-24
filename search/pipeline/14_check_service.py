"""Check the tuned configuration in a build of the search service, before the test split.

The pre-registration has the tuned configuration checked in a build of the
service before the test split is opened: the build embeds the questions exactly
as the service will, searches in the format and index the service will use, and
must reach recall@10 of at least 0.97 against the exact float32 search, over the
questions of the development split, and meet the storage rule.

- ``embed`` asks the API for the vector of every question the evaluation
  searched on the development split, one request per question over one
  connection, as the service will ask. It records how long each answer took and
  how many requests it needed, and how close each vector comes to the one the
  evaluation searched with. The vectors go to the search-data directory, the
  summary to the tracked ``service_queries_dev.json``.
- ``compare`` reads what the build of the service returned. The storage spike's
  runner measures that build in a container with one vCPU twice: under its own
  workload, for the storage rule (``service_storage.json``), and with every
  evaluated question and the filter it carries (``service_recall.json``, and
  each search's hits in ``spike/service_recall.jsonl``). The hits are held
  against the exact float32 search of the evaluation, hybrid and each leg, and
  ``service_check_dev.json`` says whether the build passed.

Usage::

    python search/pipeline/14_check_service.py embed [--max-cost-usd 0.02] [--env-file PATH]
    python search/pipeline/14_check_service.py compare
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.store import write_report
from search.corpus.table import load_passage_table
from search.eval.baseline import FilterMasks
from search.eval.hybrid import (
    QueryVectors,
    fused_hits,
    lexical_hits,
    require_committed,
    vector_hits,
)
from search.index.embed import (
    Answered,
    OpenRouterEmbedder,
    build_embedder,
    ids_sha256,
    load_openrouter_key,
    read_inputs,
    read_query_vectors,
    read_run_manifest,
    save_array,
    software_versions,
)
from search.index.lexical import file_sha256
from search.index.vector import inputs_path, passages_path, run_path

logger = logging.getLogger(__name__)

TUNING_REPORT = "hybrid_tuning_dev.json"
QUERIES_REPORT = "service_queries_dev.json"
SERVICE_CHECK_DIR = "service_check"
QUERIES_NAME = "queries.npy"
TEXTS_NAME = "query_texts.json"
ANSWERS_NAME = "answers.jsonl"
MANIFEST_NAME = "manifest.json"
#: The waits a question's vector is compared against, in seconds.
WAITS_S = (1.0, 2.0, 3.0, 5.0, 10.0)
QUANTILES = (0.5, 0.9, 0.95, 0.99)
CHECK_REPORT = "service_check_dev.json"
#: The storage spike's records, below the search-data directory.
SPIKE_DIR = "spike"
#: The settings of the spike's two runs of the build, which name their reports.
STORAGE_RUN = "service_storage"
RECALL_RUN = "service_recall"
#: The pre-registration's floor for the build's recall@10 against the exact float32 search.
RECALL_FLOOR = 0.97
#: The phases of the recall run held against the exact search: the hybrid and each leg alone.
COMPARED_PHASES = ("hybrid", "vector", "lexical")
TOP = 10


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Tuned(_Frozen):
    """The configuration the tuning chose, as its committed report records it.

    Attributes:
        candidate: The embedding model.
        target_words: The passage size.
        k: The RRF constant.
    """

    candidate: str
    target_words: int
    k: float


class Waiting(_Frozen):
    """How long the questions waited for their vectors.

    Attributes:
        quantiles_s: The waits at the 50th, 90th, 95th and 99th percentile, in seconds.
        max_s: The longest wait.
        within: The share of questions answered within each wait, by the wait in seconds.
        retried: How many questions needed more than one request.
    """

    quantiles_s: dict[str, float]
    max_s: float
    within: dict[str, float]
    retried: int


class Closeness(_Frozen):
    """How close the vectors asked for one by one come to the evaluation's.

    Attributes:
        min: The lowest cosine.
        p1: The cosine at the first percentile.
        median: The median cosine.
    """

    min: float
    p1: float
    median: float


class QueriesReport(_Frozen):
    """The questions' vectors asked for as the service asks for them.

    Attributes:
        tuned: The configuration being checked.
        inputs: The input set whose questions were embedded.
        questions: How many questions.
        query_ids_sha256: The hash of their ids, in order.
        vectors_sha256: The hash of the vectors written.
        started: When the first request was sent, in UTC.
        finished: When the last vector came back, in UTC.
        requests: How many requests were sent, retries included.
        tokens: The tokens the API counted.
        cost_usd: What the requests cost.
        providers: Who served them.
        provider: The provider the requests were routed to, without fallback.
        quantization: The precision they were routed to.
        waiting: How long the questions waited.
        closeness: How close the vectors are to the evaluation's.
        software: The versions of the libraries used.
    """

    tuned: Tuned
    inputs: str
    questions: int
    query_ids_sha256: str
    vectors_sha256: str
    started: str
    finished: str
    requests: int
    tokens: int
    cost_usd: float
    providers: tuple[str, ...]
    provider: str
    quantization: str
    waiting: Waiting
    closeness: Closeness
    software: dict[str, str]


def tuned_configuration(settings: Settings) -> Tuned:
    """The configuration the committed tuning chose.

    Raises:
        SystemExit: If the tuning's report is not committed as it stands.
    """
    path = settings.tracked.reports / TUNING_REPORT
    require_committed(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    chosen = report["chosen"]
    return Tuned(candidate=report["candidate"], target_words=chosen["target_words"], k=chosen["k"])


def summarize_waiting(answered: list[Answered]) -> Waiting:
    """How long the questions waited, from how each was answered."""
    seconds = np.array([one.seconds for one in answered], dtype=np.float64)
    return Waiting(
        quantiles_s={
            f"p{round(q * 100)}": round(float(np.quantile(seconds, q)), 3) for q in QUANTILES
        },
        max_s=round(float(seconds.max()), 3),
        within={f"{wait:g}": round(float(np.mean(seconds <= wait)), 4) for wait in WAITS_S},
        retried=sum(one.attempts > 1 for one in answered),
    )


def summarize_closeness(vectors: np.ndarray, evaluated: np.ndarray) -> Closeness:
    """How close each vector is to the evaluation's, by cosine; both are unit vectors."""
    cosines = np.einsum("ij,ij->i", vectors.astype(np.float64), evaluated.astype(np.float64))
    return Closeness(
        min=round(float(cosines.min()), 6),
        p1=round(float(np.quantile(cosines, 0.01)), 6),
        median=round(float(np.median(cosines)), 6),
    )


def service_dir(settings: Settings, candidate: str) -> Path:
    """Where the check keeps its data for one model."""
    return settings.data.search / SERVICE_CHECK_DIR / candidate


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_embed(settings: Settings, max_cost_usd: float, env_file: Path | None) -> None:
    """Ask for every evaluated question's vector one by one, as the service will."""
    tuned = tuned_configuration(settings)
    candidate = settings.vectors.candidate(tuned.candidate)
    if candidate.kind != "openrouter":
        raise SystemExit(f"{tuned.candidate} runs locally; this step checks an API model.")
    inputs_name = f"full-{tuned.target_words}"
    inputs = read_inputs(inputs_path(settings.data.search, inputs_name))
    run_dir = run_path(settings.data.search, inputs_name, tuned.candidate)
    evaluated = read_query_vectors(run_dir, read_run_manifest(run_dir))
    ids, texts = list(inputs.query_ids), list(inputs.query_texts)
    if evaluated.shape[0] != len(ids):
        raise SystemExit(f"The run over {inputs_name} has {evaluated.shape[0]} question vectors.")

    key = load_openrouter_key(env_file or settings.data.search.parents[1] / ".env")
    embedder = build_embedder(
        candidate, key=key, cap_usd=max_cost_usd, batch_inputs=1, concurrency=1
    )
    if not isinstance(embedder, OpenRouterEmbedder):
        raise SystemExit(f"{tuned.candidate} did not give an API embedder.")
    logger.info("Asking for %d question vectors one by one", len(ids))
    started = _utc()
    vectors, answered = embedder.embed_singly(texts, "query")
    finished = _utc()
    usage = embedder.take_usage()

    out = service_dir(settings, tuned.candidate)
    out.mkdir(parents=True, exist_ok=True)
    vectors_sha256 = save_array(out / QUERIES_NAME, vectors.astype(np.float32))
    (out / TEXTS_NAME).write_text(
        json.dumps({"query_ids": ids, "texts": texts}, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    with (out / ANSWERS_NAME).open("w", encoding="utf-8", newline="\n") as lines:
        for question_id, one in zip(ids, answered, strict=True):
            lines.write(
                json.dumps(
                    {
                        "query_id": question_id,
                        "seconds": round(one.seconds, 4),
                        "attempts": one.attempts,
                    }
                )
                + "\n"
            )
    report = QueriesReport(
        tuned=tuned,
        inputs=inputs_name,
        questions=len(ids),
        query_ids_sha256=ids_sha256(ids),
        vectors_sha256=vectors_sha256,
        started=started,
        finished=finished,
        requests=usage.requests,
        tokens=usage.tokens,
        cost_usd=round(usage.cost_usd, 6),
        providers=tuple(sorted(embedder.providers)),
        provider=str(candidate.provider),
        quantization=str(candidate.quantization),
        waiting=summarize_waiting(answered),
        closeness=summarize_closeness(vectors, evaluated),
        software=software_versions(),
    )
    (out / MANIFEST_NAME).write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_report(report, settings.tracked.reports / QUERIES_REPORT)
    logger.info(
        "%d vectors in %d requests, %.4f USD; waits %s; lowest cosine %.6f",
        len(ids),
        usage.requests,
        usage.cost_usd,
        report.waiting.quantiles_s,
        report.closeness.min,
    )


class Agreement(_Frozen):
    """How the build's ten best of one phase agree with the exact search's.

    Attributes:
        questions: How many questions were compared.
        missing: How many evaluated questions the build's records lack.
        recall_10: The mean share of the exact search's ten best that the build returned.
        lowest: The lowest share of one question.
        below_one: How many questions the build missed a passage of the ten best for.
        identical: How many questions got the same ten passages in the same order.
    """

    questions: int
    missing: int
    recall_10: float
    lowest: float
    below_one: int
    identical: int


class Storage(_Frozen):
    """The build held to the storage rule, as the spike's runner measured it.

    Attributes:
        budget: The rule's bounds, as the run's report records them.
        p95_ms: The p95 of the hybrid searches, outside the container.
        cold_start_s: The slowest cold start.
        memory_peak_gib: The cgroup's peak memory.
        survived: Whether the container served every request without being killed.
        within_budget: Whether all three bounds hold.
    """

    budget: dict[str, float]
    p95_ms: float | None
    cold_start_s: float | None
    memory_peak_gib: float | None
    survived: bool
    within_budget: bool


class ServiceCheck(_Frozen):
    """The check of the tuned configuration in a build of the service.

    Attributes:
        tuned: The configuration checked.
        split: The split whose questions were searched.
        store: The store the build searched.
        recall_floor: The least recall@10 against the exact search the build may have.
        phases: How the build agrees with the exact search, by phase.
        with_own_vectors: How it agrees, for the hybrid and the vector leg, when
            the exact search too searches with the vectors the service asked
            for: what is left once the API's differences between its answers
            to the same question are taken out. Decides nothing.
        storage: The build held to the storage rule.
        queries: How the questions' vectors were asked for, from ``service_queries_dev.json``.
        sources_sha256: The hash of each file the check read, by name.
        passed: Whether the hybrid reaches the floor for every question and the
            build is within the storage rule.
    """

    tuned: Tuned
    split: str
    store: str
    recall_floor: float
    phases: dict[str, Agreement]
    with_own_vectors: dict[str, Agreement]
    storage: Storage
    queries: dict[str, object]
    sources_sha256: dict[str, str]
    passed: bool


def load_step13() -> ModuleType:
    """Step 13, whose search the build is held against, as a module."""
    path = Path(__file__).resolve().with_name("13_evaluate_vectors.py")
    spec = importlib.util.spec_from_file_location("pipeline_13_evaluate_vectors", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path} cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def agreement(built: Mapping[str, Sequence[str]], exact: Mapping[str, Sequence[str]]) -> Agreement:
    """How the build's ten best agree with the exact search's, question by question.

    Args:
        built: The build's passages for each question, best first.
        exact: The exact search's, for every question that was to be compared.

    Returns:
        The agreement. A question the exact search finds nothing for counts as agreeing
        when the build finds nothing either.
    """
    shares, identical = [], 0
    for question_id, expected in exact.items():
        if question_id not in built:
            continue
        top, got = list(expected[:TOP]), list(built[question_id][:TOP])
        shares.append(len(set(top) & set(got)) / len(top) if top else float(not got))
        identical += top == got
    return Agreement(
        questions=len(shares),
        missing=len(exact) - len(shares),
        recall_10=round(float(np.mean(shares)), 6) if shares else 0.0,
        lowest=round(min(shares), 6) if shares else 0.0,
        below_one=sum(share < 1 for share in shares),
        identical=identical,
    )


def read_hits(path: Path, run: str) -> dict[str, dict[str, list[str]]]:
    """The passages the build returned for each question, by phase, from the spike's records."""
    hits: dict[str, dict[str, list[str]]] = {phase: {} for phase in COMPARED_PHASES}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            if record["run"] == run and record["phase"] in hits:
                hits[record["phase"]][record["query_id"]] = list(record["hit_ids"])
    return hits


def the_run(report_path: Path) -> Mapping[str, object]:
    """The one run of a report of the spike's runner, which must have finished.

    Raises:
        SystemExit: If the report is missing, holds another number of runs, or the run failed.
    """
    if not report_path.is_file():
        raise SystemExit(f"{report_path} does not exist. Run the spike's runner first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report["runs"]
    if not report["finished"] or len(runs) != 1 or runs[0]["failure"] is not None:
        raise SystemExit(f"{report_path} does not hold one finished run.")
    return {"budget": report["budget"], **runs[0]}


Rankings = dict[str, dict[str, list[str]]]


def exact_rankings(settings: Settings, tuned: Tuned) -> tuple[Rankings, Rankings]:
    """The exact float32 search of the evaluation: each question's ten best, by phase.

    Returns:
        The search with the evaluation's question vectors, and the same search
        with the vectors the service asked for.
    """
    step13 = load_step13()
    questions = [q for qs in step13.question_sets(settings, "dev").values() for q in qs]
    path = passages_path(settings.data.search, tuned.target_words, settings.passages.target_words)
    table = load_passage_table(path)
    found = step13.candidate_vectors(settings, f"full-{tuned.target_words}", tuned.candidate, table)
    if found is None:
        raise SystemExit(f"{tuned.candidate} has no index over full-{tuned.target_words}.")
    index, queries, _ = found
    masks = FilterMasks(table)
    bm25 = step13.registered_bm25(settings)
    lexical = step13.lexical_index_at(settings, bm25, tuned.target_words, path)
    window = settings.vectors.window
    words = lexical_hits(lexical, bm25.params, masks, questions, window)
    own_dir = service_dir(settings, tuned.candidate)
    own = QueryVectors.of(
        json.loads((own_dir / TEXTS_NAME).read_text(encoding="utf-8"))["query_ids"],
        np.load(own_dir / QUERIES_NAME, allow_pickle=False),
    )
    ids = table.passage_ids

    def tops(vectors: list[list[int]]) -> Rankings:
        rankings = {
            "hybrid": fused_hits(words, vectors, tuned.k, window),
            "vector": vectors,
            "lexical": words,
        }
        return {
            phase: {
                question.question_id: [ids[position] for position in ranking[:TOP]]
                for question, ranking in zip(questions, ranked, strict=True)
            }
            for phase, ranked in rankings.items()
        }

    return (
        tops(vector_hits(index, masks, questions, queries, window)),
        tops(vector_hits(index, masks, questions, own, window)),
    )


def run_compare(settings: Settings) -> None:
    """Hold the build's hits against the exact search and the storage rule."""
    tuned = tuned_configuration(settings)
    reports = settings.tracked.reports
    storage_path = reports / f"{STORAGE_RUN}.json"
    recall_path = reports / f"{RECALL_RUN}.json"
    records_path = settings.data.search / SPIKE_DIR / f"{RECALL_RUN}.jsonl"
    queries_path = reports / QUERIES_REPORT
    storage_run, recall_run = the_run(storage_path), the_run(recall_path)
    store = str(recall_run["name"])
    if storage_run["name"] != store:
        raise SystemExit(f"The storage run measured {storage_run['name']}, the recall run {store}.")
    verdict = storage_run["verdict"]
    budget = storage_run["budget"]
    storage = Storage(
        budget={name: float(budget[name]) for name in ("p95_ms", "cold_start_s", "memory_gib")},
        p95_ms=verdict["p95_ms"],
        cold_start_s=verdict["cold_start_s"],
        memory_peak_gib=verdict["memory_peak_gib"],
        survived=verdict["survived"],
        within_budget=verdict["within_budget"],
    )
    built = read_hits(records_path, store)
    exact, own = exact_rankings(settings, tuned)
    phases = {phase: agreement(built[phase], exact[phase]) for phase in COMPARED_PHASES}
    with_own = {phase: agreement(built[phase], own[phase]) for phase in ("hybrid", "vector")}
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if queries["tuned"] != tuned.model_dump():
        raise SystemExit(f"{queries_path} was made for another configuration.")
    hybrid = phases["hybrid"]
    check = ServiceCheck(
        tuned=tuned,
        split="dev",
        store=store,
        recall_floor=RECALL_FLOOR,
        phases=phases,
        with_own_vectors=with_own,
        storage=storage,
        queries={name: queries[name] for name in ("questions", "waiting", "closeness")},
        sources_sha256={
            path.name: file_sha256(path)
            for path in (storage_path, recall_path, records_path, queries_path)
        },
        passed=hybrid.missing == 0 and hybrid.recall_10 >= RECALL_FLOOR and storage.within_budget,
    )
    write_report(check, reports / CHECK_REPORT)
    for phase, agreed in phases.items():
        logger.info(
            "%-7s recall@10 %.4f against the exact search, lowest %.2f, %d below 1, "
            "%d identical, %d missing",
            phase,
            agreed.recall_10,
            agreed.lowest,
            agreed.below_one,
            agreed.identical,
            agreed.missing,
        )
    for phase, agreed in with_own.items():
        logger.info(
            "%-7s with the service's own vectors: recall@10 %.4f, %d below 1, %d identical",
            phase,
            agreed.recall_10,
            agreed.below_one,
            agreed.identical,
        )
    logger.info(
        "Storage: p95 %s ms, cold start %s s, peak %s GiB, within the budget: %s",
        storage.p95_ms,
        storage.cold_start_s,
        storage.memory_peak_gib,
        storage.within_budget,
    )
    logger.info("Passed: %s", check.passed)


def main() -> None:
    """Parse the command line and run the command it names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    embed = commands.add_parser("embed", help="Ask for the questions' vectors as the service will.")
    embed.add_argument("--max-cost-usd", type=float, default=0.02)
    embed.add_argument("--env-file", type=Path, default=None)
    commands.add_parser("compare", help="Hold the build's hits against the exact search.")
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    if args.command == "embed":
        run_embed(settings, args.max_cost_usd, args.env_file)
    else:
        run_compare(settings)


if __name__ == "__main__":
    main()
