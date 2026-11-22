"""Step 13: score the vector leg and the hybrid, by the rules of the addendum of 2026-11-21.

Three commands, run in this order, each refusing to run before its turn:

- ``compare`` is step A. Every candidate's vectors search the subset on their
  own, on the development split, and the rule chooses the embedding model. It
  runs only once the candidates and the question sets are registered in the
  committed pre-registration and the subset's record is committed.
- ``tune`` tries the hybrid at every passage size and every RRF constant on the
  development split, with the chosen model over the whole corpus. It runs only
  once the comparison's report is committed.
- ``test`` is the one run on the test split. It runs only with ``--confirm``,
  once an addendum records the tuning's report by its hash, and never twice.

Usage::

    python search/pipeline/13_evaluate_vectors.py compare
    python search/pipeline/13_evaluate_vectors.py tune
    python search/pipeline/13_evaluate_vectors.py test --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.store import SPEECHES_NAME, write_report
from search.corpus.table import PassageTable, load_passage_table
from search.eval.baseline import FilterMasks, QuestionResult, SetResult, speech_years, summarize_set
from search.eval.hybrid import (
    DECIDING_SET,
    VECTOR_COMPARISON_SETS,
    CandidateChoice,
    CandidateScore,
    Comparison,
    Decision,
    HybridPoint,
    QueryVectors,
    choose_candidate,
    choose_hybrid,
    compare,
    decide,
    deviating_parties,
    fused_hits,
    lexical_hits,
    mean_of,
    require_committed,
    require_registered,
    require_sets_registered,
    score_rankings,
    speech_parties,
    text_sha256,
    tune_grid,
    vector_hits,
)
from search.eval.questions import EvalQuestion, read_question_set, split_of
from search.index.analyzer import Analyzer
from search.index.embed import (
    EmbedderSpec,
    RunManifest,
    build_embedder,
    load_openrouter_key,
    read_inputs,
    read_query_vectors,
    read_run_manifest,
)
from search.index.lexical import (
    Bm25Params,
    LexicalIndex,
    build_index,
    lexical_dir,
    passage_texts,
    tokenize_corpus,
)
from search.index.vector import (
    VectorIndex,
    index_path,
    inputs_path,
    load_vector_index,
    passages_path,
    run_path,
)

logger = logging.getLogger("vectors")

#: The sets with gold speeches, which retrieval is scored on.
SCORED_SETS = ("natural", "keyword", "interpellation", "filtered", "stance")

SUBSET_REPORT = "vector_subset.json"
CANDIDATES_REPORT = "embedding_candidates.json"
BM25_REPORT = "bm25_tuning_dev.json"
COMPARISON_REPORT = "vector_comparison_dev.json"
COMPARISON_QUESTIONS = "vector_comparison_dev_questions.jsonl"
TUNING_REPORT = "hybrid_tuning_dev.json"
TEST_REPORT = "retrieval_test.json"
TEST_QUESTIONS = "retrieval_test_questions.jsonl"


class _Frozen(BaseModel):
    """Base for the reports here: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Bm25Config(_Frozen):
    """The BM25 the comparisons use, as the tuning of 2026-11-21 chose it.

    Attributes:
        analyzer: The analyzer's name.
        k1: BM25's ``k1``.
        b: BM25's ``b``.
    """

    analyzer: str
    k1: float
    b: float

    @property
    def params(self) -> Bm25Params:
        """``k1`` and ``b``."""
        return Bm25Params(k1=self.k1, b=self.b)

    def make_analyzer(self) -> Analyzer:
        """The analyzer the name stands for."""
        stem, stop = self.analyzer.split("-")
        return Analyzer(stem=stem == "stem", stopwords=stop == "stop")


class CandidateReport(_Frozen):
    """One candidate in the comparison of step A.

    Attributes:
        embedder: What produced its vectors.
        run: What its run used: texts cut, requests, tokens and cost.
        sets: Each set's result on the subset.
    """

    embedder: EmbedderSpec
    run: dict[str, float | int]
    sets: dict[str, SetResult]


class ComparisonReport(_Frozen):
    """The comparison of step A on the development split.

    Attributes:
        split: The split scored.
        sets_sha256: The hash of each question set.
        subset_passages: How many passages the subset holds.
        subset_sha256: The hash of the subset's passage ids.
        resamples: The bootstrap resamples behind each interval.
        bootstrap_seed: The seed of the resamples.
        candidates: Each candidate's result, for those whose run is whole.
        missing: The candidates without a whole run, which the comparison waits for.
        bm25_on_subset: BM25 among the subset's passages, beside them; it decides nothing.
        scores: Each candidate's standing under the rule, once every candidate is in.
        choice: The rule's choice, once every candidate is in.
    """

    split: str
    sets_sha256: dict[str, str]
    subset_passages: int
    subset_sha256: str
    resamples: int
    bootstrap_seed: int
    candidates: dict[str, CandidateReport]
    missing: tuple[str, ...]
    bm25_on_subset: dict[str, SetResult]
    scores: tuple[CandidateScore, ...]
    choice: CandidateChoice | None


class SizeReport(_Frozen):
    """What one passage size gave in the tuning, beside the hybrid's grid.

    Attributes:
        target_words: The passage size.
        passages: How many passages the corpus has at that size.
        vectors_recall_10: The vectors alone, recall@10 on the natural questions.
        bm25_recall_10: BM25 at that size, recall@10 on the natural questions.
    """

    target_words: int
    passages: int
    vectors_recall_10: float
    bm25_recall_10: float


class TuningReport(_Frozen):
    """The tuning of the hybrid on the development split.

    Attributes:
        candidate: The embedding model the comparison chose.
        tuned_on: The set the configurations were scored on.
        questions: How many questions that set has in the development split.
        bm25: The BM25 configuration inside the hybrid.
        window: How many passages each ranking holds.
        sizes: What each passage size gave beside the grid.
        grid: Every configuration tried, and its scores.
        chosen: The configuration the rule picked.
    """

    candidate: str
    tuned_on: str
    questions: int
    bm25: Bm25Config
    window: int
    sizes: tuple[SizeReport, ...]
    grid: tuple[HybridPoint, ...]
    chosen: HybridPoint


class TestReport(_Frozen):
    """The one run on the test split, and what it decides.

    Attributes:
        split: The split scored.
        sets_sha256: The hash of each question set.
        tuning_sha256: The hash of the tuning's report, as the addendum records it.
        candidate: The embedding model.
        target_words: The passage size of the vectors and the hybrid.
        k: The RRF constant.
        bm25: The BM25 configuration of the comparisons, at the corpus's own size.
        query_embedding: What embedding the test questions used.
        resamples: The bootstrap resamples behind each interval.
        bootstrap_seed: The seed of the resamples.
        comparisons: The pre-registered comparisons.
        decision: What the first comparison decides.
        production: The system that runs, by the decision.
        deviating_parties: The parties whose recall@10 under the system that runs
            lies more than 0.05 from its mean, on the natural questions.
        sets: Each system's result on each set.
    """

    split: str
    sets_sha256: dict[str, str]
    tuning_sha256: str
    candidate: str
    target_words: int
    k: float
    bm25: Bm25Config
    query_embedding: dict[str, float | int | str]
    resamples: int
    bootstrap_seed: int
    comparisons: tuple[Comparison, ...]
    decision: Decision
    production: str
    deviating_parties: dict[str, float]
    sets: dict[str, dict[str, SetResult]]


def question_sets(settings: Settings, split: str) -> dict[str, list[EvalQuestion]]:
    """The questions of one split in every scored set."""
    return {
        name: split_of(read_question_set(settings.tracked.questions / f"{name}.jsonl"), split)
        for name in SCORED_SETS
    }


def registered_bm25(settings: Settings) -> Bm25Config:
    """The BM25 configuration the tuning of 2026-11-21 chose, read from its report."""
    report = json.loads((settings.tracked.reports / BM25_REPORT).read_text(encoding="utf-8"))
    chosen = report["chosen"]
    return Bm25Config(analyzer=chosen["analyzer"], k1=chosen["k1"], b=chosen["b"])


def lexical_index_at(
    settings: Settings, bm25: Bm25Config, target_words: int, table_path: Path
) -> LexicalIndex:
    """The BM25 index at one passage size: the saved one at the corpus's own size, else built."""
    analyzer = bm25.make_analyzer()
    if target_words == settings.passages.target_words:
        return LexicalIndex.load(lexical_dir(settings.data.search, analyzer))[0]
    logger.info("Building BM25 over %s", table_path.name)
    corpus = tokenize_corpus(passage_texts(settings.data.search / SPEECHES_NAME, table_path))
    return build_index(corpus, analyzer)


def run_usage(manifest: RunManifest) -> dict[str, float | int]:
    """What a run used, as the reports record it."""
    usage = manifest.usage
    return {
        "texts": usage.texts,
        "truncated": usage.truncated,
        "requests": usage.requests,
        "tokens": usage.tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }


def candidate_vectors(
    settings: Settings, inputs: str, name: str, table: PassageTable
) -> tuple[VectorIndex, QueryVectors, RunManifest] | None:
    """One candidate's index and question vectors over an input set.

    Returns ``None`` while the candidate has no index over the set.

    Raises:
        SystemExit: If the run was made with another setup than the settings name,
            or a local model computed in another precision than float32.
    """
    directory = index_path(settings.data.search, inputs, name)
    if not (directory / "index.json").is_file():
        return None
    index, manifest = load_vector_index(directory, table.passage_ids)
    expected = EmbedderSpec.from_candidate(settings.vectors.candidate(name))
    if manifest.embedder != expected:
        raise SystemExit(f"The index of {name} over {inputs} was made with another setup.")
    run_dir = run_path(settings.data.search, inputs, name)
    run = read_run_manifest(run_dir)
    if run.precision not in (None, "fp32"):
        raise SystemExit(
            f"The run of {name} over {inputs} computed in {run.precision}. "
            "The comparisons take local models in float32."
        )
    query_ids = read_inputs(inputs_path(settings.data.search, inputs)).query_ids
    queries = QueryVectors.of(query_ids, read_query_vectors(run_dir, run))
    return index, queries, run


def corpus_costs(settings: Settings) -> dict[str, float]:
    """What embedding the whole corpus costs each API candidate, from the candidates' report."""
    report = json.loads((settings.tracked.reports / CANDIDATES_REPORT).read_text(encoding="utf-8"))
    costs = {}
    for candidate in settings.vectors.candidates:
        if candidate.kind == "openrouter":
            tokens = report["api_candidates"][candidate.name]["corpus_tokens"]
            costs[candidate.name] = round(tokens * (candidate.price_per_million_usd or 0) / 1e6, 4)
    return costs


def run_compare(settings: Settings) -> None:
    """Step A: compare the candidates on the subset and choose the embedding model."""
    sets_sha256 = require_sets_registered(settings.tracked.questions)
    require_registered(
        [candidate.revision for candidate in settings.vectors.candidates], "a candidate's revision"
    )
    subset_report = settings.tracked.reports / SUBSET_REPORT
    require_committed(subset_report)
    subset = json.loads(subset_report.read_text(encoding="utf-8"))
    start = settings.passages.target_words
    subset_name = f"subset-{start}"
    inputs = read_inputs(inputs_path(settings.data.search, subset_name))
    if inputs.manifest.passage_ids_sha256 != subset["passage_ids_sha256"]:
        raise SystemExit("The subset's input set does not match its committed record.")

    base_path = passages_path(settings.data.search, start, start)
    table = load_passage_table(base_path)
    within = np.zeros(len(table), dtype=bool)
    within[inputs.positions] = True
    questions = question_sets(settings, "dev")
    masks = FilterMasks(table)
    years, parties = speech_years(table), speech_parties(table)
    baseline, window = settings.baseline, settings.vectors.window
    lines: list[str] = []

    def summarise(label: str, results_by_set: dict[str, list[QuestionResult]]):
        for name, results in results_by_set.items():
            lines.extend(
                json.dumps({"system": label, "set": name, **json.loads(r.model_dump_json())})
                for r in results
            )
        return {
            name: summarize_set(results, baseline.bootstrap_seed, baseline.resamples)
            for name, results in results_by_set.items()
        }

    reports: dict[str, CandidateReport] = {}
    missing = []
    for candidate in settings.vectors.candidates:
        found = candidate_vectors(settings, subset_name, candidate.name, table)
        if found is None:
            missing.append(candidate.name)
            logger.info("%s: no whole run over the subset yet", candidate.name)
            continue
        index, queries, run = found
        results = {
            name: score_rankings(
                table, qs, vector_hits(index, masks, qs, queries, window), years, parties
            )
            for name, qs in questions.items()
        }
        reports[candidate.name] = CandidateReport(
            embedder=EmbedderSpec.from_candidate(candidate),
            run=run_usage(run),
            sets=summarise(candidate.name, results),
        )
        logger.info(
            "%-12s recall@10 %.3f on %d natural questions",
            candidate.name,
            mean_of(results[DECIDING_SET], "recall@10"),
            len(results[DECIDING_SET]),
        )

    bm25 = registered_bm25(settings)
    lexical = lexical_index_at(settings, bm25, start, base_path)
    bm25_results = {
        name: score_rankings(
            table,
            qs,
            lexical_hits(lexical, bm25.params, masks, qs, window, within),
            years,
            parties,
        )
        for name, qs in questions.items()
    }
    bm25_sets = summarise("bm25", bm25_results)

    scores: tuple[CandidateScore, ...] = ()
    choice = None
    if not missing:
        costs = corpus_costs(settings)
        scores = tuple(
            CandidateScore(
                name=candidate.name,
                recall_10=reports[candidate.name].sets[DECIDING_SET].overall.means["recall@10"],
                mrr_10=reports[candidate.name].sets[DECIDING_SET].overall.means["mrr@10"],
                licensed=candidate.licensed,
                corpus_cost_usd=costs.get(candidate.name),
            )
            for candidate in settings.vectors.candidates
        )
        choice = choose_candidate(
            scores, settings.vectors.tie_margin, settings.vectors.api_budget_usd
        )
        logger.info("Chosen: %s (best %s, tied %s)", choice.chosen, choice.best, choice.tied)
    write_report(
        ComparisonReport(
            split="dev",
            sets_sha256={name: sets_sha256[name] for name in SCORED_SETS},
            subset_passages=inputs.manifest.passages,
            subset_sha256=inputs.manifest.passage_ids_sha256,
            resamples=baseline.resamples,
            bootstrap_seed=baseline.bootstrap_seed,
            candidates=reports,
            missing=tuple(missing),
            bm25_on_subset=bm25_sets,
            scores=scores,
            choice=choice,
        ),
        settings.tracked.reports / COMPARISON_REPORT,
    )
    (settings.tracked.reports / COMPARISON_QUESTIONS).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def chosen_candidate(settings: Settings) -> str:
    """The embedding model the committed comparison chose."""
    path = settings.tracked.reports / COMPARISON_REPORT
    require_committed(path)
    report = ComparisonReport.model_validate_json(path.read_text(encoding="utf-8"))
    if report.choice is None:
        raise SystemExit("The comparison has not chosen a model; every candidate must be in.")
    return report.choice.chosen


def run_tune(settings: Settings) -> None:
    """Tune the hybrid's passage size and RRF constant on the development split."""
    require_sets_registered(settings.tracked.questions)
    candidate = chosen_candidate(settings)
    bm25 = registered_bm25(settings)
    natural = question_sets(settings, "dev")[DECIDING_SET]
    window = settings.vectors.window
    start = settings.passages.target_words
    grid: list[HybridPoint] = []
    sizes = []
    for words in (start, *settings.passages.trial_targets):
        path = passages_path(settings.data.search, words, start)
        table = load_passage_table(path)
        found = candidate_vectors(settings, f"full-{words}", candidate, table)
        if found is None:
            raise SystemExit(
                f"{candidate} has no whole run over full-{words}. Run steps 06 and 07."
            )
        index, queries, _ = found
        masks = FilterMasks(table)
        lexical = lexical_index_at(settings, bm25, words, path)
        words_hits = lexical_hits(lexical, bm25.params, masks, natural, window)
        vector = vector_hits(index, masks, natural, queries, window)
        years, parties = speech_years(table), speech_parties(table)
        sizes.append(
            SizeReport(
                target_words=words,
                passages=len(table),
                vectors_recall_10=mean_of(
                    score_rankings(table, natural, vector, years, parties), "recall@10"
                ),
                bm25_recall_10=mean_of(
                    score_rankings(table, natural, words_hits, years, parties), "recall@10"
                ),
            )
        )
        grid.extend(
            tune_grid(table, natural, words_hits, vector, words, settings.vectors.rrf_k, window)
        )
        logger.info(
            "%d words: %s",
            words,
            ", ".join(f"k {p.k:g} {p.recall_10:.3f}" for p in grid[-len(settings.vectors.rrf_k) :]),
        )
    chosen = choose_hybrid(grid, start, settings.vectors.rrf_k_preferred)
    logger.info(
        "Chosen: %d words, k %g, recall@10 %.3f", chosen.target_words, chosen.k, chosen.recall_10
    )
    write_report(
        TuningReport(
            candidate=candidate,
            tuned_on=DECIDING_SET,
            questions=len(natural),
            bm25=bm25,
            window=window,
            sizes=tuple(sizes),
            grid=tuple(grid),
            chosen=chosen,
        ),
        settings.tracked.reports / TUNING_REPORT,
    )


def embed_test_questions(
    settings: Settings, candidate: str, questions: list[EvalQuestion], env_file: Path | None
) -> tuple[QueryVectors, dict[str, float | int | str]]:
    """Embed the test questions with the chosen model, here and now."""
    settings_candidate = settings.vectors.candidate(candidate)
    key = None
    if settings_candidate.kind == "openrouter":
        key = load_openrouter_key(env_file or settings.data.search.parents[1] / ".env")
    embedder = build_embedder(
        settings_candidate,
        key=key,
        cap_usd=settings.vectors.run_cap_usd,
        batch_inputs=settings.vectors.api_batch_inputs,
        concurrency=settings.vectors.api_concurrency,
    )
    vectors = embedder.embed([question.text for question in questions], "query")
    usage = embedder.take_usage()
    record: dict[str, float | int | str] = {
        "model": settings_candidate.model,
        "revision": settings_candidate.revision,
        "texts": usage.texts,
        "truncated": usage.truncated,
        "tokens": usage.tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }
    return QueryVectors.of([question.question_id for question in questions], vectors), record


def run_test(settings: Settings, env_file: Path | None) -> None:
    """The one run on the test split."""
    out = settings.tracked.reports / TEST_REPORT
    if out.exists():
        raise SystemExit(f"{out} exists. The test split is run once; a rerun needs an addendum.")
    sets_sha256 = require_sets_registered(settings.tracked.questions)
    tuning_path = settings.tracked.reports / TUNING_REPORT
    require_committed(tuning_path)
    tuning_sha256 = text_sha256(tuning_path)
    require_registered([tuning_sha256], "the tuning's report")
    tuning = TuningReport.model_validate_json(tuning_path.read_text(encoding="utf-8"))
    if tuning.candidate != chosen_candidate(settings):
        raise SystemExit("The tuning's model is not the one the comparison chose.")
    chosen = tuning.chosen
    start = settings.passages.target_words
    window = settings.vectors.window
    bm25 = registered_bm25(settings)
    questions = question_sets(settings, "test")

    base_path = passages_path(settings.data.search, start, start)
    base_table = load_passage_table(base_path)
    base_lexical = lexical_index_at(settings, bm25, start, base_path)
    size_path = passages_path(settings.data.search, chosen.target_words, start)
    size_table = base_table if size_path == base_path else load_passage_table(size_path)
    size_lexical = (
        base_lexical
        if size_path == base_path
        else lexical_index_at(settings, bm25, chosen.target_words, size_path)
    )
    index, _ = load_vector_index(
        index_path(settings.data.search, f"full-{chosen.target_words}", tuning.candidate),
        size_table.passage_ids,
    )
    every = [question for qs in questions.values() for question in qs]
    queries, query_record = embed_test_questions(settings, tuning.candidate, every, env_file)

    seed, resamples = settings.baseline.bootstrap_seed, settings.baseline.resamples
    results: dict[str, dict[str, list[QuestionResult]]] = {"bm25": {}, "vectors": {}, "hybrid": {}}
    base_masks, size_masks = FilterMasks(base_table), FilterMasks(size_table)
    base_years, base_parties = speech_years(base_table), speech_parties(base_table)
    size_years, size_parties = speech_years(size_table), speech_parties(size_table)
    for name, qs in questions.items():
        results["bm25"][name] = score_rankings(
            base_table,
            qs,
            lexical_hits(base_lexical, bm25.params, base_masks, qs, window),
            base_years,
            base_parties,
        )
        vector = vector_hits(index, size_masks, qs, queries, window)
        results["vectors"][name] = score_rankings(size_table, qs, vector, size_years, size_parties)
        words_hits = lexical_hits(size_lexical, bm25.params, size_masks, qs, window)
        results["hybrid"][name] = score_rankings(
            size_table,
            qs,
            fused_hits(words_hits, vector, chosen.k, window),
            size_years,
            size_parties,
        )

    comparisons = [
        compare(
            DECIDING_SET,
            "hybrid",
            results["hybrid"][DECIDING_SET],
            "bm25",
            results["bm25"][DECIDING_SET],
            seed,
            resamples,
        )
    ]
    comparisons.extend(
        compare(
            name,
            "vectors",
            results["vectors"][name],
            "bm25",
            results["bm25"][name],
            seed,
            resamples,
        )
        for name in VECTOR_COMPARISON_SETS
    )
    decision = decide(comparisons[0])
    production = "hybrid" if decision.outcome == "A" else "bm25"
    sets = {
        system: {
            name: summarize_set(system_results, seed, resamples)
            for name, system_results in by_set.items()
        }
        for system, by_set in results.items()
    }
    natural = sets[production][DECIDING_SET]
    deviating = deviating_parties(
        {party: summary.means["recall@10"] for party, summary in natural.by_party.items()},
        natural.overall.means["recall@10"],
    )
    report = TestReport(
        split="test",
        sets_sha256={name: sets_sha256[name] for name in SCORED_SETS},
        tuning_sha256=tuning_sha256,
        candidate=tuning.candidate,
        target_words=chosen.target_words,
        k=chosen.k,
        bm25=bm25,
        query_embedding=query_record,
        resamples=resamples,
        bootstrap_seed=seed,
        comparisons=tuple(comparisons),
        decision=decision,
        production=production,
        deviating_parties=deviating,
        sets=sets,
    )
    lines = [
        json.dumps({"system": system, "set": name, **json.loads(result.model_dump_json())})
        for system, by_set in results.items()
        for name, system_results in by_set.items()
        for result in system_results
    ]
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    (settings.tracked.reports / TEST_QUESTIONS).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    for comparison in comparisons:
        logger.info(
            "%s against %s, %s, %s: %+.3f [%.3f, %.3f]",
            comparison.system,
            comparison.against,
            comparison.question_set,
            comparison.metric,
            comparison.difference,
            *comparison.interval,
        )
    logger.info("Outcome %s: %s runs", decision.outcome, production)


def main() -> None:
    """Parse the command line and run the command it names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("compare", help="Step A: compare the candidates on the subset.")
    commands.add_parser("tune", help="Tune the hybrid on the development split.")
    testing = commands.add_parser("test", help="The one run on the test split.")
    testing.add_argument("--confirm", action="store_true", help="Required: run the test split.")
    testing.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    if args.command == "compare":
        run_compare(settings)
    elif args.command == "tune":
        run_tune(settings)
    else:
        if not args.confirm:
            raise SystemExit("The test split is run once. Pass --confirm to run it.")
        run_test(settings, args.env_file)


if __name__ == "__main__":
    main()
