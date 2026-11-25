"""The figures the method section of the page shows, read from the tracked reports.

The site writes no figure by hand: every number in the method section comes
from the file this module writes, and every number in that file comes from a
report under ``data/reports/search/`` or from the settings. Each source is
recorded with the SHA-256 of its text, so the page can be traced back to the
reports it was built from.

The report on the answer step's test run, ``answer_test.json``, is written only
once every quotation of the drawn questions has been graded by hand. Until it
exists the file says so, and the page says the result is pending. The same
holds for ``answer_test_sealed.json``, which describes the first run on the test
split once that report is committed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from search.config import Settings
from search.parties import PARTY_ORDER

#: The reports the method section reads, all under ``data/reports/search/``.
REPORTS = (
    "anforanden_build.json",
    "passages.json",
    "reconciliation.json",
    "deep_links.json",
    "questions/questions_manifest.json",
    "review_generation_1.json",
    "judge_agreement.json",
    "vector_comparison_dev.json",
    "hybrid_tuning_dev.json",
    "retrieval_test.json",
    "service_check_dev.json",
    "answer_dev.json",
    "answer_test_run.json",
    "answer_test_run_hybrid.json",
    "answer_test_hand_draw.json",
    "page_example_links.json",
    "source_dashes.json",
)

#: Reports that exist only once Martin has graded the drawn quotations.
LATER_REPORTS = ("answer_test.json", "answer_test_sealed.json")

#: The question sets, in the order the page lists them.
SET_ORDER = ("natural", "keyword", "interpellation", "filtered", "stance", "unanswerable")

#: The sets the retrieval run scored.
RETRIEVAL_SETS = ("natural", "keyword", "interpellation", "filtered", "stance")

#: Rules the pre-registration fixes, which the page states beside the results:
#: the least lead of the hybrid over BM25 that puts it in production, the
#: deviation from the mean at which a party is named on the method page, and how
#: long the service waits for a question's vector before BM25 answers alone.
MIN_HYBRID_LEAD = 0.03
PARTY_MARGIN = 0.05
VECTOR_LIMIT_S = 30.0

#: The shorter wait the addendum on the tuned configuration compares the limit with.
SHORT_WAIT_S = 3

#: How the system is built, as the pre-registration and its addenda fix it: the
#: passages each question gets, the most quotations an answer holds, the prompt
#: rounds the answer step was allowed, the search service's processors, and the
#: level of every interval.
CANDIDATES = 10
MAX_QUOTES = 4
PROMPT_ROUNDS = 3
SERVICE_VCPUS = 1
INTERVAL_LEVEL = 0.95

#: The launch rule of the pre-registration: the least share of unanswerable
#: questions the system abstains on, the most share of answerable ones it
#: abstains on needlessly, and the least quote precision.
LAUNCH_ABSTAINS = 0.90
LAUNCH_NEEDLESS = 0.15
LAUNCH_PRECISION = 0.85


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Design(_Frozen):
    """How the system is built, as the pre-registration fixes it."""

    candidates: int
    max_quotes: int
    prompt_rounds: int
    service_vcpus: int
    interval_level: float


class Corpus(_Frozen):
    """The search corpus: what it holds and how it was checked."""

    speeches: int
    first_date: date
    last_date: date
    paragraphs: int
    words: int
    passages: int
    passage_target_words: int
    passage_median_words: float
    left_out_session: str
    left_out_hard_wrapped: int
    left_out_other_parties: int
    set_aside_paragraphs: int
    set_aside_rules: int
    coverage_paired: int
    coverage_of: int
    coverage: float
    coverage_threshold: float
    links_sampled: int
    links_passed: int


class QuestionSet(_Frozen):
    """One question set and its split."""

    name: str
    questions: int
    dev: int
    test: int


class Questions(_Frozen):
    """The question sets and the review of the generated ones."""

    sets: tuple[QuestionSet, ...]
    model: str
    review_approved: int
    review_drawn: int
    review_threshold: float


class Judge(_Frozen):
    """The language model as a judge, and why its grades are not used."""

    pairs: int
    kappa: float
    threshold: float
    accepted: bool
    exact: float
    disagreements: int
    stricter: int


class EmbeddingScore(_Frozen):
    """One embedding model's recall@10 on the natural questions of the development subset."""

    name: str
    recall_10: float
    licensed: bool


class Comparison(_Frozen):
    """One pre-registered comparison on the test split."""

    question_set: str
    system: str
    against: str
    questions: int
    system_mean: float
    against_mean: float
    difference: float
    low: float
    high: float


class SetScores(_Frozen):
    """recall@10 of each system on one question set of the test split."""

    name: str
    questions: int
    bm25: float
    vectors: float
    hybrid: float


class PartyRecall(_Frozen):
    """The hybrid's recall@10 on the natural questions of one party's speeches."""

    party: str
    questions: int
    recall_10: float
    deviation: float


class Retrieval(_Frozen):
    """How the search was chosen and how it did on the test split."""

    embedding: str
    embedding_model: str
    provider: str
    precision: str
    dimensions: int
    subset_passages: int
    embedding_scores: tuple[EmbeddingScore, ...]
    passage_words: int
    rrf_k: float
    window: int
    tuned_recall_10: float
    bm25_k1: float
    bm25_b: float
    outcome: str
    production: str
    min_difference: float
    comparisons: tuple[Comparison, ...]
    sets: tuple[SetScores, ...]
    natural_mean: float
    parties: tuple[PartyRecall, ...]
    party_margin: float


class Round(_Frozen):
    """One round of the answer step on the development split, or the baseline."""

    name: str
    balanced_accuracy: float
    abstains_unanswerable: float
    needless_abstention: float


class AnswerDev(_Frozen):
    """The answer step on the development split."""

    answerable: int
    unanswerable: int
    rounds: tuple[Round, ...]
    chosen_round: int


class SelectorResult(_Frozen):
    """One selector on the test split, its quotations graded by hand."""

    name: str
    balanced_accuracy: float
    abstains_unanswerable: float
    needless_abstention: float
    quote_precision: float | None
    graded_quotes: int
    parties: dict[str, float]


class AnswerTest(_Frozen):
    """The test report of the answer step, once the hand grades are in."""

    drawn_questions: int
    graded_pairs: int
    selectors: tuple[SelectorResult, ...]
    precision_difference: tuple[float, float, float] | None
    accuracy_difference: tuple[float, float, float]
    beats_baseline: bool
    runs: str
    launch: bool
    launch_rule: dict[str, bool]
    candidate_parties: dict[str, float]


class AnswerRuns(_Frozen):
    """The runs of the answer step on the test split."""

    model: str
    effort: str
    questions: int
    first_retrieval: str
    first_cost_usd: float
    hybrid_cost_usd: float
    hand_questions: int
    hand_pairs: int
    launch_rule: LaunchRule
    report: AnswerTest | None
    sealed_balanced_accuracy: dict[str, float] | None


class Links(_Frozen):
    """The check of the links the page's examples show, and why two failed.

    Attributes:
        speeches: How many speeches the examples link to.
        passed: How many pages passed the deep-link check.
        passed_with_folded_dashes: How many pass once the page's en dashes
            are written as hyphens.
        xml_records: How many pages came back as the open-data XML record.
        dashless_from: The first session of the corpus whose dataset holds no en dash.
        dashless_to: The last such session.
    """

    speeches: int
    passed: int
    passed_with_folded_dashes: int
    xml_records: int
    dashless_from: str
    dashless_to: str


class LaunchRule(_Frozen):
    """The launch rule, as the pre-registration fixes it."""

    abstains_unanswerable: float
    needless_abstention: float
    quote_precision: float


class Operations(_Frozen):
    """What the service measured, and what the page's budget buys."""

    vector_questions: int
    vector_wait_p50_s: float
    vector_wait_p95_s: float
    vector_wait_max_s: float
    vector_wait_short_s: int
    vector_wait_over_short: float
    vector_limit_s: float
    search_p95_ms: float
    search_p95_budget_ms: float
    cold_start_s: float
    cold_start_budget_s: float
    memory_peak_gib: float
    memory_budget_gib: float
    monthly_budget_usd: float
    cost_per_question_usd: float
    questions_per_month: int
    party_views_per_month: int


class MethodFigures(_Frozen):
    """Everything the method section shows, with the reports it came from."""

    sources: dict[str, str]
    design: Design
    corpus: Corpus
    questions: Questions
    judge: Judge
    retrieval: Retrieval
    answer_dev: AnswerDev
    answer_test: AnswerRuns
    operations: Operations
    links: Links


def text_sha256(path: Path) -> str:
    """The SHA-256 of a report's text, with Windows line ends read as Unix ones."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load(reports: Path, name: str, sources: dict[str, str]) -> Any:
    path = reports / name
    sources[name] = text_sha256(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus(build: Any, passages: Any, reconciliation: Any, links: Any) -> Corpus:
    sessions = [entry for entry in build["sessions"] if entry["speeches"]]
    wrapped = [
        entry["session"]
        for entry in build["sessions"]
        if entry["excluded"].get("hard_wrapped_source", 0)
    ]
    if len(wrapped) != 1:
        raise ValueError(f"Expected one session left out as hard-wrapped, found {wrapped}.")
    return Corpus(
        speeches=build["speeches"],
        first_date=min(date.fromisoformat(entry["first_date"]) for entry in sessions),
        last_date=max(date.fromisoformat(entry["last_date"]) for entry in sessions),
        paragraphs=build["paragraphs"],
        words=build["words"],
        passages=passages["passages"],
        passage_target_words=passages["target_words"],
        passage_median_words=passages["words_per_passage"]["median"],
        left_out_session=wrapped[0],
        left_out_hard_wrapped=build["excluded"]["hard_wrapped_source"],
        left_out_other_parties=build["excluded"]["outside_the_eight_parties"],
        set_aside_paragraphs=sum(build["set_aside"].values()),
        set_aside_rules=len(build["set_aside"]),
        coverage_paired=reconciliation["coverage"]["numerator"],
        coverage_of=reconciliation["coverage"]["denominator"],
        coverage=reconciliation["coverage"]["share"],
        coverage_threshold=reconciliation["coverage_threshold"],
        links_sampled=links["totals"]["sampled"],
        links_passed=links["totals"]["passed"],
    )


def _questions(manifest: Any, review: Any) -> Questions:
    return Questions(
        sets=tuple(
            QuestionSet(
                name=name,
                questions=manifest["sets"][name]["questions"],
                dev=manifest["sets"][name]["dev"],
                test=manifest["sets"][name]["test"],
            )
            for name in SET_ORDER
        ),
        model=manifest["model"],
        review_approved=review["approved"],
        review_drawn=len(review["verdicts"]),
        review_threshold=review["threshold"],
    )


def _judge(agreement: Any) -> Judge:
    stats = agreement["agreement"]
    table = stats["table"]
    disagreements = sum(count for key, count in table.items() if key[0] != key[-1])
    return Judge(
        pairs=stats["pairs"],
        kappa=stats["kappa"],
        threshold=stats["threshold"],
        accepted=stats["accepted"],
        exact=stats["exact"],
        disagreements=disagreements,
        stricter=table.get("2-1", 0) + table.get("2-0", 0) + table.get("1-0", 0),
    )


def _retrieval(settings: Settings, vectors: Any, tuning: Any, test: Any) -> Retrieval:
    candidate = settings.vectors.candidate(test["candidate"])
    by_name = {entry["name"]: entry for entry in vectors["scores"]}
    order = [entry.name for entry in settings.vectors.candidates]
    natural = test["sets"]["hybrid"]["natural"]
    mean = natural["overall"]["means"]["recall@10"]
    return Retrieval(
        embedding=test["candidate"],
        embedding_model=candidate.model,
        provider=str(candidate.provider),
        precision=str(candidate.quantization),
        dimensions=candidate.dimension,
        subset_passages=vectors["subset_passages"],
        embedding_scores=tuple(
            EmbeddingScore(
                name=name,
                recall_10=by_name[name]["recall_10"],
                licensed=by_name[name]["licensed"],
            )
            for name in order
            if name in by_name
        ),
        passage_words=test["target_words"],
        rrf_k=test["k"],
        window=tuning["window"],
        tuned_recall_10=tuning["chosen"]["recall_10"],
        bm25_k1=test["bm25"]["k1"],
        bm25_b=test["bm25"]["b"],
        outcome=test["decision"]["outcome"],
        production=test["production"],
        min_difference=MIN_HYBRID_LEAD,
        comparisons=tuple(
            Comparison(
                question_set=entry["question_set"],
                system=entry["system"],
                against=entry["against"],
                questions=entry["questions"],
                system_mean=entry["system_mean"],
                against_mean=entry["against_mean"],
                difference=entry["difference"],
                low=entry["interval"][0],
                high=entry["interval"][1],
            )
            for entry in test["comparisons"]
        ),
        sets=tuple(
            SetScores(
                name=name,
                questions=test["sets"]["hybrid"][name]["overall"]["questions"],
                bm25=test["sets"]["bm25"][name]["overall"]["means"]["recall@10"],
                vectors=test["sets"]["vectors"][name]["overall"]["means"]["recall@10"],
                hybrid=test["sets"]["hybrid"][name]["overall"]["means"]["recall@10"],
            )
            for name in RETRIEVAL_SETS
        ),
        natural_mean=mean,
        parties=tuple(
            PartyRecall(
                party=party,
                questions=natural["by_party"][party]["questions"],
                recall_10=natural["by_party"][party]["means"]["recall@10"],
                deviation=test["deviating_parties"][party],
            )
            for party in PARTY_ORDER
            if party in test["deviating_parties"]
        ),
        party_margin=PARTY_MARGIN,
    )


def _links(build: Any, links: Any, dashes: Any) -> Links:
    in_corpus = [entry["session"] for entry in build["sessions"] if entry["speeches"]]
    dashless = {entry["session"] for entry in dashes["sessions"] if not entry["with_en_dash"]}
    run = [session for session in in_corpus if session in dashless]
    if not run:
        raise ValueError("Every session of the corpus holds an en dash.")
    span = in_corpus[in_corpus.index(run[0]) : in_corpus.index(run[-1]) + 1]
    if span != run:
        raise ValueError("The sessions without an en dash are not one run.")
    return Links(
        speeches=links["speeches"],
        passed=links["totals"]["passed"],
        passed_with_folded_dashes=links["passed_with_folded_dashes"],
        xml_records=links["xml_records"],
        dashless_from=run[0],
        dashless_to=run[-1],
    )


def _answer_dev(report: Any) -> AnswerDev:
    rounds = tuple(
        Round(
            name=entry["name"],
            balanced_accuracy=entry["abstention"]["balanced_accuracy"],
            abstains_unanswerable=entry["abstention"]["abstains_unanswerable"],
            needless_abstention=entry["abstention"]["needless_abstention"],
        )
        for entry in report["selectors"]
    )
    return AnswerDev(
        answerable=report["threshold"]["answerable"],
        unanswerable=report["threshold"]["unanswerable"],
        rounds=rounds,
        chosen_round=report["leading_round"],
    )


def _answer_test(report: Any) -> AnswerTest:
    selectors = tuple(
        SelectorResult(
            name=entry["name"],
            balanced_accuracy=entry["abstention"]["balanced_accuracy"],
            abstains_unanswerable=entry["abstention"]["abstains_unanswerable"],
            needless_abstention=entry["abstention"]["needless_abstention"],
            quote_precision=entry["quote_precision"],
            graded_quotes=entry["graded_quotes"],
            parties=entry["parties"],
        )
        for entry in report["selectors"]
    )
    running = next(entry for entry in report["selectors"] if entry["name"] == report["runs"])
    comparison = report["comparison"]
    precision = comparison["quote_precision"]
    return AnswerTest(
        drawn_questions=report["drawn_questions"],
        graded_pairs=report["graded_pairs"],
        selectors=selectors,
        precision_difference=None if precision is None else tuple(precision),
        accuracy_difference=tuple(comparison["balanced_accuracy"]),
        beats_baseline=report["beats_baseline"],
        runs=report["runs"],
        launch=report["launch"],
        launch_rule=running["launch_rule"],
        candidate_parties=report["candidate_parties"],
    )


def _operations(settings: Settings, check: Any, hybrid_run: Any) -> Operations:
    waiting = check["queries"]["waiting"]
    storage = check["storage"]
    discount = settings.generation.batch_discount
    questions = hybrid_run["retrieval"]["questions"]
    per_question = hybrid_run["cost_usd"] / questions / (1 - discount)
    budget = settings.answer.monthly_budget_usd
    return Operations(
        vector_questions=check["queries"]["questions"],
        vector_wait_p50_s=waiting["quantiles_s"]["p50"],
        vector_wait_p95_s=waiting["quantiles_s"]["p95"],
        vector_wait_max_s=waiting["max_s"],
        vector_wait_short_s=SHORT_WAIT_S,
        vector_wait_over_short=round(1 - waiting["within"][str(SHORT_WAIT_S)], 4),
        vector_limit_s=VECTOR_LIMIT_S,
        search_p95_ms=storage["p95_ms"],
        search_p95_budget_ms=storage["budget"]["p95_ms"],
        cold_start_s=storage["cold_start_s"],
        cold_start_budget_s=storage["budget"]["cold_start_s"],
        memory_peak_gib=storage["memory_peak_gib"],
        memory_budget_gib=storage["budget"]["memory_gib"],
        monthly_budget_usd=budget,
        cost_per_question_usd=round(per_question, 4),
        questions_per_month=round(budget / per_question),
        party_views_per_month=round(budget / (len(PARTY_ORDER) * per_question)),
    )


def method_figures(settings: Settings) -> MethodFigures:
    """Read every figure the method section shows.

    Args:
        settings: The settings, for the tracked reports' directory, the
            embedding model's provider, the batch discount and the page's
            monthly budget.

    Returns:
        The figures, with the SHA-256 of each report read.

    Raises:
        FileNotFoundError: If a report the section needs does not exist.
    """
    reports = settings.tracked.reports
    sources: dict[str, str] = {}
    read = {name: _load(reports, name, sources) for name in REPORTS}
    later = {
        name: _load(reports, name, sources) for name in LATER_REPORTS if (reports / name).is_file()
    }
    first, hybrid = read["answer_test_run.json"], read["answer_test_run_hybrid.json"]
    draw = read["answer_test_hand_draw.json"]
    sealed = later.get("answer_test_sealed.json")
    return MethodFigures(
        sources=dict(sorted(sources.items())),
        design=Design(
            candidates=CANDIDATES,
            max_quotes=MAX_QUOTES,
            prompt_rounds=PROMPT_ROUNDS,
            service_vcpus=SERVICE_VCPUS,
            interval_level=INTERVAL_LEVEL,
        ),
        corpus=_corpus(
            read["anforanden_build.json"],
            read["passages.json"],
            read["reconciliation.json"],
            read["deep_links.json"],
        ),
        questions=_questions(
            read["questions/questions_manifest.json"], read["review_generation_1.json"]
        ),
        judge=_judge(read["judge_agreement.json"]),
        retrieval=_retrieval(
            settings,
            read["vector_comparison_dev.json"],
            read["hybrid_tuning_dev.json"],
            read["retrieval_test.json"],
        ),
        answer_dev=_answer_dev(read["answer_dev.json"]),
        answer_test=AnswerRuns(
            model=hybrid["model"],
            effort=hybrid["effort"],
            questions=hybrid["retrieval"]["questions"],
            first_retrieval="hybrid" if "embedding" in first["retrieval"] else "bm25",
            first_cost_usd=first["cost_usd"],
            hybrid_cost_usd=hybrid["cost_usd"],
            hand_questions=len(draw["questions"]),
            hand_pairs=len(draw["pairs"]),
            launch_rule=LaunchRule(
                abstains_unanswerable=LAUNCH_ABSTAINS,
                needless_abstention=LAUNCH_NEEDLESS,
                quote_precision=LAUNCH_PRECISION,
            ),
            report=_answer_test(later["answer_test.json"]) if "answer_test.json" in later else None,
            sealed_balanced_accuracy=None
            if sealed is None
            else {
                name: shares["balanced_accuracy"] for name, shares in sealed["abstention"].items()
            },
        ),
        operations=_operations(settings, read["service_check_dev.json"], hybrid),
        links=_links(
            read["anforanden_build.json"],
            read["page_example_links.json"],
            read["source_dashes.json"],
        ),
    )
