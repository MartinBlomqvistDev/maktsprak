"""Step 14: run the answer step on the development split, and once on the test split.

Commands, in order:

- ``candidates`` ranks the ten passages BM25 gives each question of the
  natural, filtered, stance and unanswerable sets, under its filter when it has
  one, and records them. The index must match the corpus on disk.
- ``baseline`` chooses the baseline selector's threshold by balanced accuracy
  and answers every question with it.
- ``estimate`` prints what a round of each model, and the judge's grades still
  to give, would cost. It calls nothing.
- ``round`` answers every question with one model as one message batch, sends
  a failed request once more, and lets the baseline answer where it fails
  again. The round is recorded with its model, its prompt's hash and its
  effort. Round 1 runs Opus 5.5, Sonnet 5.5 and Haiku 5.5 with the same
  prompt, or Opus 5.5 alone if the judge's kappa fell short of 0.6. Rounds 2
  and 3 run only the model chosen after round 1, each with a prompt of its
  own. It needs ``--confirm``.
- ``judge`` grades every quotation given to an answerable question, and every
  candidate of one, that has no grade yet. It needs ``--confirm``.
- ``report`` computes each selector's metrics, the comparisons with the
  baseline and the choice of model from what is recorded, and writes them
  among the tracked reports.
- ``test-run`` answers the test split once with the model and the prompt the
  development split chose, and the baseline with its threshold. It records
  what ran and computes nothing from it: the answers stay sealed. It needs
  ``--max-usd`` and ``--confirm``.

Nothing runs before the question sets' hashes and the addenda on the answer
step stand in the pre-registration as committed at ``HEAD``, and nothing paid
runs before the judge's agreement with the hand grades is known. Only
``test-run`` reads the test split, once, after the addendum on the test run is
committed, and no round runs after it. The paid commands on the development
split stop before a batch that could take what the rounds and the judge have
spent beyond ``answer.dev_budget_usd``, or beyond ``--max-usd`` when that is
lower; the test run stops at its own ``--max-usd``.

Usage::

    python search/pipeline/14_run_answer_step.py candidates
    python search/pipeline/14_run_answer_step.py baseline
    python search/pipeline/14_run_answer_step.py estimate
    python search/pipeline/14_run_answer_step.py round --number 1 --model claude-haiku-5-5 \\
        --confirm
    python search/pipeline/14_run_answer_step.py judge --confirm
    python search/pipeline/14_run_answer_step.py report
    python search/pipeline/14_run_answer_step.py test-run --max-usd 11 --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ConfigDict

from search.answer.assemble import assemble
from search.answer.candidates import (
    CANDIDATES,
    build_candidates,
    check_index,
    load_speeches,
    rank_passages,
)
from search.answer.models import Answer, Candidate, Speech
from search.answer.selector import (
    PROMPT_DIR,
    PROMPT_NAME,
    BaselineSelector,
    ClaudeSelector,
    choose_threshold,
    failed,
    fall_back,
    load_selector_prompt,
)
from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import REPO_ROOT, Settings
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report
from search.corpus.table import PassageTable, load_passage_table
from search.eval.answer_metrics import (
    Abstention,
    QuestionOutcome,
    QuoteOutcome,
    abstention,
    accuracy_difference,
    choose_model,
    choose_round,
    choose_round_by_accuracy,
    party_shares,
    precision_difference,
    quote_precision,
    ungraded,
)
from search.eval.assemble import MANIFEST_NAME, QuestionsManifest
from search.eval.baseline import FilterMasks
from search.eval.generation import append_log, load_prompt, read_log
from search.eval.llm import Completion, cost_usd, load_key, run_batch
from search.eval.questions import EvalQuestion, read_question_set, split_of
from search.eval.quote_judge import (
    GradingPair,
    QuoteGrade,
    check_judge_prompt,
    grades_by_key,
    grading_requests,
    pending,
    read_grades,
    text_sha256,
)
from search.eval.runner import now
from search.index.analyzer import ANALYZERS
from search.index.lexical import Bm25Params, IndexManifest, LexicalIndex, lexical_dir

logger = logging.getLogger("answer_step")

PREREGISTRATION = "docs/search/PREREGISTRATION.md"

#: The heading of the addendum that fixes the answer step.
ADDENDUM = "### 2026-11-22: the answer step, before it is built"

#: The heading of the addendum that runs three models in round 1.
MODELS_ADDENDUM = "### 2026-11-23: three models in the first round"

#: The sets the answer step is run on, as the addendum names them.
ANSWER_SETS = ("natural", "filtered", "stance", "unanswerable")

#: The rounds the addendum allows on the development split.
MAX_ROUNDS = 3

#: The split the test run reads, once, and the addendum it waits for.
TEST_SPLIT = "test"
TEST_ADDENDUM = "### 2026-11-24: the test run of the answer step, graded by hand"
TEST_RUN_REPORT = "answer_test_run.json"

#: The models of round 1, and the one that runs alone if the judge's kappa falls short.
ROUND_ONE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
WITHOUT_THE_JUDGE = "claude-opus-5-5"

#: The effort and the most output tokens every round runs with, as the addenda fix them.
REGISTERED_EFFORT = "low"
REGISTERED_MAX_TOKENS = 8000

TUNING_REPORT = "bm25_tuning_dev.json"
AGREEMENT_REPORT = "judge_agreement.json"
REPORT = "answer_dev.json"
QUESTIONS_REPORT = "answer_dev_questions.jsonl"

CANDIDATES_NAME = "candidates.jsonl"
RUN_NAME = "candidates_run.json"
THRESHOLD_NAME = "baseline_threshold.json"
ANSWERS_DIR = "answers"
ROUNDS_DIR = "rounds"
JUDGE_DIR = "judge"
GRADES_NAME = "grades.jsonl"
ROUND_NAME = "round.json"

#: The input tokens of a request, estimated from its words. Fitted on the
#: judge's 100 calibration requests: 604 tokens of overhead, which holds the
#: judge's own prompt, and 2.63 tokens per word of the passage.
TOKENS_OVERHEAD = 604
TOKENS_PER_WORD = 2.63

#: Output tokens per request, thinking included, assumed for an estimate: low,
#: middle and high. The judge used 127 on average at low effort.
SELECTOR_OUTPUT = (300, 600, 1500)
JUDGE_OUTPUT = (100, 130, 250)


class QuestionCandidates(BaseModel):
    """One question and the candidates its answer is chosen from.

    Attributes:
        question_id: The question's key.
        question_set: Its set.
        question: Its text.
        answerable: Whether it has a gold speech.
        candidates: The passages BM25 ranked highest, best first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    question_set: str
    question: str
    answerable: bool
    candidates: tuple[Candidate, ...]


class CandidatesRun(BaseModel):
    """What the candidates were ranked with.

    Attributes:
        analyzer: The index's analyzer.
        k1: BM25's ``k1``.
        b: BM25's ``b``.
        limit: The most candidates per question.
        passages: How many passages the index holds.
        passage_ids_sha256: The hash of the index's passage ids.
        passages_sha256: The hash of the passages file.
        speeches_sha256: The hash of the speeches file.
        sets_sha256: The hash of each question set read.
        questions: How many questions.
        created: When the candidates were ranked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyzer: str
    k1: float
    b: float
    limit: int
    passages: int
    passage_ids_sha256: str
    passages_sha256: str
    speeches_sha256: str
    sets_sha256: dict[str, str]
    questions: int
    created: datetime


class ThresholdChoice(BaseModel):
    """The baseline selector's threshold, chosen by balanced accuracy.

    Attributes:
        threshold: The share of the highest possible score below which it abstains.
        balanced_accuracy: The balanced accuracy the threshold reaches.
        answerable: How many answerable questions it was chosen on.
        unanswerable: How many unanswerable ones.
        without_candidates: How many questions had no candidate at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float
    balanced_accuracy: float
    answerable: int
    unanswerable: int
    without_candidates: int


class RoundRecord(BaseModel):
    """One round of the language model on the development split.

    Attributes:
        number: The round, from 1.
        model: The model.
        effort: Its effort.
        max_tokens: The most output tokens of one answer.
        prompt_sha256: The hash of the prompt file, a copy of which is kept beside the record.
        batches: The message batches, by attempt: ``first`` and ``again``.
        created: When the round began.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int
    model: str
    effort: str
    max_tokens: int
    prompt_sha256: str
    batches: dict[str, str] = {}
    created: datetime


class SealedRun(BaseModel):
    """The one run of the answer step on the test split: what ran, and nothing it measured.

    The run's results stay sealed until the retrieval run on the test split
    has been made, so this record holds no metric.

    Attributes:
        model: The model that selected.
        round: The development round whose prompt ran.
        prompt_sha256: That prompt's hash.
        effort: The model's effort.
        max_tokens: The most output tokens of one answer.
        threshold: The baseline's threshold, chosen on the development split.
        retrieval: What the test questions' candidates were ranked with.
        batches: The message batches, by attempt.
        answers_sha256: The hash of each answers file, by selector.
        cost_usd: What the requests cost.
        created: When the run ended.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    round: str
    prompt_sha256: str
    effort: str
    max_tokens: int
    threshold: float
    retrieval: CandidatesRun
    batches: dict[str, str]
    answers_sha256: dict[str, str]
    cost_usd: float
    created: datetime


class AnsweredQuestion(BaseModel):
    """A line of an answers file: one question and its answer.

    Attributes:
        question_id: The question.
        answer: The answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    answer: Answer


class SelectorSummary(BaseModel):
    """How one selector, or one round, did on the development split.

    Attributes:
        name: ``baseline`` or a round's name, such as ``round-1-opus-5-5``.
        model: The round's model.
        effort: The round's effort.
        prompt_sha256: The round's prompt.
        abstention: Balanced accuracy and its shares.
        quote_precision: The share of quotations on answerable questions graded
            2, or ``None`` while some of them still wait for the judge.
        score: The mean of quote precision and balanced accuracy, or ``None``
            without a quote precision.
        quotes: How many quotations on answerable questions.
        pending_grades: How many of them still wait for the judge.
        ungraded: How many of them have no grade.
        quotes_unanswerable: How many quotations were given to unanswerable questions.
        rejected: The rejected selections, by reason.
        fallbacks: The questions the baseline answered, by reason.
        cost_usd: What the round's requests cost.
        cost_per_question_usd: That cost over every question of the split.
        parties: The share of each party among the quotations on answerable questions.
        launch_rule: Whether each part of the launch rule holds on the point estimates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    model: str | None = None
    effort: str | None = None
    prompt_sha256: str | None = None
    abstention: Abstention
    quote_precision: float | None
    score: float | None
    quotes: int
    pending_grades: int
    ungraded: int
    quotes_unanswerable: int
    rejected: dict[str, int]
    fallbacks: dict[str, int]
    cost_usd: float
    cost_per_question_usd: float
    parties: dict[str, float]
    launch_rule: dict[str, bool]


class Comparison(BaseModel):
    """One round against the baseline: differences with their paired 95% intervals.

    Attributes:
        name: The round.
        quote_precision: The difference, round minus baseline, and the interval's
            bounds; ``None`` while either has quotations that wait for the judge.
        balanced_accuracy: The same for balanced accuracy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    quote_precision: tuple[float, float, float] | None
    balanced_accuracy: tuple[float, float, float]


class AnswerReport(BaseModel):
    """The answer step on the development split. Description: nothing here decides.

    Attributes:
        split: The split run.
        sets_sha256: The hash of each question set read.
        retrieval: What the candidates were ranked with.
        threshold: The baseline's threshold.
        judge_prompt_sha256: The judge's prompt.
        judge_accepted: Whether the judge's kappa reached 0.6, or ``None`` before it is known.
        selectors: Each selector and round.
        comparisons: Each round against the baseline.
        candidate_parties: The share of each party among the candidates of the
            answerable questions.
        answerable_with_a_grade_2: How many answerable questions have at least
            one candidate the judge grades 2, and how many have a graded candidate.
        chosen_model: The model chosen after round 1, or ``None`` before round 1
            is answered by each of its models and graded.
        leading_round: Among the chosen model's rounds, the one with the highest
            mean of quote precision and balanced accuracy so far, or the highest
            balanced accuracy if the judge's kappa fell short.
        cost_usd: What the rounds and the judge have cost, by part.
        resamples: The bootstrap resamples behind each interval.
        bootstrap_seed: The seed of the resamples.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    sets_sha256: dict[str, str]
    retrieval: CandidatesRun
    threshold: ThresholdChoice
    judge_prompt_sha256: str
    judge_accepted: bool | None
    selectors: tuple[SelectorSummary, ...]
    comparisons: tuple[Comparison, ...]
    candidate_parties: dict[str, float]
    answerable_with_a_grade_2: tuple[int, int]
    chosen_model: str | None
    leading_round: int | None
    cost_usd: dict[str, float]
    resamples: int
    bootstrap_seed: int


# --- The registration -------------------------------------------------------------


def committed_preregistration() -> str:
    """The pre-registration as committed at ``HEAD``."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{PREREGISTRATION}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8")


def check_registered(settings: Settings) -> QuestionsManifest:
    """Stop unless the sets match their hashes, and the hashes and the addenda are committed.

    Returns:
        The manifest of the question sets.
    """
    path = settings.tracked.questions / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"{path} does not exist.")
    manifest = QuestionsManifest.model_validate_json(path.read_text(encoding="utf-8"))
    registered = committed_preregistration()
    if ADDENDUM not in registered:
        raise SystemExit(f"The addendum on the answer step is not in {PREREGISTRATION} at HEAD.")
    if MODELS_ADDENDUM not in registered:
        raise SystemExit(f"The addendum on three models is not in {PREREGISTRATION} at HEAD.")
    for name in ANSWER_SETS:
        record = manifest.sets[name]
        actual = hashlib.sha256((settings.tracked.questions / record.file).read_bytes()).hexdigest()
        if actual != record.sha256:
            raise SystemExit(f"{record.file} no longer matches its hash in the manifest.")
        if record.sha256 not in registered:
            raise SystemExit(f"The hash of {name} is not in the committed {PREREGISTRATION}.")
    return manifest


def split_questions(settings: Settings, split: str) -> list[EvalQuestion]:
    """One split's questions of the four sets, set by set."""
    questions: list[EvalQuestion] = []
    for name in ANSWER_SETS:
        questions.extend(
            split_of(read_question_set(settings.tracked.questions / f"{name}.jsonl"), split)
        )
    return questions


def dev_questions(settings: Settings) -> list[EvalQuestion]:
    """The development split's questions of the four sets."""
    return split_questions(settings, "dev")


def held_out_questions(settings: Settings) -> list[EvalQuestion]:
    """The test split's questions of the four sets, read only by the test run."""
    return split_questions(settings, TEST_SPLIT)


# --- Files ------------------------------------------------------------------------


def work_dir(settings: Settings, split: str = "dev") -> Path:
    """Where the step's bulk data lives: the candidates, answers, completions and grades."""
    return settings.data.search / "answer" / split


def run_name(number: int, model: str) -> str:
    """A round's name: its number and its model, as in ``round-1-opus-5-5``."""
    return f"round-{number}-{model.removeprefix('claude-')}"


def round_dir(settings: Settings, name: str) -> Path:
    """Where one round's record, prompt and completions live."""
    return work_dir(settings) / ROUNDS_DIR / name


def answers_path(settings: Settings, name: str, split: str = "dev") -> Path:
    """The answers of one selector or round: ``baseline``, a round's name or ``selector``."""
    return work_dir(settings, split) / ANSWERS_DIR / f"{name}.jsonl"


def write_lines(path: Path, entries: Sequence[BaseModel]) -> None:
    """Write a JSON Lines file whole, one entry per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def read_candidates(
    settings: Settings, split: str = "dev"
) -> tuple[CandidatesRun, list[QuestionCandidates]]:
    """The recorded candidates and what they were ranked with."""
    directory = work_dir(settings, split)
    run_path = directory / RUN_NAME
    if not run_path.is_file():
        raise SystemExit("No candidates yet. Run the candidates command first.")
    run = CandidatesRun.model_validate_json(run_path.read_text(encoding="utf-8"))
    return run, read_log(directory / CANDIDATES_NAME, QuestionCandidates)


def read_answers(settings: Settings, name: str, split: str = "dev") -> dict[str, Answer]:
    """The answers of one selector or round, by question."""
    path = answers_path(settings, name, split)
    if not path.is_file():
        raise SystemExit(f"{path} does not exist.")
    return {line.question_id: line.answer for line in read_log(path, AnsweredQuestion)}


def by_price(settings: Settings, models: Sequence[str]) -> list[str]:
    """The models from the cheapest to the most expensive, by list price."""
    prices = settings.generation.prices
    return sorted(models, key=lambda model: (prices[model].input, prices[model].output))


def round_records(settings: Settings) -> list[RoundRecord]:
    """Every round begun, answered or not: by number, and within a number by price."""
    records = [
        RoundRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (work_dir(settings) / ROUNDS_DIR).glob(f"*/{ROUND_NAME}")
    ]
    order = by_price(settings, ROUND_ONE_MODELS)
    return sorted(records, key=lambda record: (record.number, order.index(record.model)))


def recorded_rounds(settings: Settings) -> list[RoundRecord]:
    """The rounds with answers, in order."""
    return [
        record
        for record in round_records(settings)
        if answers_path(settings, run_name(record.number, record.model)).is_file()
    ]


# --- Candidates and the baseline -------------------------------------------------


def check_tuned(settings: Settings) -> None:
    """Stop unless the configured retrieval is the one the tuning chose on the development split."""
    tuning = json.loads((settings.tracked.reports / TUNING_REPORT).read_text(encoding="utf-8"))
    chosen = tuning["chosen"]
    answer = settings.answer
    if (chosen["analyzer"], chosen["k1"], chosen["b"]) != (answer.analyzer, answer.k1, answer.b):
        raise SystemExit(
            f"The answer step is set to {answer.analyzer}, k1 {answer.k1}, b {answer.b}; "
            f"the tuning chose {chosen['analyzer']}, k1 {chosen['k1']}, b {chosen['b']}."
        )


def load_retrieval(settings: Settings) -> tuple[LexicalIndex, IndexManifest, PassageTable]:
    """The configured BM25 index and the passages; an index built from another corpus is refused."""
    analyzer = next(item for item in ANALYZERS if item.name == settings.answer.analyzer)
    index, index_manifest = LexicalIndex.load(lexical_dir(settings.data.search, analyzer))
    table = load_passage_table(settings.data.search / PASSAGES_NAME)
    check_index(
        index_manifest,
        table,
        settings.data.search / SPEECHES_NAME,
        settings.data.search / PASSAGES_NAME,
    )
    return index, index_manifest, table


def bm25_params(settings: Settings) -> Bm25Params:
    """The configured ``k1`` and ``b``."""
    return Bm25Params(k1=settings.answer.k1, b=settings.answer.b)


def make_candidates(settings: Settings, manifest: QuestionsManifest, split: str = "dev") -> None:
    """Rank each question's candidates and record them."""
    check_tuned(settings)
    index, index_manifest, table = load_retrieval(settings)
    params = bm25_params(settings)
    masks = FilterMasks(table)
    questions = dev_questions(settings) if split == "dev" else held_out_questions(settings)
    hits = {
        question.question_id: rank_passages(index, question.text, params, masks.of(question))
        for question in questions
    }
    speeches = load_speeches(
        settings.data.search / SPEECHES_NAME,
        {table.anf_ids[hit.passage] for question_hits in hits.values() for hit in question_hits},
    )
    records = [
        QuestionCandidates(
            question_id=question.question_id,
            question_set=question.question_set,
            question=question.text,
            answerable=bool(question.gold),
            candidates=tuple(build_candidates(hits[question.question_id], table, speeches)),
        )
        for question in questions
    ]
    directory = work_dir(settings, split)
    write_lines(directory / CANDIDATES_NAME, records)
    run = CandidatesRun(
        analyzer=settings.answer.analyzer,
        k1=params.k1,
        b=params.b,
        limit=CANDIDATES,
        passages=index_manifest.passages,
        passage_ids_sha256=index_manifest.passage_ids_sha256,
        passages_sha256=index_manifest.passages_sha256,
        speeches_sha256=index_manifest.speeches_sha256,
        sets_sha256={name: manifest.sets[name].sha256 for name in ANSWER_SETS},
        questions=len(records),
        created=now(),
    )
    write_report(run, directory / RUN_NAME)
    counts = Counter(len(record.candidates) for record in records)
    logger.info(
        "%d questions; candidates per question: %s",
        len(records),
        dict(sorted(counts.items())),
    )


def speeches_for(settings: Settings, records: Sequence[QuestionCandidates]) -> dict[str, Speech]:
    """The speeches the candidates belong to, read from the corpus as it is now."""
    return load_speeches(
        settings.data.search / SPEECHES_NAME,
        {candidate.anf_id for record in records for candidate in record.candidates},
    )


def baseline_selector(
    settings: Settings, records: Sequence[QuestionCandidates]
) -> tuple[BaselineSelector, ThresholdChoice]:
    """The baseline selector, with its threshold chosen by balanced accuracy on these questions."""
    index, _, _ = load_retrieval(settings)
    params = bm25_params(settings)
    probe = BaselineSelector(index, params, threshold=0.0)
    scores = {
        record.question_id: probe.relative_score(record.question, record.candidates)
        for record in records
    }
    answerable = [scores[record.question_id] for record in records if record.answerable]
    unanswerable = [scores[record.question_id] for record in records if not record.answerable]
    threshold, accuracy = choose_threshold(answerable, unanswerable)
    choice = ThresholdChoice(
        threshold=threshold,
        balanced_accuracy=round(accuracy, 6),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        without_candidates=sum(score is None for score in scores.values()),
    )
    return BaselineSelector(index, params, threshold), choice


def read_threshold(settings: Settings) -> ThresholdChoice:
    """The baseline's recorded threshold."""
    path = work_dir(settings) / THRESHOLD_NAME
    if not path.is_file():
        raise SystemExit("No baseline yet. Run the baseline command first.")
    return ThresholdChoice.model_validate_json(path.read_text(encoding="utf-8"))


def run_baseline(settings: Settings) -> None:
    """Choose the baseline's threshold, then answer every question with it."""
    _, records = read_candidates(settings)
    selector, choice = baseline_selector(settings, records)
    write_report(choice, work_dir(settings) / THRESHOLD_NAME)
    speeches = speeches_for(settings, records)
    answers = [
        AnsweredQuestion(
            question_id=record.question_id,
            answer=assemble(
                selector.select(record.question, record.candidates), record.candidates, speeches
            ),
        )
        for record in records
    ]
    write_lines(answers_path(settings, "baseline"), answers)
    abstained = sum(line.answer.abstained for line in answers)
    logger.info(
        "Threshold %.4f, balanced accuracy %.3f; the baseline abstains on %d of %d questions",
        choice.threshold,
        choice.balanced_accuracy,
        abstained,
        len(answers),
    )


# --- Cost ------------------------------------------------------------------------


def _discount(settings: Settings) -> float:
    return settings.generation.batch_discount


def spent(settings: Settings) -> dict[str, float]:
    """What the rounds and the judge have cost so far, by part, each round at its model's price."""
    parts: dict[str, float] = {}
    for record in round_records(settings):
        name = run_name(record.number, record.model)
        price = settings.generation.prices[record.model]
        total = sum(
            cost_usd(completion.usage, price, _discount(settings))
            for path in sorted(round_dir(settings, name).glob("completions_*.jsonl"))
            for completion in read_log(path, Completion)
        )
        if total:
            parts[name] = round(total, 4)
    grades = read_log(work_dir(settings) / JUDGE_DIR / GRADES_NAME, QuoteGrade)
    if grades:
        parts["judge"] = round(sum(grade.cost_usd for grade in grades), 4)
    return parts


def estimate_tokens(words: int, overhead_words: int = 0) -> int:
    """The input tokens of a request with *words* words beyond *overhead_words*."""
    return round(TOKENS_OVERHEAD + TOKENS_PER_WORD * max(words - overhead_words, 0))


def judge_overhead_words() -> int:
    """The words of the judge's own prompt, which the fitted overhead already holds."""
    prompt = load_prompt("judge")
    return len((prompt.system + " " + prompt.user.format(question="", passage="")).split())


def request_words(params: Mapping[str, object]) -> int:
    """The words of a request's system prompt and user message."""
    messages = params["messages"]
    assert isinstance(messages, list)
    return len((str(params["system"]) + " " + str(messages[0]["content"])).split())


def batch_costs(
    settings: Settings, model: str, input_tokens: Sequence[int], outputs: Sequence[int]
) -> tuple[float, ...]:
    """What a batch of requests would cost, at each assumed output per request."""
    price = settings.generation.prices[model]
    share = 1 - _discount(settings)
    return tuple(
        (price.input * sum(input_tokens) + price.output * output * len(input_tokens)) / 1e6 * share
        for output in outputs
    )


def round_costs(
    settings: Settings, model: str, requests: Mapping[str, Mapping[str, object]]
) -> tuple[float, tuple[float, ...]]:
    """The mean input tokens of a round's requests, and their cost on *model* at each output."""
    overhead = judge_overhead_words()
    tokens = [estimate_tokens(request_words(params), overhead) for params in requests.values()]
    mean = sum(tokens) / len(tokens) if tokens else 0.0
    return mean, batch_costs(settings, model, tokens, SELECTOR_OUTPUT)


def judge_costs(settings: Settings, pairs: Sequence[GradingPair]) -> tuple[float, ...]:
    """What grading *pairs* would cost, at each assumed output."""
    tokens = [estimate_tokens(len(pair.text.split())) for pair in pairs]
    return batch_costs(settings, settings.judge.model, tokens, JUDGE_OUTPUT)


def check_budget(settings: Settings, high_estimate: float, cap_usd: float | None = None) -> None:
    """Stop if the batch could take the spending beyond the development budget.

    Args:
        settings: The settings, whose ``answer.dev_budget_usd`` is the budget.
        high_estimate: What the batch could cost, at the high assumption of output.
        cap_usd: A lower cap on the spending of the rounds and the judge
            together, as a command was given it, or ``None`` for the budget alone.
    """
    so_far = sum(spent(settings).values())
    cap = settings.answer.dev_budget_usd
    if cap_usd is not None:
        cap = min(cap, cap_usd)
    if so_far + high_estimate > cap:
        raise SystemExit(
            f"Spent {so_far:.2f} USD; this batch could cost up to {high_estimate:.2f} USD, "
            f"beyond the budget of {cap:.2f} USD."
        )


# --- The models and the judge's agreement ------------------------------------------


def judge_accepted(settings: Settings) -> bool | None:
    """Whether the judge's kappa reached the threshold, or ``None`` before it is known."""
    path = settings.tracked.reports / AGREEMENT_REPORT
    if not path.is_file():
        return None
    return bool(json.loads(path.read_text(encoding="utf-8"))["agreement"]["accepted"])


def check_judge_known(settings: Settings) -> None:
    """Stop while the judge's agreement with the hand grades is not known: nothing paid runs."""
    if judge_accepted(settings) is None:
        raise SystemExit(
            f"The judge's agreement is not known yet ({AGREEMENT_REPORT} is missing), "
            "and nothing paid runs before it is."
        )


def check_round_settings(settings: Settings) -> None:
    """Stop unless the configured effort and output limit are the ones the addenda fix."""
    answer = settings.answer
    if (answer.effort, answer.max_tokens) != (REGISTERED_EFFORT, REGISTERED_MAX_TOKENS):
        raise SystemExit(
            f"Every round runs at {REGISTERED_EFFORT} effort with at most "
            f"{REGISTERED_MAX_TOKENS} output tokens; the settings say {answer.effort} "
            f"and {answer.max_tokens}."
        )


def round_one_models(settings: Settings) -> list[str]:
    """The models of round 1, cheapest first: all three, or Opus 5.5 alone without the judge."""
    if judge_accepted(settings) is False:
        return [WITHOUT_THE_JUDGE]
    return by_price(settings, ROUND_ONE_MODELS)


def chosen_model(
    settings: Settings, records: Sequence[QuestionCandidates], log: Sequence[QuoteGrade]
) -> str | None:
    """The model that goes on after round 1, or ``None`` before it can be chosen.

    It can be chosen once the judge's agreement is known, each model of round 1
    has answered, and, when the judge's kappa reached 0.6, every quotation of
    round 1 on an answerable question has had its grade.
    """
    accepted = judge_accepted(settings)
    if accepted is None:
        return None
    models = round_one_models(settings)
    answered = {record.model for record in recorded_rounds(settings) if record.number == 1}
    if not set(models) <= answered:
        return None
    if not accepted:
        return WITHOUT_THE_JUDGE
    grades = grades_by_key(log)
    scores: dict[str, tuple[float | None, float]] = {}
    for model in models:
        answers = read_answers(settings, run_name(1, model))
        if pending(quote_pairs(records, answers), log):
            return None
        outcomes = outcomes_of(records, answers, grades)
        scores[model] = (quote_precision(outcomes), abstention(outcomes).balanced_accuracy)
    return choose_model(scores, models)


def check_round(settings: Settings, number: int, model: str, prompt_sha256: str) -> None:
    """Stop unless the addenda let *model* run round *number* with this prompt.

    Round 1 runs each of its models with one and the same prompt. Rounds 2 and
    3 run only the model chosen after round 1, each after the round before it
    and with a prompt that model has not run.
    """
    if not 1 <= number <= MAX_ROUNDS:
        raise SystemExit(f"The addendum allows rounds 1 to {MAX_ROUNDS}.")
    if answers_path(settings, "selector", TEST_SPLIT).is_file():
        raise SystemExit("The test split has been run, and no round runs after it.")
    check_round_settings(settings)
    check_judge_known(settings)
    begun = round_records(settings)
    if number == 1:
        if model not in ROUND_ONE_MODELS:
            raise SystemExit(f"Round 1 runs only {', '.join(ROUND_ONE_MODELS)}.")
        if model not in round_one_models(settings):
            raise SystemExit(
                f"The judge's kappa fell short of 0.6, so round 1 runs {WITHOUT_THE_JUDGE} alone."
            )
        others = [record for record in begun if record.number == 1 and record.model != model]
        if any(record.prompt_sha256 != prompt_sha256 for record in others):
            raise SystemExit("Round 1 runs every model with the same prompt, and it has changed.")
        return
    _, records = read_candidates(settings)
    log = read_log(work_dir(settings) / JUDGE_DIR / GRADES_NAME, QuoteGrade)
    chosen = chosen_model(settings, records, log)
    if chosen is None:
        raise SystemExit(
            "The model is not chosen yet: each model of round 1 must answer, and its "
            "quotations be graded, first."
        )
    if model != chosen:
        raise SystemExit(f"Rounds 2 and 3 run only the chosen model, {chosen}.")
    if not answers_path(settings, run_name(number - 1, model)).is_file():
        raise SystemExit(f"{run_name(number - 1, model)} has no answers yet.")
    for record in begun:
        if (
            record.number != number
            and record.model == model
            and record.prompt_sha256 == prompt_sha256
        ):
            raise SystemExit(f"{run_name(record.number, model)} already ran this prompt.")


# --- The language model's rounds ------------------------------------------------


def claude_selector(settings: Settings, model: str) -> ClaudeSelector:
    """One model's selector, with the prompt as it stands and the configured effort."""
    answer = settings.answer
    return ClaudeSelector(
        load_selector_prompt(),
        model,
        answer.effort,
        answer.max_tokens,
        settings.generation.prices[model],
    )


def round_requests(
    selector: ClaudeSelector, records: Sequence[QuestionCandidates]
) -> dict[str, dict[str, object]]:
    """The round's requests, one per question with candidates, by question id."""
    return {
        record.question_id: selector.request(record.question, record.candidates)
        for record in records
        if record.candidates
    }


def _batch(
    settings: Settings,
    directory: Path,
    record: RoundRecord,
    attempt: str,
    requests: Mapping[str, Mapping[str, object]],
    cap_usd: float | None = None,
    split: str = "dev",
) -> tuple[RoundRecord, list[Completion]]:
    """Run one batch of a round, or pick up the one already submitted, and log its completions."""
    log = directory / f"completions_{attempt}.jsonl"
    if log.is_file():
        return record, read_log(log, Completion)
    if attempt not in record.batches:
        high = round_costs(settings, record.model, requests)[1][-1]
        if split == "dev":
            check_budget(settings, high, cap_usd)
        else:
            check_test_budget(settings, directory, record.model, high, cap_usd)
    current = record

    def remember(batch_id: str) -> None:
        nonlocal current
        current = RoundRecord.model_validate(
            {**current.model_dump(), "batches": {**current.batches, attempt: batch_id}}
        )
        write_report(current, directory / ROUND_NAME)

    completions = run_batch(requests, on_submitted=remember, batch_id=record.batches.get(attempt))
    ordered = [completions[custom_id] for custom_id in requests if custom_id in completions]
    append_log(log, ordered)
    return current, ordered


def answer_all(
    settings: Settings,
    selector: ClaudeSelector,
    baseline: BaselineSelector,
    records: Sequence[QuestionCandidates],
    directory: Path,
    record: RoundRecord,
    cap_usd: float | None,
    split: str = "dev",
) -> list[AnsweredQuestion]:
    """Answer each question: one batch, a failed request once more, then the baseline.

    Returns:
        Every question's answer, assembled from the corpus, in the order of *records*.
    """
    requests = round_requests(selector, records)
    record, first = _batch(settings, directory, record, "first", requests, cap_usd, split)
    sent = {completion.custom_id: [completion] for completion in first}
    again = {
        question_id: params
        for question_id, params in requests.items()
        if question_id not in sent or failed(sent[question_id][-1])
    }
    if again:
        record, second = _batch(settings, directory, record, "again", again, cap_usd, split)
        for completion in second:
            sent.setdefault(completion.custom_id, []).append(completion)

    speeches = speeches_for(settings, records)
    answers = []
    for item in records:
        if not item.candidates:
            selected = selector.select(item.question, item.candidates)
        else:
            completions = sent.get(item.question_id) or [
                Completion(custom_id=item.question_id, error="no result")
            ]
            selected = selector.read(completions, _discount(settings))
            if selected.output is None:
                selected = fall_back(
                    selected, baseline.select(item.question, item.candidates), "failed"
                )
        answers.append(
            AnsweredQuestion(
                question_id=item.question_id,
                answer=assemble(selected, item.candidates, speeches),
            )
        )
    return answers


def run_round(settings: Settings, number: int, model: str, cap_usd: float | None = None) -> None:
    """Answer every question with one model, as one round, under *cap_usd* if given."""
    name = run_name(number, model)
    if answers_path(settings, name).is_file():
        raise SystemExit(f"{name} is already answered.")
    selector = claude_selector(settings, model)
    check_round(settings, number, model, selector.prompt.sha256)
    _, records = read_candidates(settings)
    index, _, _ = load_retrieval(settings)
    baseline = BaselineSelector(
        index, bm25_params(settings), threshold=read_threshold(settings).threshold
    )
    directory = round_dir(settings, name)
    record_path = directory / ROUND_NAME
    if record_path.is_file():
        record = RoundRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        if (record.prompt_sha256, record.effort) != (selector.prompt.sha256, selector.effort):
            raise SystemExit(f"{name} began with another prompt or effort.")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROMPT_DIR / f"{PROMPT_NAME}.md", directory / f"{PROMPT_NAME}.md")
        record = RoundRecord(
            number=number,
            model=model,
            effort=selector.effort,
            max_tokens=selector.max_tokens,
            prompt_sha256=selector.prompt.sha256,
            created=now(),
        )
        write_report(record, record_path)

    answers = answer_all(settings, selector, baseline, records, directory, record, cap_usd)
    write_lines(answers_path(settings, name), answers)
    fallbacks = sum(line.answer.selected.fallback is not None for line in answers)
    rejected = sum(len(line.answer.rejected) for line in answers)
    logger.info(
        "%s: %d questions, %d answered by the baseline after two failures, "
        "%d rejected selections; spent so far %s",
        name,
        len(answers),
        fallbacks,
        rejected,
        spent(settings),
    )


# --- The test run --------------------------------------------------------------------


def check_test_budget(
    settings: Settings, directory: Path, model: str, high_estimate: float, cap_usd: float | None
) -> None:
    """Stop unless the test run has a cap, and this batch could stay within it."""
    if cap_usd is None:
        raise SystemExit("The test run needs a cap: pass --max-usd.")
    price = settings.generation.prices[model]
    so_far = sum(
        cost_usd(completion.usage, price, _discount(settings))
        for path in sorted(directory.glob("completions_*.jsonl"))
        for completion in read_log(path, Completion)
    )
    if so_far + high_estimate > cap_usd:
        raise SystemExit(
            f"The test run has spent {so_far:.2f} USD; this batch could cost up to "
            f"{high_estimate:.2f} USD, beyond its cap of {cap_usd:.2f} USD."
        )


def run_test(settings: Settings, manifest: QuestionsManifest, cap_usd: float | None) -> None:
    """Answer the test split once, with the chosen model and prompt, and seal the answers.

    Nothing the run measures is computed or logged. The answers stay sealed
    until the retrieval run on the test split has been made, as the addendum on
    the test run sets out; only what ran is recorded.
    """
    if TEST_ADDENDUM not in committed_preregistration():
        raise SystemExit(f"The addendum on the test run is not in {PREREGISTRATION} at HEAD.")
    selector_path = answers_path(settings, "selector", TEST_SPLIT)
    if selector_path.is_file():
        raise SystemExit("The test split has been run once, which is all the addenda allow.")
    check_round_settings(settings)
    check_judge_known(settings)
    chosen = dev_choice(settings)
    name = run_name(chosen.number, chosen.model)
    prompt = load_selector_prompt(round_dir(settings, name))
    if prompt.sha256 != chosen.prompt_sha256:
        raise SystemExit(f"The copy of {name}'s prompt no longer matches its hash.")

    directory = work_dir(settings, TEST_SPLIT)
    if not (directory / RUN_NAME).is_file():
        make_candidates(settings, manifest, TEST_SPLIT)
    retrieval, records = read_candidates(settings, TEST_SPLIT)
    threshold = read_threshold(settings)
    index, _, _ = load_retrieval(settings)
    baseline = BaselineSelector(index, bm25_params(settings), threshold=threshold.threshold)
    baseline_path = answers_path(settings, "baseline", TEST_SPLIT)
    if not baseline_path.is_file():
        speeches = speeches_for(settings, records)
        lines = [
            AnsweredQuestion(
                question_id=record.question_id,
                answer=assemble(
                    baseline.select(record.question, record.candidates),
                    record.candidates,
                    speeches,
                ),
            )
            for record in records
        ]
        write_lines(baseline_path, lines)

    price = settings.generation.prices[chosen.model]
    selector = ClaudeSelector(prompt, chosen.model, chosen.effort, chosen.max_tokens, price)
    record_path = directory / ROUND_NAME
    if record_path.is_file():
        record = RoundRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    else:
        record = RoundRecord(
            number=chosen.number,
            model=chosen.model,
            effort=chosen.effort,
            max_tokens=chosen.max_tokens,
            prompt_sha256=chosen.prompt_sha256,
            created=now(),
        )
        write_report(record, record_path)
    answers = answer_all(
        settings, selector, baseline, records, directory, record, cap_usd, TEST_SPLIT
    )
    write_lines(selector_path, answers)

    record = RoundRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    cost = sum(
        cost_usd(completion.usage, price, _discount(settings))
        for path in sorted(directory.glob("completions_*.jsonl"))
        for completion in read_log(path, Completion)
    )
    sealed = SealedRun(
        model=chosen.model,
        round=name,
        prompt_sha256=chosen.prompt_sha256,
        effort=chosen.effort,
        max_tokens=chosen.max_tokens,
        threshold=threshold.threshold,
        retrieval=retrieval,
        batches=record.batches,
        answers_sha256={
            "baseline": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "selector": hashlib.sha256(selector_path.read_bytes()).hexdigest(),
        },
        cost_usd=round(cost, 4),
        created=now(),
    )
    write_report(sealed, settings.tracked.reports / TEST_RUN_REPORT)
    logger.info(
        "The test split is answered and sealed: %d questions with the prompt of %s, %.2f USD",
        len(answers),
        name,
        cost,
    )


# --- The judge -----------------------------------------------------------------------


def candidate_text(candidate: Candidate) -> str:
    """A candidate's text as the judge reads it: its paragraphs, separated by a blank line."""
    return "\n\n".join(candidate.paragraphs)


def answer_names(settings: Settings) -> list[str]:
    """The selectors and rounds with answers: the baseline, then each round."""
    names = ["baseline"] if answers_path(settings, "baseline").is_file() else []
    return names + [run_name(record.number, record.model) for record in recorded_rounds(settings)]


def candidate_pairs(records: Sequence[QuestionCandidates]) -> list[GradingPair]:
    """Each candidate of an answerable question, paired with its question."""
    return [
        GradingPair(
            question_id=record.question_id, question=record.question, text=candidate_text(candidate)
        )
        for record in records
        if record.answerable
        for candidate in record.candidates
    ]


def quote_pairs(
    records: Sequence[QuestionCandidates], answers: Mapping[str, Answer]
) -> list[GradingPair]:
    """Each quotation given to an answerable question, paired with its question."""
    return [
        GradingPair(question_id=record.question_id, question=record.question, text=quote.text)
        for record in records
        if record.answerable
        for quote in answers[record.question_id].quotes
    ]


def grading_pairs(settings: Settings, records: Sequence[QuestionCandidates]) -> list[GradingPair]:
    """Every pair the judge must grade: the candidates, then each selector's quotations."""
    pairs = candidate_pairs(records)
    for name in answer_names(settings):
        pairs.extend(quote_pairs(records, read_answers(settings, name)))
    return pairs


def run_judge(settings: Settings, cap_usd: float | None = None) -> None:
    """Grade every pair without a grade, sending a failed one once more, under *cap_usd*."""
    check_judge_known(settings)
    _, records = read_candidates(settings)
    prompt = load_prompt("judge")
    check_judge_prompt(prompt)
    judge = settings.judge
    prices = settings.generation.prices[judge.model]
    directory = work_dir(settings) / JUDGE_DIR
    log_path = directory / GRADES_NAME
    pairs = grading_pairs(settings, records)
    for attempt in ("first", "again"):
        todo = pending(pairs, read_log(log_path, QuoteGrade))
        if not todo:
            break
        check_budget(settings, judge_costs(settings, todo)[-1], cap_usd)
        requests = grading_requests(todo, prompt, judge.model, judge.effort, judge.max_tokens)

        def remember(batch_id: str, attempt: str = attempt, count: int = len(todo)) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "batches.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps({"batch_id": batch_id, "attempt": attempt, "pairs": count}) + "\n"
                )

        completions = run_batch(requests, on_submitted=remember)
        append_log(
            log_path, read_grades(todo, completions, prompt, prices, _discount(settings), now())
        )
    grades = grades_by_key(read_log(log_path, QuoteGrade))
    missing = sum(grades.get(pair.key) is None for pair in pairs)
    logger.info(
        "%d pairs, %d without a grade; spent so far %s", len(pairs), missing, spent(settings)
    )


# --- The report -----------------------------------------------------------------------


def outcomes_of(
    records: Sequence[QuestionCandidates],
    answers: Mapping[str, Answer],
    grades: Mapping[str, int | None],
) -> list[QuestionOutcome]:
    """Each question's outcome under one selector, with the judge's grades where it has them."""
    outcomes = []
    for record in records:
        answer = answers[record.question_id]
        quotes = []
        for quote in answer.quotes:
            pair = GradingPair(
                question_id=record.question_id, question=record.question, text=quote.text
            )
            quotes.append(
                QuoteOutcome(
                    passage_id=quote.passage_id,
                    first_paragraph=quote.first_paragraph,
                    last_paragraph=quote.last_paragraph,
                    party=quote.party,
                    text_sha256=text_sha256(quote.text),
                    grade=grades.get(pair.key),
                )
            )
        outcomes.append(
            QuestionOutcome(
                question_id=record.question_id,
                question_set=record.question_set,
                answerable=record.answerable,
                selector=answer.selected.selector,
                fallback=answer.selected.fallback,
                quotes=tuple(quotes),
                rejected=tuple(item.reason for item in answer.rejected),
                cost_usd=round(answer.selected.cost_usd, 6),
            )
        )
    return outcomes


def summarize(
    name: str,
    outcomes: Sequence[QuestionOutcome],
    record: RoundRecord | None,
    pending_grades: int,
) -> SelectorSummary:
    """One selector's or round's metrics."""
    shares = abstention(outcomes)
    precision = quote_precision(outcomes) if pending_grades == 0 else None
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    cost = sum(outcome.cost_usd for outcome in outcomes)
    return SelectorSummary(
        name=name,
        model=record.model if record else None,
        effort=record.effort if record else None,
        prompt_sha256=record.prompt_sha256 if record else None,
        abstention=shares,
        quote_precision=None if precision is None else round(precision, 6),
        score=None if precision is None else round((precision + shares.balanced_accuracy) / 2, 6),
        quotes=sum(len(outcome.quotes) for outcome in answerable),
        pending_grades=pending_grades,
        ungraded=ungraded(outcomes),
        quotes_unanswerable=sum(
            len(outcome.quotes) for outcome in outcomes if not outcome.answerable
        ),
        rejected=dict(
            sorted(Counter(reason for outcome in outcomes for reason in outcome.rejected).items())
        ),
        fallbacks=dict(
            sorted(Counter(outcome.fallback for outcome in outcomes if outcome.fallback).items())
        ),
        cost_usd=round(cost, 4),
        cost_per_question_usd=round(cost / len(outcomes), 6),
        parties=party_shares(quote.party for outcome in answerable for quote in outcome.quotes),
        launch_rule={
            "abstains_on_90_percent_of_unanswerable": shares.abstains_unanswerable >= 0.90,
            "needless_abstention_at_most_15_percent": shares.needless_abstention <= 0.15,
            "quote_precision_at_least_085": precision is not None and precision >= 0.85,
        },
    )


def leading_round(
    summaries: Sequence[SelectorSummary],
    rounds: Mapping[str, RoundRecord],
    chosen: str | None,
    accepted: bool | None,
) -> int | None:
    """Among the chosen model's rounds, the one that would go to the test run so far."""
    if chosen is None:
        return None
    mine = [
        (rounds[summary.name].number, summary)
        for summary in summaries
        if summary.name in rounds and rounds[summary.name].model == chosen
    ]
    if not mine:
        return None
    if accepted is False:
        return choose_round_by_accuracy(
            [(number, summary.abstention.balanced_accuracy) for number, summary in mine]
        )
    graded = [
        (number, summary.quote_precision, summary.abstention.balanced_accuracy)
        for number, summary in mine
        if summary.quote_precision is not None
    ]
    return choose_round(graded) if graded else None


def _rounded(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(values[0], 6), round(values[1], 6), round(values[2], 6))


def dev_summaries(
    settings: Settings, records: Sequence[QuestionCandidates], log: Sequence[QuoteGrade]
) -> tuple[
    dict[str, RoundRecord],
    dict[str, list[QuestionOutcome]],
    dict[str, int],
    tuple[SelectorSummary, ...],
]:
    """Each selector's and round's outcomes and metrics on the development split.

    Returns:
        The rounds by name, the outcomes by name, how many quotations of each
        still wait for the judge, and the summaries, the baseline first.
    """
    grades = grades_by_key(log)
    rounds = {run_name(record.number, record.model): record for record in recorded_rounds(settings)}
    names = answer_names(settings)
    answers = {name: read_answers(settings, name) for name in names}
    outcomes = {name: outcomes_of(records, answers[name], grades) for name in names}
    waiting = {name: len(pending(quote_pairs(records, answers[name]), log)) for name in names}
    summaries = tuple(
        summarize(name, outcomes[name], rounds.get(name), waiting[name]) for name in names
    )
    return rounds, outcomes, waiting, summaries


def dev_choice(settings: Settings) -> RoundRecord:
    """The round whose model and prompt go to the test run, as the addenda choose it."""
    _, records = read_candidates(settings)
    log = read_log(work_dir(settings) / JUDGE_DIR / GRADES_NAME, QuoteGrade)
    chosen = chosen_model(settings, records, log)
    if chosen is None:
        raise SystemExit("No model is chosen on the development split yet.")
    rounds, _, _, summaries = dev_summaries(settings, records, log)
    number = leading_round(summaries, rounds, chosen, judge_accepted(settings))
    if number is None:
        raise SystemExit(f"No round of {chosen} can be chosen yet.")
    return rounds[run_name(number, chosen)]


def make_report(settings: Settings) -> None:
    """Compute every selector's metrics and the comparisons, and write the tracked reports."""
    run, records = read_candidates(settings)
    threshold = read_threshold(settings)
    log = read_log(work_dir(settings) / JUDGE_DIR / GRADES_NAME, QuoteGrade)
    grades = grades_by_key(log)
    answer = settings.answer
    rounds, outcomes, waiting, summaries = dev_summaries(settings, records, log)
    names = [summary.name for summary in summaries]

    comparisons = []
    for name in names:
        if name == "baseline":
            continue
        precision: tuple[float, float, float] | None = None
        if waiting[name] == 0 and waiting["baseline"] == 0:
            precision = _rounded(
                precision_difference(
                    outcomes[name], outcomes["baseline"], answer.bootstrap_seed, answer.resamples
                )
            )
        comparisons.append(
            Comparison(
                name=name,
                quote_precision=precision,
                balanced_accuracy=_rounded(
                    accuracy_difference(
                        outcomes[name],
                        outcomes["baseline"],
                        answer.bootstrap_seed,
                        answer.resamples,
                    )
                ),
            )
        )

    candidate_grades = [
        [grades.get(pair.key) for pair in candidate_pairs([record])]
        for record in records
        if record.answerable
    ]
    graded = [values for values in candidate_grades if any(value is not None for value in values)]
    accepted = judge_accepted(settings)
    chosen = chosen_model(settings, records, log)
    report = AnswerReport(
        split="dev",
        sets_sha256=run.sets_sha256,
        retrieval=run,
        threshold=threshold,
        judge_prompt_sha256=load_prompt("judge").sha256,
        judge_accepted=accepted,
        selectors=summaries,
        comparisons=tuple(comparisons),
        candidate_parties=party_shares(
            candidate.party
            for record in records
            if record.answerable
            for candidate in record.candidates
        ),
        answerable_with_a_grade_2=(
            sum(any(value == 2 for value in values) for values in graded),
            len(graded),
        ),
        chosen_model=chosen,
        leading_round=leading_round(summaries, rounds, chosen, accepted),
        cost_usd=spent(settings),
        resamples=answer.resamples,
        bootstrap_seed=answer.bootstrap_seed,
    )
    write_report(report, settings.tracked.reports / REPORT)
    lines = [
        json.dumps({"run": name, **outcome.model_dump(mode="json")}, ensure_ascii=False)
        for name in names
        for outcome in outcomes[name]
    ]
    (settings.tracked.reports / QUESTIONS_REPORT).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    for summary in summaries:
        logger.info(
            "%-18s balanced accuracy %.3f (abstains on %.2f of unanswerable, answers %.2f of "
            "answerable), quote precision %s, %d quotes, %d waiting for the judge",
            summary.name,
            summary.abstention.balanced_accuracy,
            summary.abstention.abstains_unanswerable,
            summary.abstention.answers_answerable,
            "n/a" if summary.quote_precision is None else f"{summary.quote_precision:.3f}",
            summary.quotes,
            summary.pending_grades,
        )
    if chosen is not None:
        logger.info("Chosen after round 1: %s", chosen)


# --- The command line ----------------------------------------------------------------


def print_estimate(settings: Settings) -> None:
    """Print what a round of each model and the judge's pending pairs would cost."""
    _, records = read_candidates(settings)
    for model in by_price(settings, ROUND_ONE_MODELS):
        requests = round_requests(claude_selector(settings, model), records)
        mean_tokens, costs = round_costs(settings, model, requests)
        logger.info(
            "A round of %s at %s effort: %d requests, about %.0f input tokens each; in a "
            "batch %.2f / %.2f / %.2f USD at %d / %d / %d output tokens per request",
            model,
            settings.answer.effort,
            len(requests),
            mean_tokens,
            *costs,
            *SELECTOR_OUTPUT,
        )
    pairs = grading_pairs(settings, records)
    todo = pending(pairs, read_log(work_dir(settings) / JUDGE_DIR / GRADES_NAME, QuoteGrade))
    logger.info(
        "The judge: %d pairs to grade; in a batch %.2f / %.2f / %.2f USD at %d / %d / %d "
        "output tokens per pair",
        len(todo),
        *judge_costs(settings, todo),
        *JUDGE_OUTPUT,
    )
    logger.info("Spent so far: %s", spent(settings))


def _confirmed(args: argparse.Namespace, what: str) -> None:
    if not args.confirm:
        raise SystemExit(f"{what} calls the API; pass --confirm.")


def main() -> None:
    """Parse the command line and run the command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="The .env file that holds ANTHROPIC_API_KEY. Defaults to .env in the data root.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("candidates", help="Rank and record each question's candidates.")
    commands.add_parser("baseline", help="Choose the threshold and answer with the baseline.")
    commands.add_parser("estimate", help="Print what a round of each model and the judge cost.")
    cap_help = "A cap in USD, below the budget, on what the rounds and the judge spend together."
    paid_round = commands.add_parser("round", help="Answer with one model.")
    paid_round.add_argument("--number", type=int, required=True)
    paid_round.add_argument("--model", choices=ROUND_ONE_MODELS, required=True)
    paid_round.add_argument("--max-usd", type=float, default=None, help=cap_help)
    paid_round.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    judge = commands.add_parser("judge", help="Grade the pairs without a grade.")
    judge.add_argument("--max-usd", type=float, default=None, help=cap_help)
    judge.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    commands.add_parser("report", help="Write the metrics and the comparisons.")
    test_run = commands.add_parser("test-run", help="Answer the test split once, and seal it.")
    test_run.add_argument("--max-usd", type=float, required=True, help="The test run's cap in USD.")
    test_run.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    manifest = check_registered(settings)
    if args.command == "candidates":
        make_candidates(settings, manifest)
    elif args.command == "baseline":
        run_baseline(settings)
    elif args.command == "estimate":
        print_estimate(settings)
    elif args.command == "round":
        _confirmed(args, "A round")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_round(settings, args.number, args.model, args.max_usd)
    elif args.command == "judge":
        _confirmed(args, "The judge")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_judge(settings, args.max_usd)
    elif args.command == "test-run":
        _confirmed(args, "The test run")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_test(settings, manifest, args.max_usd)
    else:
        make_report(settings)


if __name__ == "__main__":
    main()
