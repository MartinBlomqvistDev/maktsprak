"""Step 14: run the answer step on the development split.

Commands, in order:

- ``candidates`` ranks the ten passages BM25 gives each question of the
  natural, filtered, stance and unanswerable sets, under its filter when it has
  one, and records them. The index must match the corpus on disk.
- ``baseline`` chooses the baseline selector's threshold by balanced accuracy
  and answers every question with it.
- ``estimate`` prints what a round of the language model, and the judge's
  grades still to give, would cost. It calls nothing.
- ``round`` answers every question with the language model as one message
  batch, sends a failed request once more, and lets the baseline answer where
  it fails again. The round is recorded with its prompt's hash and its effort.
  It needs ``--confirm``.
- ``judge`` grades every quotation given to an answerable question, and every
  candidate of one, that has no grade yet. It needs ``--confirm``.
- ``report`` computes each selector's metrics and the comparisons with the
  baseline from what is recorded, and writes them among the tracked reports.

Nothing runs before the question sets' hashes and the addendum on the answer
step stand in the pre-registration as committed at ``HEAD``. Only the
development split is read; the test split stays closed until comparison 1 has
decided the retrieval. The paid commands stop before a batch that could take
what the rounds and the judge have spent beyond ``answer.dev_budget_usd``.

Usage::

    python search/pipeline/14_run_answer_step.py candidates
    python search/pipeline/14_run_answer_step.py baseline
    python search/pipeline/14_run_answer_step.py estimate --effort low
    python search/pipeline/14_run_answer_step.py round --number 1 --effort low --confirm
    python search/pipeline/14_run_answer_step.py judge --confirm
    python search/pipeline/14_run_answer_step.py report
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
    choose_round,
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

#: The sets the answer step is run on, as the addendum names them.
ANSWER_SETS = ("natural", "filtered", "stance", "unanswerable")

#: The rounds the addendum allows on the development split.
MAX_ROUNDS = 3

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
        name: ``baseline`` or ``round-<n>``.
        effort: The round's effort.
        prompt_sha256: The round's prompt.
        abstention: Balanced accuracy and its shares.
        quote_precision: The share of quotations on answerable questions graded
            2, or ``None`` while some of them still wait for the judge.
        quotes: How many quotations on answerable questions.
        pending_grades: How many of them still wait for the judge.
        ungraded: How many of them have no grade.
        quotes_unanswerable: How many quotations were given to unanswerable questions.
        rejected: The rejected selections, by reason.
        fallbacks: The questions the baseline answered, by reason.
        cost_usd: What the round's requests cost.
        parties: The share of each party among the quotations on answerable questions.
        launch_rule: Whether each part of the launch rule holds on the point estimates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    effort: str | None = None
    prompt_sha256: str | None = None
    abstention: Abstention
    quote_precision: float | None
    quotes: int
    pending_grades: int
    ungraded: int
    quotes_unanswerable: int
    rejected: dict[str, int]
    fallbacks: dict[str, int]
    cost_usd: float
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
        leading_round: The round with the highest mean of quote precision and
            balanced accuracy so far.
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
    """Stop unless the sets match their hashes, and the hashes and the addendum are committed.

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
    for name in ANSWER_SETS:
        record = manifest.sets[name]
        actual = hashlib.sha256((settings.tracked.questions / record.file).read_bytes()).hexdigest()
        if actual != record.sha256:
            raise SystemExit(f"{record.file} no longer matches its hash in the manifest.")
        if record.sha256 not in registered:
            raise SystemExit(f"The hash of {name} is not in the committed {PREREGISTRATION}.")
    return manifest


def dev_questions(settings: Settings) -> list[EvalQuestion]:
    """The development split's questions of the four sets, set by set."""
    questions: list[EvalQuestion] = []
    for name in ANSWER_SETS:
        questions.extend(
            split_of(read_question_set(settings.tracked.questions / f"{name}.jsonl"), "dev")
        )
    return questions


# --- Files ------------------------------------------------------------------------


def work_dir(settings: Settings) -> Path:
    """Where the step's bulk data lives: the candidates, answers, completions and grades."""
    return settings.data.search / "answer" / "dev"


def round_dir(settings: Settings, number: int) -> Path:
    """Where one round's record, prompt and completions live."""
    return work_dir(settings) / ROUNDS_DIR / f"round-{number}"


def answers_path(settings: Settings, name: str) -> Path:
    """The answers of one selector or round, ``baseline`` or ``round-<n>``."""
    return work_dir(settings) / ANSWERS_DIR / f"{name}.jsonl"


def write_lines(path: Path, entries: Sequence[BaseModel]) -> None:
    """Write a JSON Lines file whole, one entry per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def read_candidates(settings: Settings) -> tuple[CandidatesRun, list[QuestionCandidates]]:
    """The recorded candidates and what they were ranked with."""
    directory = work_dir(settings)
    run_path = directory / RUN_NAME
    if not run_path.is_file():
        raise SystemExit("No candidates yet. Run the candidates command first.")
    run = CandidatesRun.model_validate_json(run_path.read_text(encoding="utf-8"))
    return run, read_log(directory / CANDIDATES_NAME, QuestionCandidates)


def read_answers(settings: Settings, name: str) -> dict[str, Answer]:
    """The answers of one selector or round, by question."""
    path = answers_path(settings, name)
    if not path.is_file():
        raise SystemExit(f"{path} does not exist.")
    return {line.question_id: line.answer for line in read_log(path, AnsweredQuestion)}


def recorded_rounds(settings: Settings) -> list[RoundRecord]:
    """The rounds with answers, in order."""
    rounds = []
    for number in range(1, MAX_ROUNDS + 1):
        record_path = round_dir(settings, number) / ROUND_NAME
        if record_path.is_file() and answers_path(settings, f"round-{number}").is_file():
            rounds.append(RoundRecord.model_validate_json(record_path.read_text(encoding="utf-8")))
    return rounds


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


def make_candidates(settings: Settings, manifest: QuestionsManifest) -> None:
    """Rank each question's candidates and record them."""
    check_tuned(settings)
    index, index_manifest, table = load_retrieval(settings)
    params = bm25_params(settings)
    masks = FilterMasks(table)
    questions = dev_questions(settings)
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
    directory = work_dir(settings)
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
    """What the rounds and the judge have cost so far, by part."""
    price = settings.generation.prices[settings.answer.model]
    parts: dict[str, float] = {}
    for number in range(1, MAX_ROUNDS + 1):
        total = sum(
            cost_usd(completion.usage, price, _discount(settings))
            for path in sorted(round_dir(settings, number).glob("completions_*.jsonl"))
            for completion in read_log(path, Completion)
        )
        if total:
            parts[f"round-{number}"] = round(total, 4)
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
    settings: Settings, requests: Mapping[str, Mapping[str, object]]
) -> tuple[float, tuple[float, ...]]:
    """The mean input tokens of a round's requests, and their cost at each assumed output."""
    overhead = judge_overhead_words()
    tokens = [estimate_tokens(request_words(params), overhead) for params in requests.values()]
    mean = sum(tokens) / len(tokens) if tokens else 0.0
    return mean, batch_costs(settings, settings.answer.model, tokens, SELECTOR_OUTPUT)


def judge_costs(settings: Settings, pairs: Sequence[GradingPair]) -> tuple[float, ...]:
    """What grading *pairs* would cost, at each assumed output."""
    tokens = [estimate_tokens(len(pair.text.split())) for pair in pairs]
    return batch_costs(settings, settings.judge.model, tokens, JUDGE_OUTPUT)


def check_budget(settings: Settings, high_estimate: float) -> None:
    """Stop if the batch could take the spending beyond the development budget."""
    so_far = sum(spent(settings).values())
    cap = settings.answer.dev_budget_usd
    if so_far + high_estimate > cap:
        raise SystemExit(
            f"Spent {so_far:.2f} USD; this batch could cost up to {high_estimate:.2f} USD, "
            f"beyond the budget of {cap:.2f} USD."
        )


# --- The language model's rounds ------------------------------------------------


def claude_selector(settings: Settings, effort: str) -> ClaudeSelector:
    """The language model's selector at an effort, with the prompt as it stands."""
    answer = settings.answer
    return ClaudeSelector(
        load_selector_prompt(),
        answer.model,
        effort,
        answer.max_tokens,
        settings.generation.prices[answer.model],
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
) -> tuple[RoundRecord, list[Completion]]:
    """Run one batch of a round, or pick up the one already submitted, and log its completions."""
    log = directory / f"completions_{attempt}.jsonl"
    if log.is_file():
        return record, read_log(log, Completion)
    if attempt not in record.batches:
        check_budget(settings, round_costs(settings, requests)[1][-1])
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


def run_round(settings: Settings, number: int, effort: str) -> None:
    """Answer every question with the language model, as one round."""
    if not 1 <= number <= MAX_ROUNDS:
        raise SystemExit(f"The addendum allows rounds 1 to {MAX_ROUNDS}.")
    if number > 1 and not answers_path(settings, f"round-{number - 1}").is_file():
        raise SystemExit(f"Round {number - 1} has no answers yet.")
    if answers_path(settings, f"round-{number}").is_file():
        raise SystemExit(f"Round {number} is already answered.")
    _, records = read_candidates(settings)
    index, _, _ = load_retrieval(settings)
    baseline = BaselineSelector(
        index, bm25_params(settings), threshold=read_threshold(settings).threshold
    )
    selector = claude_selector(settings, effort)
    directory = round_dir(settings, number)
    record_path = directory / ROUND_NAME
    if record_path.is_file():
        record = RoundRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        if (record.prompt_sha256, record.effort) != (selector.prompt.sha256, effort):
            raise SystemExit(f"Round {number} began with another prompt or effort.")
    else:
        for earlier in recorded_rounds(settings):
            if (earlier.prompt_sha256, earlier.effort) == (selector.prompt.sha256, effort):
                raise SystemExit(f"Round {earlier.number} already ran this prompt at this effort.")
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROMPT_DIR / f"{PROMPT_NAME}.md", directory / f"{PROMPT_NAME}.md")
        record = RoundRecord(
            number=number,
            model=settings.answer.model,
            effort=effort,
            max_tokens=settings.answer.max_tokens,
            prompt_sha256=selector.prompt.sha256,
            created=now(),
        )
        write_report(record, record_path)

    requests = round_requests(selector, records)
    record, first = _batch(settings, directory, record, "first", requests)
    sent = {completion.custom_id: [completion] for completion in first}
    again = {
        question_id: params
        for question_id, params in requests.items()
        if question_id not in sent or failed(sent[question_id][-1])
    }
    if again:
        record, second = _batch(settings, directory, record, "again", again)
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
    write_lines(answers_path(settings, f"round-{number}"), answers)
    fallbacks = sum(line.answer.selected.fallback is not None for line in answers)
    rejected = sum(len(line.answer.rejected) for line in answers)
    logger.info(
        "Round %d: %d questions, %d answered by the baseline after two failures, "
        "%d rejected selections; spent so far %s",
        number,
        len(answers),
        fallbacks,
        rejected,
        spent(settings),
    )


# --- The judge -----------------------------------------------------------------------


def candidate_text(candidate: Candidate) -> str:
    """A candidate's text as the judge reads it: its paragraphs, separated by a blank line."""
    return "\n\n".join(candidate.paragraphs)


def answer_names(settings: Settings) -> list[str]:
    """The selectors and rounds with answers: the baseline, then each round."""
    names = ["baseline"] if answers_path(settings, "baseline").is_file() else []
    return names + [f"round-{record.number}" for record in recorded_rounds(settings)]


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


def run_judge(settings: Settings) -> None:
    """Grade every pair without a grade, sending a failed one once more."""
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
        check_budget(settings, judge_costs(settings, todo)[-1])
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
    return SelectorSummary(
        name=name,
        effort=record.effort if record else None,
        prompt_sha256=record.prompt_sha256 if record else None,
        abstention=shares,
        quote_precision=None if precision is None else round(precision, 6),
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
        cost_usd=round(sum(outcome.cost_usd for outcome in outcomes), 4),
        parties=party_shares(quote.party for outcome in answerable for quote in outcome.quotes),
        launch_rule={
            "abstains_on_90_percent_of_unanswerable": shares.abstains_unanswerable >= 0.90,
            "needless_abstention_at_most_15_percent": shares.needless_abstention <= 0.15,
            "quote_precision_at_least_085": precision is not None and precision >= 0.85,
        },
    )


def judge_accepted(settings: Settings) -> bool | None:
    """Whether the judge's kappa reached the threshold, or ``None`` before it is known."""
    path = settings.tracked.reports / AGREEMENT_REPORT
    if not path.is_file():
        return None
    return bool(json.loads(path.read_text(encoding="utf-8"))["agreement"]["accepted"])


def _rounded(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(values[0], 6), round(values[1], 6), round(values[2], 6))


def make_report(settings: Settings) -> None:
    """Compute every selector's metrics and the comparisons, and write the tracked reports."""
    run, records = read_candidates(settings)
    threshold = read_threshold(settings)
    log = read_log(work_dir(settings) / JUDGE_DIR / GRADES_NAME, QuoteGrade)
    grades = grades_by_key(log)
    answer = settings.answer
    rounds = {f"round-{record.number}": record for record in recorded_rounds(settings)}
    names = answer_names(settings)
    answers = {name: read_answers(settings, name) for name in names}
    outcomes = {name: outcomes_of(records, answers[name], grades) for name in names}
    waiting = {name: len(pending(quote_pairs(records, answers[name]), log)) for name in names}
    summaries = tuple(
        summarize(name, outcomes[name], rounds.get(name), waiting[name]) for name in names
    )

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
    leading = [
        (rounds[summary.name].number, summary.quote_precision, summary.abstention.balanced_accuracy)
        for summary in summaries
        if summary.name in rounds and summary.quote_precision is not None
    ]
    report = AnswerReport(
        split="dev",
        sets_sha256=run.sets_sha256,
        retrieval=run,
        threshold=threshold,
        judge_prompt_sha256=load_prompt("judge").sha256,
        judge_accepted=judge_accepted(settings),
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
        leading_round=choose_round(leading) if leading else None,
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
            "%-9s balanced accuracy %.3f (abstains on %.2f of unanswerable, answers %.2f of "
            "answerable), quote precision %s, %d quotes, %d waiting for the judge",
            summary.name,
            summary.abstention.balanced_accuracy,
            summary.abstention.abstains_unanswerable,
            summary.abstention.answers_answerable,
            "n/a" if summary.quote_precision is None else f"{summary.quote_precision:.3f}",
            summary.quotes,
            summary.pending_grades,
        )


# --- The command line ----------------------------------------------------------------


def print_estimate(settings: Settings, effort: str) -> None:
    """Print what a round at *effort* and the judge's pending pairs would cost."""
    _, records = read_candidates(settings)
    requests = round_requests(claude_selector(settings, effort), records)
    mean_tokens, costs = round_costs(settings, requests)
    logger.info(
        "A round at %s effort: %d requests, about %.0f input tokens each; in a batch "
        "%.2f / %.2f / %.2f USD at %d / %d / %d output tokens per request",
        effort,
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
    efforts = ["low", "medium", "high", "xhigh", "max"]
    estimate = commands.add_parser("estimate", help="Print what a round and the judge would cost.")
    estimate.add_argument("--effort", choices=efforts, default=None)
    paid_round = commands.add_parser("round", help="Answer with the language model.")
    paid_round.add_argument("--number", type=int, required=True)
    paid_round.add_argument("--effort", choices=efforts, default=None)
    paid_round.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    judge = commands.add_parser("judge", help="Grade the pairs without a grade.")
    judge.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    commands.add_parser("report", help="Write the metrics and the comparisons.")
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    manifest = check_registered(settings)
    effort = getattr(args, "effort", None) or settings.answer.effort
    if args.command == "candidates":
        make_candidates(settings, manifest)
    elif args.command == "baseline":
        run_baseline(settings)
    elif args.command == "estimate":
        print_estimate(settings, effort)
    elif args.command == "round":
        _confirmed(args, "A round")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_round(settings, args.number, effort)
    elif args.command == "judge":
        _confirmed(args, "The judge")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_judge(settings)
    else:
        make_report(settings)


if __name__ == "__main__":
    main()
