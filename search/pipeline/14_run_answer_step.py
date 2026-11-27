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
  candidate of one, that has no grade yet, once the judge's kappa has reached
  0.6. It needs ``--confirm``.
- ``report`` computes each selector's metrics, the comparisons with the
  baseline and the choice of model from what is recorded, and writes them
  among the tracked reports.
- ``test-run`` answers the test split once with the model and the prompt the
  development split chose, and the baseline with its threshold. It records
  what ran and computes nothing from it: the answers stay sealed. It needs
  ``--max-usd`` and ``--confirm``.

With ``--hybrid``, three of the commands follow the addendum of 2026-11-24 on
the hybrid's candidates, and keep everything they write apart from the rest:

- ``candidates --hybrid`` asks for each question's vector one question per
  request, as the page asks, and ranks each question's candidates with the
  hybrid that comparison 1 put in production. A question whose vector took
  longer than the page waits gets BM25's ten. It needs ``--confirm``.
- ``baseline --hybrid`` chooses the baseline's threshold again on the
  development split, from the hybrid's candidates.
- ``test-run --hybrid`` answers the test split once more, from the hybrid's
  candidates, with the model, the prompt, the effort and the output limit of
  the first run on the test split, whose answers it never reads. The answers
  are not sealed, but the record holds only what ran. It needs ``--max-usd``
  and ``--confirm``.

The judge's kappa fell short of 0.6, so the quotations of that run are graded
by hand, and three commands follow it, none of them paid:

- ``hand-form`` draws 100 answerable questions of the test split with the seed
  20261124, records the draw, and writes the form that shows each quotation
  either selector gave them, once, in an order drawn with the same seed.
- ``test-report`` reads the grades the form handed over, once the draw's record
  is committed, and computes the comparisons and the launch rule.
- ``open-sealed`` opens the first run on the test split, once the test report
  is committed, and describes how its selectors abstain.

Nothing runs before the question sets' hashes and the addenda on the answer
step stand in the pre-registration as committed at ``HEAD``, and nothing paid
runs before the judge's agreement with the hand grades is known. Only
``test-run`` reads the test split: once from BM25's candidates, after the
addendum on the test run is committed, and once from the hybrid's, after the
addendum on them is. No round runs after either. The paid commands on the
development split stop before a batch that could take what the rounds and the
judge have spent beyond ``answer.dev_budget_usd``, or beyond ``--max-usd`` when
that is lower; a run on the test split stops at its own ``--max-usd``.

Usage::

    python search/pipeline/14_run_answer_step.py candidates
    python search/pipeline/14_run_answer_step.py baseline
    python search/pipeline/14_run_answer_step.py estimate
    python search/pipeline/14_run_answer_step.py round --number 1 --model claude-haiku-5-5 \\
        --confirm
    python search/pipeline/14_run_answer_step.py judge --confirm
    python search/pipeline/14_run_answer_step.py report
    python search/pipeline/14_run_answer_step.py test-run --max-usd 11 --confirm
    python search/pipeline/14_run_answer_step.py candidates --hybrid --confirm
    python search/pipeline/14_run_answer_step.py baseline --hybrid
    python search/pipeline/14_run_answer_step.py test-run --hybrid --max-usd 7 --confirm
    python search/pipeline/14_run_answer_step.py hand-form
    python search/pipeline/14_run_answer_step.py test-report --answers bedomning-testdelen.json
    python search/pipeline/14_run_answer_step.py open-sealed
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
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.answer.assemble import assemble
from search.answer.candidates import (
    CANDIDATES,
    build_candidates,
    check_index,
    fuse_candidates,
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
from search.config import REPO_ROOT, Prices, Settings
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
from search.eval.hand_grades import (
    HAND_QUESTIONS,
    HAND_SEED,
    HandDraw,
    draw_questions,
    hand_form,
    name_pairs,
    order_pairs,
    read_hand_grades,
)
from search.eval.hybrid import QueryVectors, lexical_hits, require_committed, vector_hits
from search.eval.hybrid import text_sha256 as file_text_sha256
from search.eval.judge import GRADES
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
from search.index.embed import (
    Answered,
    OpenRouterEmbedder,
    build_embedder,
    load_openrouter_key,
    save_array,
)
from search.index.embed import Usage as EmbeddingUsage
from search.index.lexical import (
    Bm25Params,
    Hit,
    IndexManifest,
    LexicalIndex,
    file_sha256,
    lexical_dir,
)
from search.index.vector import VectorIndex, index_path, load_vector_index

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

#: The addendum that answers the test split again, from the hybrid's candidates,
#: and the record of that run. What comes from the hybrid's candidates lives
#: below its own directory.
HYBRID_ADDENDUM = (
    "### 2026-11-24: the answer step on the hybrid's candidates, run again on the test split"
)
HYBRID_TEST_RUN_REPORT = "answer_test_run_hybrid.json"
HYBRID = "hybrid"

#: Where the candidates come from: BM25 alone, or the hybrid.
Retrieval = Literal["bm25", "hybrid"]

#: The hybrid's tuning on the development split, and the retrieval run on the
#: test split that put it in production.
HYBRID_TUNING_REPORT = "hybrid_tuning_dev.json"
RETRIEVAL_TEST_REPORT = "retrieval_test.json"

#: The questions' vectors as the page asks for them, and their record.
QUERY_VECTORS_NAME = "query_vectors.npy"
QUERY_RECORD_NAME = "query_vectors.json"

#: How long the page waits for a question's vector before it answers from BM25 alone.
VECTOR_WAIT_S = 30.0

#: Grading the run from the hybrid's candidates by hand: the draw's record, the
#: form's name and directory, the reports, and the selectors in report order.
HAND_DRAW_REPORT = "answer_test_hand_draw.json"
HAND_FORM_KEY = "bedomning-testdelen"
HAND_DIR = "hand"
TEST_REPORT = "answer_test.json"
TEST_QUESTIONS_REPORT = "answer_test_questions.jsonl"
SEALED_REPORT = "answer_test_sealed.json"
SELECTORS = ("baseline", "selector")

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


class HybridCandidatesRun(CandidatesRun):
    """What the hybrid's candidates were ranked with: BM25 as above, the vectors and the fusion.

    Attributes:
        embedding: The embedding model, by its name in the settings.
        target_words: The passage size of the vectors.
        rrf_k: The RRF constant.
        window: How many passages each of the two rankings holds.
        tuning_sha256: The hash of the tuning's report, as the addendum on the
            tuned configuration records it.
        vectors_sha256: The hash of the vector index's matrix, from its manifest.
        query_vectors_sha256: The hash of the questions' vectors.
        bm25_alone: The questions whose vector took longer than the page waits,
            which got BM25's ten.
    """

    embedding: str
    target_words: int
    rrf_k: float
    window: int
    tuning_sha256: str
    vectors_sha256: str
    query_vectors_sha256: str
    bm25_alone: tuple[str, ...]


class QueryVectorsRecord(BaseModel):
    """The questions' vectors, asked for one question per request, as the page asks for them.

    Attributes:
        embedding: The embedding model, by its name in the settings.
        model: Its id at OpenRouter.
        revision: The commit of its weights.
        provider: The provider the requests were routed to, without fallback.
        quantization: The precision that provider serves it in.
        question_ids: The questions, in the order of the vectors.
        vectors_sha256: The hash of the vectors' file.
        seconds: How long each question waited for its vector, by question.
        attempts: How many requests each question needed, by question.
        requests: How many requests were sent.
        tokens: The input tokens the API counted.
        cost_usd: What the requests cost.
        providers: The providers that answered, as the API named them.
        started: When the first request was sent.
        finished: When the last vector came.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding: str
    model: str
    revision: str
    provider: str
    quantization: str
    question_ids: tuple[str, ...]
    vectors_sha256: str
    seconds: dict[str, float]
    attempts: dict[str, int]
    requests: int
    tokens: int
    cost_usd: float
    providers: tuple[str, ...]
    started: datetime
    finished: datetime


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


class HeldOutRun(BaseModel):
    """A run of the answer step on the test split: what ran, and nothing it measured.

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


class SealedRun(HeldOutRun):
    """The first run of the answer step on the test split, from BM25's candidates.

    Its results stayed sealed until the retrieval run on the test split had been
    made, so this record holds no metric.
    """


class HybridTestRun(HeldOutRun):
    """The second run of the answer step on the test split, from the hybrid's candidates.

    It was made after the retrieval run on the test split, so it is not sealed.
    Like the first, the record holds no metric: the comparisons are computed
    with the hand grades.

    Attributes:
        retrieval: What the test questions' candidates were ranked with.
    """

    retrieval: HybridCandidatesRun


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
            2, or ``None`` while some of them still wait for the judge, or when
            it is not measured because the judge's kappa fell short of 0.6.
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
        launch_rule: Whether each part of the launch rule holds on the point
            estimates; the part on quote precision is ``None`` when quote
            precision is not measured.
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
    launch_rule: dict[str, bool | None]


class Comparison(BaseModel):
    """One round against the baseline: differences with their paired 95% intervals.

    Attributes:
        name: The round.
        quote_precision: The difference, round minus baseline, and the interval's
            bounds; ``None`` while either has quotations that wait for a grade,
            or when quote precision is not measured.
        balanced_accuracy: The same for balanced accuracy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    quote_precision: tuple[float, float, float] | None
    balanced_accuracy: tuple[float, float, float]


#: The part of the launch rule that reads quote precision.
PRECISION_PART = "quote_precision_at_least_085"

#: What the block of the judge's grades says, in the report itself.
UNACCEPTED_JUDGE_NOTE = (
    "The judge's kappa against the hand grades fell short of 0.6, so quote precision is not "
    "measured on the development split. These figures come from the judge's grades and are "
    "description without decision value: no model, round or launch rule was decided from "
    "them. The grades in answer_dev_questions.jsonl are the judge's as well."
)


class JudgeFigures(BaseModel):
    """What one selector's or round's quotations come to under the judge's grades.

    Attributes:
        quote_precision: The share of its quotations on answerable questions the
            judge grades 2, or ``None`` while some of them still wait for the judge.
        score: The mean of that share and balanced accuracy, or ``None`` without it.
        quote_precision_at_least_085: Whether that share reaches 0.85.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quote_precision: float | None
    score: float | None
    quote_precision_at_least_085: bool


class UnacceptedJudge(BaseModel):
    """The judge's grades of the development split after its kappa fell short: description.

    The addendum of 2026-11-23 does not measure quote precision on the
    development split when the judge's kappa falls short of 0.6, and the model
    and the round are then chosen by balanced accuracy alone. What the judge's
    grades come to is kept here, apart from the registered metrics, and nothing
    is decided from it.

    Attributes:
        note: The same, for whoever reads the report on its own.
        selectors: Each selector's and round's figures under the judge's grades.
        comparisons: Each round's difference in that share from the baseline,
            with its paired 95% interval, or ``None`` while grades wait.
        answerable_with_a_grade_2: How many answerable questions have at least
            one candidate the judge grades 2, and how many have a graded candidate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    note: str = UNACCEPTED_JUDGE_NOTE
    selectors: dict[str, JudgeFigures]
    comparisons: dict[str, tuple[float, float, float] | None]
    answerable_with_a_grade_2: tuple[int, int]


class AnswerReport(BaseModel):
    """The answer step on the development split. Description: nothing here decides.

    When the judge's kappa fell short of 0.6, quote precision, the score, the
    part of the launch rule on quote precision, its intervals and the count of
    candidates graded 2 are left out of the registered fields, and what the
    judge's grades give for them stands apart in ``unaccepted_judge``.

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
            one candidate the judge grades 2, and how many have a graded
            candidate; ``None`` when the judge's kappa fell short.
        chosen_model: The model chosen after round 1, or ``None`` before round 1
            is answered by each of its models and graded.
        leading_round: Among the chosen model's rounds, the one with the highest
            mean of quote precision and balanced accuracy so far, or the highest
            balanced accuracy if the judge's kappa fell short.
        cost_usd: What the rounds and the judge have cost, by part.
        resamples: The bootstrap resamples behind each interval.
        bootstrap_seed: The seed of the resamples.
        unaccepted_judge: The judge's figures, as description, when its kappa
            fell short; ``None`` otherwise.
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
    answerable_with_a_grade_2: tuple[int, int] | None
    chosen_model: str | None
    leading_round: int | None
    cost_usd: dict[str, float]
    resamples: int
    bootstrap_seed: int
    unaccepted_judge: UnacceptedJudge | None = None


class HandSummary(BaseModel):
    """How one selector did on the test split, with its quotations graded by hand.

    Attributes:
        name: ``baseline`` or ``selector``.
        abstention: Balanced accuracy and its shares, on every question of the split.
        quote_precision: The share of its quotations on the drawn questions graded
            2, or ``None`` when it gave none.
        graded_quotes: Its quotations on the drawn questions.
        quotes: Its quotations on every answerable question.
        quotes_unanswerable: Its quotations on unanswerable questions.
        rejected: The rejected selections, by reason.
        fallbacks: The questions the baseline answered in its place, by reason.
        cost_usd: What its requests cost.
        cost_per_question_usd: That cost over every question of the split.
        parties: The share of each party among its quotations on answerable questions.
        launch_rule: Whether each part of the launch rule holds on the point estimates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    abstention: Abstention
    quote_precision: float | None
    graded_quotes: int
    quotes: int
    quotes_unanswerable: int
    rejected: dict[str, int]
    fallbacks: dict[str, int]
    cost_usd: float
    cost_per_question_usd: float
    parties: dict[str, float]
    launch_rule: dict[str, bool]


class AnswerTestReport(BaseModel):
    """The answer step on the test split, from the hybrid's candidates, graded by hand.

    Attributes:
        split: The split run.
        run_sha256: The hash of the run's record.
        draw_sha256: The hash of the draw's record.
        grades_sha256: The hash of the answers the form handed over.
        retrieval: What the candidates were ranked with.
        threshold: The baseline's threshold.
        drawn_questions: How many answerable questions were drawn.
        graded_pairs: How many pairs of question and quotation were graded.
        grades: How many pairs got each grade.
        selectors: The baseline, then the language model.
        comparison: The language model against the baseline, with paired intervals.
        beats_baseline: Whether the lower bound of both intervals lies above 0.
        runs: The selector that runs: the language model if it beats the baseline.
        launch: Whether the launch rule holds for the selector that runs.
        candidate_parties: The share of each party among the candidates of the
            answerable questions.
        resamples: The bootstrap resamples behind each interval.
        bootstrap_seed: The seed of the resamples.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    run_sha256: str
    draw_sha256: str
    grades_sha256: str
    retrieval: HybridCandidatesRun
    threshold: float
    drawn_questions: int
    graded_pairs: int
    grades: dict[str, int]
    selectors: tuple[HandSummary, ...]
    comparison: Comparison
    beats_baseline: bool
    runs: str
    launch: bool
    candidate_parties: dict[str, float]
    resamples: int
    bootstrap_seed: int


class SealedDescription(BaseModel):
    """The first run on the test split, opened after the report on the second: description.

    Attributes:
        report_sha256: The hash of the test report, committed before the run was opened.
        run: The first run's record.
        abstention: Each selector's balanced accuracy and the two shares behind it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_sha256: str
    run: SealedRun
    abstention: dict[str, Abstention]


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


def questions_of(settings: Settings, split: str) -> list[EvalQuestion]:
    """The questions of the development split, or of the test split."""
    return dev_questions(settings) if split == "dev" else held_out_questions(settings)


# --- Files ------------------------------------------------------------------------


def work_dir(settings: Settings, split: str = "dev", retrieval: Retrieval = "bm25") -> Path:
    """Where the step's bulk data lives: the candidates, answers, completions and grades.

    What comes from the hybrid's candidates lives apart, below ``hybrid``.
    """
    base = settings.data.search / "answer"
    return base / split if retrieval == "bm25" else base / HYBRID / split


def run_name(number: int, model: str) -> str:
    """A round's name: its number and its model, as in ``round-1-opus-5-5``."""
    return f"round-{number}-{model.removeprefix('claude-')}"


def round_dir(settings: Settings, name: str) -> Path:
    """Where one round's record, prompt and completions live."""
    return work_dir(settings) / ROUNDS_DIR / name


def answers_path(
    settings: Settings, name: str, split: str = "dev", retrieval: Retrieval = "bm25"
) -> Path:
    """The answers of one selector or round: ``baseline``, a round's name or ``selector``."""
    return work_dir(settings, split, retrieval) / ANSWERS_DIR / f"{name}.jsonl"


def write_lines(path: Path, entries: Sequence[BaseModel]) -> None:
    """Write a JSON Lines file whole, one entry per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def read_candidates(
    settings: Settings, split: str = "dev", retrieval: Retrieval = "bm25"
) -> tuple[CandidatesRun, list[QuestionCandidates]]:
    """The recorded candidates and what they were ranked with."""
    directory = work_dir(settings, split, retrieval)
    run_path = directory / RUN_NAME
    if not run_path.is_file():
        command = "candidates" if retrieval == "bm25" else "candidates --hybrid"
        raise SystemExit(f"No candidates yet. Run the {command} command first.")
    kind = CandidatesRun if retrieval == "bm25" else HybridCandidatesRun
    run = kind.model_validate_json(run_path.read_text(encoding="utf-8"))
    return run, read_log(directory / CANDIDATES_NAME, QuestionCandidates)


def read_answers(
    settings: Settings, name: str, split: str = "dev", retrieval: Retrieval = "bm25"
) -> dict[str, Answer]:
    """The answers of one selector or round, by question."""
    path = answers_path(settings, name, split, retrieval)
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


def record_candidates(
    settings: Settings,
    questions: Sequence[EvalQuestion],
    hits: Mapping[str, Sequence[Hit]],
    table: PassageTable,
    directory: Path,
) -> list[QuestionCandidates]:
    """Turn each question's hits into candidates, and write them to *directory*."""
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
    write_lines(directory / CANDIDATES_NAME, records)
    counts = Counter(len(record.candidates) for record in records)
    logger.info(
        "%d questions; candidates per question: %s",
        len(records),
        dict(sorted(counts.items())),
    )
    return records


def bm25_run(
    settings: Settings,
    manifest: QuestionsManifest,
    index_manifest: IndexManifest,
    params: Bm25Params,
    questions: int,
) -> CandidatesRun:
    """What BM25 ranked the candidates with."""
    return CandidatesRun(
        analyzer=settings.answer.analyzer,
        k1=params.k1,
        b=params.b,
        limit=CANDIDATES,
        passages=index_manifest.passages,
        passage_ids_sha256=index_manifest.passage_ids_sha256,
        passages_sha256=index_manifest.passages_sha256,
        speeches_sha256=index_manifest.speeches_sha256,
        sets_sha256={name: manifest.sets[name].sha256 for name in ANSWER_SETS},
        questions=questions,
        created=now(),
    )


def make_candidates(settings: Settings, manifest: QuestionsManifest, split: str = "dev") -> None:
    """Rank each question's candidates with BM25 and record them."""
    check_tuned(settings)
    index, index_manifest, table = load_retrieval(settings)
    params = bm25_params(settings)
    masks = FilterMasks(table)
    questions = questions_of(settings, split)
    hits = {
        question.question_id: rank_passages(index, question.text, params, masks.of(question))
        for question in questions
    }
    directory = work_dir(settings, split)
    records = record_candidates(settings, questions, hits, table, directory)
    write_report(
        bm25_run(settings, manifest, index_manifest, params, len(records)), directory / RUN_NAME
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


def read_threshold(settings: Settings, retrieval: Retrieval = "bm25") -> ThresholdChoice:
    """The baseline's threshold, as recorded on the development split."""
    path = work_dir(settings, "dev", retrieval) / THRESHOLD_NAME
    if not path.is_file():
        command = "baseline" if retrieval == "bm25" else "baseline --hybrid"
        raise SystemExit(f"No threshold yet. Run the {command} command first.")
    return ThresholdChoice.model_validate_json(path.read_text(encoding="utf-8"))


def baseline_answers(
    settings: Settings, selector: BaselineSelector, records: Sequence[QuestionCandidates]
) -> list[AnsweredQuestion]:
    """The baseline's answer to each question, assembled from the corpus."""
    speeches = speeches_for(settings, records)
    return [
        AnsweredQuestion(
            question_id=record.question_id,
            answer=assemble(
                selector.select(record.question, record.candidates), record.candidates, speeches
            ),
        )
        for record in records
    ]


def run_baseline(settings: Settings) -> None:
    """Choose the baseline's threshold, then answer every question with it."""
    _, records = read_candidates(settings)
    selector, choice = baseline_selector(settings, records)
    write_report(choice, work_dir(settings) / THRESHOLD_NAME)
    answers = baseline_answers(settings, selector, records)
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
    if any(
        answers_path(settings, "selector", TEST_SPLIT, retrieval).is_file()
        for retrieval in ("bm25", HYBRID)
    ):
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


# --- The hybrid's candidates ------------------------------------------------------


class HybridConfiguration(BaseModel):
    """The hybrid that comparison 1 put in production, as the tuning chose it.

    Attributes:
        embedding: The embedding model, by its name in the settings.
        target_words: The passage size.
        k: The RRF constant.
        window: How many passages each of the two rankings holds.
        tuning_sha256: The hash of the tuning's report.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding: str
    target_words: int
    k: float
    window: int
    tuning_sha256: str


def check_hybrid_registered() -> None:
    """Stop unless the addendum on the hybrid's candidates stands in the pre-registration."""
    if HYBRID_ADDENDUM not in committed_preregistration():
        raise SystemExit(
            f"The addendum on the hybrid's candidates is not in {PREREGISTRATION} at HEAD."
        )


def hybrid_configuration(settings: Settings) -> HybridConfiguration:
    """The hybrid that comparison 1 put in production, from the tracked reports.

    Raises:
        SystemExit: If the tuning's report is not the one the pre-registration
            records by its hash, if the retrieval run on the test split did not
            put this hybrid in production with the BM25 the answer step uses, or
            if the hybrid's passages are not the corpus's own, which the answer
            step quotes from.
    """
    tuning_path = settings.tracked.reports / HYBRID_TUNING_REPORT
    test_path = settings.tracked.reports / RETRIEVAL_TEST_REPORT
    for path in (tuning_path, test_path):
        if not path.is_file():
            raise SystemExit(f"{path} does not exist.")
    tuning_sha256 = file_text_sha256(tuning_path)
    if tuning_sha256 not in committed_preregistration():
        raise SystemExit(f"{HYBRID_TUNING_REPORT} is not the tuning the pre-registration records.")
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    chosen = tuning["chosen"]
    if test.get("production") != HYBRID or test.get("tuning_sha256") != tuning_sha256:
        raise SystemExit(
            "The retrieval run on the test split did not put this hybrid in production."
        )
    ran = (test["candidate"], test["target_words"], test["k"])
    if ran != (tuning["candidate"], chosen["target_words"], chosen["k"]):
        raise SystemExit("The retrieval run on the test split ran another hybrid than the tuning.")
    answer, bm25 = settings.answer, test["bm25"]
    if (bm25["analyzer"], bm25["k1"], bm25["b"]) != (answer.analyzer, answer.k1, answer.b):
        raise SystemExit("The hybrid's BM25 is not the one the answer step is set to.")
    if chosen["target_words"] != settings.passages.target_words:
        raise SystemExit("The hybrid's passages are not the corpus's own, which are quoted from.")
    if tuning["window"] != settings.vectors.window:
        raise SystemExit("The settings' window is not the one the hybrid was tuned with.")
    return HybridConfiguration(
        embedding=tuning["candidate"],
        target_words=chosen["target_words"],
        k=chosen["k"],
        window=tuning["window"],
        tuning_sha256=tuning_sha256,
    )


def load_hybrid_index(
    settings: Settings, config: HybridConfiguration, table: PassageTable
) -> tuple[VectorIndex, str]:
    """The vectors of the corpus's passages, and the hash of their matrix.

    Raises:
        ValueError: If the index does not match its manifest or the passages file.
    """
    index, manifest = load_vector_index(
        index_path(settings.data.search, f"full-{config.target_words}", config.embedding),
        table.passage_ids,
    )
    return index, manifest.vectors_sha256


def ask_for_vectors(
    settings: Settings, embedding: str, texts: Sequence[str], env_file: Path | None
) -> tuple[np.ndarray, list[Answered], EmbeddingUsage, tuple[str, ...]]:
    """Ask the API for each text's vector, one text per request over one connection.

    This is how the page asks for a question's vector, so the waits recorded
    are the ones its questions would have.

    Returns:
        The vectors in the order of *texts*, how each was answered, what the
        requests used, and the providers that answered.
    """
    candidate = settings.vectors.candidate(embedding)
    key = load_openrouter_key(env_file or settings.data.search.parents[1] / ".env")
    embedder = build_embedder(
        candidate, key=key, cap_usd=settings.vectors.run_cap_usd, batch_inputs=1, concurrency=1
    )
    if not isinstance(embedder, OpenRouterEmbedder):
        raise SystemExit(f"{embedding} does not run through the API, as the page's model does.")
    vectors, answered = embedder.embed_singly(texts, "query")
    return vectors, answered, embedder.take_usage(), tuple(sorted(embedder.providers))


def question_vectors(
    settings: Settings,
    config: HybridConfiguration,
    questions: Sequence[EvalQuestion],
    split: str,
    env_file: Path | None,
) -> tuple[QueryVectors, QueryVectorsRecord]:
    """Each question's vector, as the page asks for it: asked for once, then read back.

    Raises:
        SystemExit: If the recorded vectors are of other questions or another
            model, or their file no longer matches its hash.
    """
    directory = work_dir(settings, split, HYBRID)
    record_path = directory / QUERY_RECORD_NAME
    vectors_path = directory / QUERY_VECTORS_NAME
    ids = [question.question_id for question in questions]
    if record_path.is_file():
        record = QueryVectorsRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        if list(record.question_ids) != ids or record.embedding != config.embedding:
            raise SystemExit(f"{record_path} holds other questions' vectors, or another model's.")
        if file_sha256(vectors_path) != record.vectors_sha256:
            raise SystemExit(f"{vectors_path} no longer matches its hash.")
        return QueryVectors.of(ids, np.load(vectors_path, allow_pickle=False)), record

    candidate = settings.vectors.candidate(config.embedding)
    logger.info("Asking for %d question vectors one by one", len(ids))
    started = now()
    vectors, answered, usage, providers = ask_for_vectors(
        settings, config.embedding, [question.text for question in questions], env_file
    )
    finished = now()
    directory.mkdir(parents=True, exist_ok=True)
    record = QueryVectorsRecord(
        embedding=config.embedding,
        model=candidate.model,
        revision=candidate.revision,
        provider=str(candidate.provider),
        quantization=str(candidate.quantization),
        question_ids=tuple(ids),
        vectors_sha256=save_array(vectors_path, np.asarray(vectors, dtype=np.float32)),
        seconds={qid: round(one.seconds, 4) for qid, one in zip(ids, answered, strict=True)},
        attempts={qid: one.attempts for qid, one in zip(ids, answered, strict=True)},
        requests=usage.requests,
        tokens=usage.tokens,
        cost_usd=round(usage.cost_usd, 6),
        providers=providers,
        started=started,
        finished=finished,
    )
    write_report(record, record_path)
    logger.info(
        "%d vectors in %d requests, %.6f USD; the longest wait %.1f s",
        len(ids),
        record.requests,
        record.cost_usd,
        max(record.seconds.values(), default=0.0),
    )
    return QueryVectors.of(ids, np.load(vectors_path, allow_pickle=False)), record


def make_hybrid_candidates(
    settings: Settings, manifest: QuestionsManifest, split: str, env_file: Path | None
) -> None:
    """Rank each question's candidates with the hybrid and record them.

    A question gets the ten passages the hybrid ranks highest under its filter,
    each with its BM25 score. A question whose vector came later than the page
    waits gets BM25's ten, as the page gives it.
    """
    check_hybrid_registered()
    check_tuned(settings)
    config = hybrid_configuration(settings)
    index, index_manifest, table = load_retrieval(settings)
    vector_index, vectors_sha256 = load_hybrid_index(settings, config, table)
    params = bm25_params(settings)
    masks = FilterMasks(table)
    questions = questions_of(settings, split)
    vectors, query_record = question_vectors(settings, config, questions, split, env_file)
    late = {qid for qid, seconds in query_record.seconds.items() if seconds > VECTOR_WAIT_S}
    timely = [question for question in questions if question.question_id not in late]
    words = lexical_hits(index, params, masks, timely, config.window)
    by_vector = vector_hits(vector_index, masks, timely, vectors, config.window)
    hits: dict[str, list[Hit]] = {
        question.question_id: fuse_candidates(
            index, question.text, params, lexical, vector, config.k
        )
        for question, lexical, vector in zip(timely, words, by_vector, strict=True)
    }
    for question in questions:
        if question.question_id in late:
            hits[question.question_id] = rank_passages(
                index, question.text, params, masks.of(question)
            )
    directory = work_dir(settings, split, HYBRID)
    records = record_candidates(settings, questions, hits, table, directory)
    run = HybridCandidatesRun(
        **bm25_run(settings, manifest, index_manifest, params, len(records)).model_dump(),
        embedding=config.embedding,
        target_words=config.target_words,
        rrf_k=config.k,
        window=config.window,
        tuning_sha256=config.tuning_sha256,
        vectors_sha256=vectors_sha256,
        query_vectors_sha256=query_record.vectors_sha256,
        bm25_alone=tuple(sorted(late)),
    )
    write_report(run, directory / RUN_NAME)
    logger.info(
        "%d of %d questions waited longer than %g s for their vector and got BM25's ten",
        len(late),
        len(questions),
        VECTOR_WAIT_S,
    )


def run_hybrid_baseline(settings: Settings) -> None:
    """Choose the baseline's threshold again on the development split, from the hybrid's candidates.

    The rule is the one of 2026-11-22: the highest balanced accuracy, and a tie
    to the lowest value. Once the run on the test split from the hybrid's
    candidates has begun, the threshold stays.
    """
    check_hybrid_registered()
    if (work_dir(settings, TEST_SPLIT, HYBRID) / ROUND_NAME).is_file():
        raise SystemExit(
            "The run on the test split from the hybrid's candidates has begun; its threshold stays."
        )
    _, records = read_candidates(settings, "dev", HYBRID)
    _, choice = baseline_selector(settings, records)
    write_report(choice, work_dir(settings, "dev", HYBRID) / THRESHOLD_NAME)
    logger.info(
        "Threshold %.4f, balanced accuracy %.3f, from the hybrid's candidates",
        choice.threshold,
        choice.balanced_accuracy,
    )


# --- The test run --------------------------------------------------------------------


def run_cost(settings: Settings, directory: Path, price: Prices) -> float:
    """What the batches of one run on the test split have cost so far."""
    return sum(
        cost_usd(completion.usage, price, _discount(settings))
        for path in sorted(directory.glob("completions_*.jsonl"))
        for completion in read_log(path, Completion)
    )


def held_out_round(directory: Path, chosen: RoundRecord) -> RoundRecord:
    """A test run's record: read back, or begun now with the chosen round's settings."""
    record_path = directory / ROUND_NAME
    if record_path.is_file():
        return RoundRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    record = RoundRecord(
        number=chosen.number,
        model=chosen.model,
        effort=chosen.effort,
        max_tokens=chosen.max_tokens,
        prompt_sha256=chosen.prompt_sha256,
        created=now(),
    )
    write_report(record, record_path)
    return record


def check_test_budget(
    settings: Settings, directory: Path, model: str, high_estimate: float, cap_usd: float | None
) -> None:
    """Stop unless the test run has a cap, and this batch could stay within it."""
    if cap_usd is None:
        raise SystemExit("The test run needs a cap: pass --max-usd.")
    so_far = run_cost(settings, directory, settings.generation.prices[model])
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
        write_lines(baseline_path, baseline_answers(settings, baseline, records))

    price = settings.generation.prices[chosen.model]
    selector = ClaudeSelector(prompt, chosen.model, chosen.effort, chosen.max_tokens, price)
    record = held_out_round(directory, chosen)
    answers = answer_all(
        settings, selector, baseline, records, directory, record, cap_usd, TEST_SPLIT
    )
    write_lines(selector_path, answers)

    record = RoundRecord.model_validate_json((directory / ROUND_NAME).read_text(encoding="utf-8"))
    cost = run_cost(settings, directory, price)
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


def run_hybrid_test(
    settings: Settings, manifest: QuestionsManifest, cap_usd: float | None, env_file: Path | None
) -> None:
    """Answer the test split once more, from the hybrid's candidates.

    The addendum on the hybrid's candidates sets this run out. The model, the
    prompt, the effort and the output limit are those of the first run on the
    test split; its record is read for them, and its answers are never read.
    The baseline's threshold is the one chosen again on the development split
    from the hybrid's candidates. Only what ran is recorded.
    """
    check_hybrid_registered()
    directory = work_dir(settings, TEST_SPLIT, HYBRID)
    selector_path = answers_path(settings, "selector", TEST_SPLIT, HYBRID)
    if selector_path.is_file():
        raise SystemExit(
            "The test split has been answered from the hybrid's candidates once, "
            "which is all the addendum allows."
        )
    first_path = settings.tracked.reports / TEST_RUN_REPORT
    if not first_path.is_file():
        raise SystemExit(
            f"{first_path} does not exist; the first run on the test split comes first."
        )
    first = SealedRun.model_validate_json(first_path.read_text(encoding="utf-8"))
    check_round_settings(settings)
    check_judge_known(settings)
    chosen = dev_choice(settings)
    name = run_name(chosen.number, chosen.model)
    if (chosen.model, name, chosen.prompt_sha256, chosen.effort, chosen.max_tokens) != (
        first.model,
        first.round,
        first.prompt_sha256,
        first.effort,
        first.max_tokens,
    ):
        raise SystemExit(
            "This run would differ from the first run on the test split in its model, "
            "its prompt, its effort or its output limit."
        )
    prompt = load_selector_prompt(round_dir(settings, name))
    if prompt.sha256 != chosen.prompt_sha256:
        raise SystemExit(f"The copy of {name}'s prompt no longer matches its hash.")
    threshold = read_threshold(settings, HYBRID)

    if not (directory / RUN_NAME).is_file():
        make_hybrid_candidates(settings, manifest, TEST_SPLIT, env_file)
    retrieval, records = read_candidates(settings, TEST_SPLIT, HYBRID)
    index, _, _ = load_retrieval(settings)
    baseline = BaselineSelector(index, bm25_params(settings), threshold=threshold.threshold)
    baseline_path = answers_path(settings, "baseline", TEST_SPLIT, HYBRID)
    if not baseline_path.is_file():
        write_lines(baseline_path, baseline_answers(settings, baseline, records))

    price = settings.generation.prices[chosen.model]
    selector = ClaudeSelector(prompt, chosen.model, chosen.effort, chosen.max_tokens, price)
    record = held_out_round(directory, chosen)
    answers = answer_all(
        settings, selector, baseline, records, directory, record, cap_usd, TEST_SPLIT
    )
    write_lines(selector_path, answers)

    record = RoundRecord.model_validate_json((directory / ROUND_NAME).read_text(encoding="utf-8"))
    cost = run_cost(settings, directory, price)
    report = HybridTestRun(
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
    write_report(report, settings.tracked.reports / HYBRID_TEST_RUN_REPORT)
    logger.info(
        "The test split is answered from the hybrid's candidates: %d questions with the "
        "prompt of %s, %.2f USD",
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
    """Grade every pair without a grade, sending a failed one once more, under *cap_usd*.

    The judge grades only once its kappa against the hand grades has reached
    0.6. Short of it, quote precision is not measured on the development split,
    so nothing the judge would grade is used.
    """
    check_judge_known(settings)
    if judge_accepted(settings) is False:
        raise SystemExit(
            "The judge's kappa fell short of 0.6, so it grades nothing more: quote "
            "precision is not measured on the development split."
        )
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
    measured: bool = True,
) -> SelectorSummary:
    """One selector's or round's metrics.

    Quote precision, the score and the launch rule's part on quote precision
    stand under their registered names only when *measured*, that is when the
    judge's kappa reached 0.6; :func:`judge_figures` gives them otherwise.
    """
    shares = abstention(outcomes)
    precision = quote_precision(outcomes) if pending_grades == 0 and measured else None
    rule: dict[str, bool | None] = dict(launch_rule(shares, precision))
    if not measured:
        rule[PRECISION_PART] = None
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
        launch_rule=rule,
    )


def judge_figures(outcomes: Sequence[QuestionOutcome], pending_grades: int) -> JudgeFigures:
    """One selector's or round's figures under the judge's grades, as description."""
    shares = abstention(outcomes)
    precision = quote_precision(outcomes) if pending_grades == 0 else None
    return JudgeFigures(
        quote_precision=None if precision is None else round(precision, 6),
        score=None if precision is None else round((precision + shares.balanced_accuracy) / 2, 6),
        quote_precision_at_least_085=launch_rule(shares, precision)[PRECISION_PART],
    )


def launch_rule(shares: Abstention, precision: float | None) -> dict[str, bool]:
    """Each part of the launch rule, read on the point estimates.

    The shares are compared to nine decimals, so that a share exactly at a
    limit is not put past it by the rounding of floats.
    """
    return {
        "abstains_on_90_percent_of_unanswerable": round(shares.abstains_unanswerable, 9) >= 0.90,
        "needless_abstention_at_most_15_percent": round(shares.needless_abstention, 9) <= 0.15,
        "quote_precision_at_least_085": precision is not None and round(precision, 9) >= 0.85,
    }


def launch_decision(
    precision: tuple[float, float, float] | None,
    accuracy: tuple[float, float, float],
    launch_rules: Mapping[str, Mapping[str, bool | None]],
) -> tuple[bool, str, bool]:
    """Whether the language model beats the baseline, which selector runs, and if it launches.

    The language model beats the baseline when the lower bound of both
    intervals lies above 0. The selector that runs is the language model if it
    does and the baseline otherwise, and the launch rule is read for it.

    Args:
        precision: The difference in quote precision, language model minus
            baseline, with its interval; ``None`` when a selector has no
            graded quotation.
        accuracy: The same for balanced accuracy.
        launch_rules: Each selector's launch rule, by name.

    Returns:
        Whether it beats the baseline, the selector that runs, and whether
        every part of the launch rule holds for that selector.
    """
    beats = precision is not None and precision[1] > 0 and accuracy[1] > 0
    runs = "selector" if beats else "baseline"
    return beats, runs, all(launch_rules[runs].values())


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
    measured = judge_accepted(settings) is True
    summaries = tuple(
        summarize(name, outcomes[name], rounds.get(name), waiting[name], measured) for name in names
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
    accepted = judge_accepted(settings)

    comparisons = []
    judge_intervals: dict[str, tuple[float, float, float] | None] = {}
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
        judge_intervals[name] = precision
        comparisons.append(
            Comparison(
                name=name,
                quote_precision=precision if accepted is True else None,
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
    with_a_grade_2 = (sum(any(value == 2 for value in values) for values in graded), len(graded))
    unaccepted = None
    if accepted is False:
        unaccepted = UnacceptedJudge(
            selectors={name: judge_figures(outcomes[name], waiting[name]) for name in names},
            comparisons=judge_intervals,
            answerable_with_a_grade_2=with_a_grade_2,
        )
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
        answerable_with_a_grade_2=None if accepted is False else with_a_grade_2,
        chosen_model=chosen,
        leading_round=leading_round(summaries, rounds, chosen, accepted),
        cost_usd=spent(settings),
        resamples=answer.resamples,
        bootstrap_seed=answer.bootstrap_seed,
        unaccepted_judge=unaccepted,
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


# --- Grading the test run by hand ------------------------------------------------------


def hybrid_test_answers(
    settings: Settings,
) -> tuple[HybridTestRun, str, list[QuestionCandidates], dict[str, dict[str, Answer]]]:
    """The run from the hybrid's candidates: its record, the record's hash, its candidates, answers.

    Raises:
        SystemExit: If the record is missing, or an answers file is not the one it names.
    """
    path = settings.tracked.reports / HYBRID_TEST_RUN_REPORT
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist; the run from the hybrid's candidates comes first."
        )
    run = HybridTestRun.model_validate_json(path.read_text(encoding="utf-8"))
    answers: dict[str, dict[str, Answer]] = {}
    for name in SELECTORS:
        answers_file = answers_path(settings, name, TEST_SPLIT, HYBRID)
        if not answers_file.is_file():
            raise SystemExit(f"{answers_file} does not exist.")
        if hashlib.sha256(answers_file.read_bytes()).hexdigest() != run.answers_sha256[name]:
            raise SystemExit(f"{answers_file} is not the file the run's record names.")
        answers[name] = read_answers(settings, name, TEST_SPLIT, HYBRID)
    _, records = read_candidates(settings, TEST_SPLIT, HYBRID)
    return run, file_text_sha256(path), records, answers


def make_hand_draw(settings: Settings) -> tuple[HandDraw, list[GradingPair]]:
    """Draw the questions graded by hand, and order the pairs of their quotations."""
    run, run_sha256, records, answers = hybrid_test_answers(settings)
    drawn = draw_questions(
        (record.question_id for record in records if record.answerable), HAND_SEED, HAND_QUESTIONS
    )
    chosen = [record for record in records if record.question_id in set(drawn)]
    ordered = order_pairs(
        (pair for name in SELECTORS for pair in quote_pairs(chosen, answers[name])), HAND_SEED
    )
    draw = HandDraw(
        seed=HAND_SEED,
        run_sha256=run_sha256,
        answers_sha256=dict(run.answers_sha256),
        questions=tuple(drawn),
        pairs=name_pairs(ordered),
    )
    return draw, ordered


def hand_dir(settings: Settings) -> Path:
    """Where the form and the answers it handed over are kept."""
    return work_dir(settings, TEST_SPLIT, HYBRID) / HAND_DIR


def run_hand_form(settings: Settings) -> None:
    """Draw the questions whose quotations are graded by hand, record the draw, write the form.

    The draw follows from the seed, so a second call writes the same form. A
    recorded draw that differs from it stops the call.
    """
    check_hybrid_registered()
    draw, ordered = make_hand_draw(settings)
    path = settings.tracked.reports / HAND_DRAW_REPORT
    if path.is_file() and HandDraw.model_validate_json(path.read_text(encoding="utf-8")) != draw:
        raise SystemExit(f"{path} records another draw, and the draw is made once.")
    write_report(draw, path)
    form = hand_dir(settings) / f"{HAND_FORM_KEY}.html"
    form.parent.mkdir(parents=True, exist_ok=True)
    form.write_text(hand_form(ordered, draw.pairs, HAND_FORM_KEY), encoding="utf-8", newline="\n")
    logger.info(
        "%d questions drawn and %d pairs to grade; the form is %s",
        len(draw.questions),
        len(draw.pairs),
        form,
    )


def hand_summary(name: str, outcomes: Sequence[QuestionOutcome], drawn: set[str]) -> HandSummary:
    """One selector's metrics on the test split, with quote precision on the drawn questions."""
    shares = abstention(outcomes)
    graded = [outcome for outcome in outcomes if outcome.question_id in drawn]
    precision = quote_precision(graded)
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    cost = sum(outcome.cost_usd for outcome in outcomes)
    return HandSummary(
        name=name,
        abstention=shares,
        quote_precision=None if precision is None else round(precision, 6),
        graded_quotes=sum(len(outcome.quotes) for outcome in graded),
        quotes=sum(len(outcome.quotes) for outcome in answerable),
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
        launch_rule=launch_rule(shares, precision),
    )


def run_test_report(settings: Settings, answers_file: Path) -> None:
    """Compute the comparisons and the launch rule of the test run with the hand grades.

    The draw's record must be committed as it stands, so the pairs were fixed
    before the grades were read. Quote precision is computed over each
    selector's quotations on the drawn questions, and its paired bootstrap
    resamples them; balanced accuracy is computed on every question of the test
    split. The language model beats the baseline when the lower bound of both
    intervals lies above 0, and the launch rule is read on the point estimates
    of the selector that runs.
    """
    check_hybrid_registered()
    draw_path = settings.tracked.reports / HAND_DRAW_REPORT
    require_committed(draw_path)
    recorded = HandDraw.model_validate_json(draw_path.read_text(encoding="utf-8"))
    draw, _ = make_hand_draw(settings)
    if draw != recorded:
        raise SystemExit("The run's answers no longer give the recorded draw.")
    try:
        hand = read_hand_grades(answers_file, draw, HAND_FORM_KEY)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    run, run_sha256, records, answers = hybrid_test_answers(settings)
    drawn = set(draw.questions)
    outcomes = {name: outcomes_of(records, answers[name], hand) for name in SELECTORS}
    summaries = tuple(hand_summary(name, outcomes[name], drawn) for name in SELECTORS)
    graded = {
        name: [outcome for outcome in outcomes[name] if outcome.question_id in drawn]
        for name in SELECTORS
    }
    answer = settings.answer
    precision: tuple[float, float, float] | None = None
    if all(summary.graded_quotes for summary in summaries):
        precision = _rounded(
            precision_difference(
                graded["selector"], graded["baseline"], answer.bootstrap_seed, answer.resamples
            )
        )
    accuracy = _rounded(
        accuracy_difference(
            outcomes["selector"], outcomes["baseline"], answer.bootstrap_seed, answer.resamples
        )
    )
    beats, runs, launch = launch_decision(
        precision, accuracy, {summary.name: summary.launch_rule for summary in summaries}
    )
    report = AnswerTestReport(
        split=TEST_SPLIT,
        run_sha256=run_sha256,
        draw_sha256=file_text_sha256(draw_path),
        grades_sha256=hashlib.sha256(answers_file.read_bytes()).hexdigest(),
        retrieval=run.retrieval,
        threshold=run.threshold,
        drawn_questions=len(draw.questions),
        graded_pairs=len(draw.pairs),
        grades={str(grade): sum(value == grade for value in hand.values()) for grade in GRADES},
        selectors=summaries,
        comparison=Comparison(
            name="selector", quote_precision=precision, balanced_accuracy=accuracy
        ),
        beats_baseline=beats,
        runs=runs,
        launch=launch,
        candidate_parties=party_shares(
            candidate.party
            for record in records
            if record.answerable
            for candidate in record.candidates
        ),
        resamples=answer.resamples,
        bootstrap_seed=answer.bootstrap_seed,
    )
    write_report(report, settings.tracked.reports / TEST_REPORT)
    lines = [
        json.dumps(
            {"run": name, "drawn": outcome.question_id in drawn, **outcome.model_dump(mode="json")},
            ensure_ascii=False,
        )
        for name in SELECTORS
        for outcome in outcomes[name]
    ]
    (settings.tracked.reports / TEST_QUESTIONS_REPORT).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    kept = hand_dir(settings) / f"{HAND_FORM_KEY}.json"
    if answers_file.resolve() != kept.resolve():
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(answers_file, kept)
    for summary in summaries:
        logger.info(
            "%-8s balanced accuracy %.3f (abstains on %.2f of unanswerable, answers %.2f of "
            "answerable), quote precision %s over %d graded quotes",
            summary.name,
            summary.abstention.balanced_accuracy,
            summary.abstention.abstains_unanswerable,
            summary.abstention.answers_answerable,
            "n/a" if summary.quote_precision is None else f"{summary.quote_precision:.3f}",
            summary.graded_quotes,
        )
    logger.info(
        "The language model %s the baseline; %s runs, and the launch rule %s",
        "beats" if beats else "does not beat",
        runs,
        "holds" if report.launch else "does not hold",
    )


def run_sealed_description(settings: Settings) -> None:
    """Open the first run on the test split, and describe how its selectors abstain.

    The addendum on the hybrid's candidates opens the first run only after the
    comparisons and the launch rule of the second have been computed with the
    hand grades, so the test report must be committed as it stands first. The
    first run decides nothing: the balanced accuracy and the two shares behind
    it are description, and its quotations are not graded.
    """
    check_hybrid_registered()
    report_path = settings.tracked.reports / TEST_REPORT
    require_committed(report_path)
    first_path = settings.tracked.reports / TEST_RUN_REPORT
    first = SealedRun.model_validate_json(first_path.read_text(encoding="utf-8"))
    for name in SELECTORS:
        path = answers_path(settings, name, TEST_SPLIT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != first.answers_sha256[name]:
            raise SystemExit(f"{path} is not the file the sealed record names.")
    _, records = read_candidates(settings, TEST_SPLIT)
    shares = {
        name: abstention(outcomes_of(records, read_answers(settings, name, TEST_SPLIT), {}))
        for name in SELECTORS
    }
    description = SealedDescription(
        report_sha256=file_text_sha256(report_path), run=first, abstention=shares
    )
    write_report(description, settings.tracked.reports / SEALED_REPORT)
    for name in SELECTORS:
        logger.info(
            "First run, %-8s balanced accuracy %.3f (abstains on %.2f of unanswerable, "
            "answers %.2f of answerable)",
            name,
            shares[name].balanced_accuracy,
            shares[name].abstains_unanswerable,
            shares[name].answers_answerable,
        )


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
        help="The .env file that holds the API keys. Defaults to .env in the data root.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    hybrid_help = "From the hybrid's candidates, as the addendum on them sets out."
    candidates = commands.add_parser(
        "candidates", help="Rank and record each question's candidates."
    )
    candidates.add_argument("--hybrid", action="store_true", help=hybrid_help)
    candidates.add_argument(
        "--confirm", action="store_true", help="Required with --hybrid: this calls the API."
    )
    baseline = commands.add_parser(
        "baseline", help="Choose the threshold and answer with the baseline."
    )
    baseline.add_argument("--hybrid", action="store_true", help=hybrid_help)
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
    test_run.add_argument("--hybrid", action="store_true", help=hybrid_help)
    test_run.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    commands.add_parser("hand-form", help="Draw the questions graded by hand and write the form.")
    test_report = commands.add_parser(
        "test-report", help="Compute the test run's comparisons with the hand grades."
    )
    test_report.add_argument(
        "--answers", type=Path, required=True, help="The answers the form handed over, as JSON."
    )
    commands.add_parser(
        "open-sealed", help="Describe the first run on the test split, after the test report."
    )
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    manifest = check_registered(settings)
    if args.command == "candidates" and args.hybrid:
        _confirmed(args, "Ranking with the hybrid")
        make_hybrid_candidates(settings, manifest, "dev", args.env_file)
    elif args.command == "candidates":
        make_candidates(settings, manifest)
    elif args.command == "baseline" and args.hybrid:
        run_hybrid_baseline(settings)
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
        if args.hybrid:
            run_hybrid_test(settings, manifest, args.max_usd, args.env_file)
        else:
            run_test(settings, manifest, args.max_usd)
    elif args.command == "hand-form":
        run_hand_form(settings)
    elif args.command == "test-report":
        run_test_report(settings, args.answers)
    elif args.command == "open-sealed":
        run_sealed_description(settings)
    else:
        make_report(settings)


if __name__ == "__main__":
    main()
