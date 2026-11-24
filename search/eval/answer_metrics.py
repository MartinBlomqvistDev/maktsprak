"""What the answer step is measured on: quote precision, abstention and party balance.

The definitions are the addendum's of 2026-11-22 in
``docs/search/PREREGISTRATION.md``:

- A selector abstains when its answer holds no quotation. It abstains
  needlessly when it abstains on an answerable question, whatever its
  candidates hold.
- Balanced accuracy is the mean of two shares: the unanswerable questions on
  which a selector abstains, and the answerable ones on which it does not.
- Quote precision is the share of a selector's quotations on answerable
  questions that the judge grades 2. A quotation the judge did not grade is
  not graded 2. A quotation given to an unanswerable question is not graded:
  it counts as a failure to abstain.

Selection by a language model beats the baseline when, on the test split, the
95% interval of the difference has a lower bound above 0 for quote precision
and for balanced accuracy alike, each under a paired bootstrap with 10 000
resamples. :func:`precision_difference` resamples the answerable questions
and computes each selector's precision over its quotations on the questions
drawn; :func:`accuracy_difference` resamples the answerable and the
unanswerable questions apart, so each keeps its size.

The addendum of 2026-11-23 runs three models in round 1 and chooses one of
them with :func:`choose_model`, unless the judge's kappa falls short of 0.6.
The rounds after it are chosen among with :func:`choose_round`, or with
:func:`choose_round_by_accuracy` when the kappa falls short.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.eval.stats import LEVEL, RESAMPLES
from search.parties import PARTY_ORDER

#: The grade that counts as answering the question.
ANSWERS = 2

#: How far below the best a model may come, in score and in quote precision,
#: and still be chosen after round 1.
MODEL_MARGIN = 0.01


class QuoteOutcome(BaseModel):
    """One quotation an answer gave, and its grade.

    Attributes:
        passage_id: The candidate quoted from.
        first_paragraph: The first paragraph quoted, numbered from 1 within the passage.
        last_paragraph: The last paragraph quoted.
        party: The speech's party.
        text_sha256: The hash of the quotation's text.
        grade: The judge's grade, or ``None`` when it has none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str
    first_paragraph: int
    last_paragraph: int
    party: str
    text_sha256: str
    grade: int | None = None


class QuestionOutcome(BaseModel):
    """How one selector answered one question.

    Attributes:
        question_id: The question.
        question_set: Its set.
        answerable: Whether it has a gold speech.
        selector: The selector that answered.
        fallback: Why the baseline answered in place of the language model, if it did.
        quotes: The quotations given.
        rejected: Why each rejected selection, or the output, was rejected.
        cost_usd: What the requests for the question cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    question_set: str
    answerable: bool
    selector: str
    fallback: str | None = None
    quotes: tuple[QuoteOutcome, ...]
    rejected: tuple[str, ...] = ()
    cost_usd: float = 0.0

    @property
    def abstained(self) -> bool:
        """Whether the answer holds no quotation."""
        return not self.quotes


class Abstention(BaseModel):
    """How a selector abstains.

    Attributes:
        balanced_accuracy: The mean of the two shares below.
        abstains_unanswerable: The share of unanswerable questions abstained on.
        answers_answerable: The share of answerable questions answered.
        needless_abstention: The share of answerable questions abstained on.
        answerable: How many answerable questions.
        unanswerable: How many unanswerable ones.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    balanced_accuracy: float
    abstains_unanswerable: float
    answers_answerable: float
    needless_abstention: float
    answerable: int
    unanswerable: int


def _split(outcomes: Sequence[QuestionOutcome]) -> tuple[list[QuestionOutcome], ...]:
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]
    if not answerable or not unanswerable:
        raise ValueError("Abstention needs answerable and unanswerable questions.")
    return answerable, unanswerable


def abstention(outcomes: Sequence[QuestionOutcome]) -> Abstention:
    """Balanced accuracy and its two shares.

    Raises:
        ValueError: If the outcomes lack answerable or unanswerable questions.
    """
    answerable, unanswerable = _split(outcomes)
    abstains = sum(outcome.abstained for outcome in unanswerable) / len(unanswerable)
    answers = sum(not outcome.abstained for outcome in answerable) / len(answerable)
    return Abstention(
        balanced_accuracy=(abstains + answers) / 2,
        abstains_unanswerable=abstains,
        answers_answerable=answers,
        needless_abstention=1 - answers,
        answerable=len(answerable),
        unanswerable=len(unanswerable),
    )


def precision_counts(outcomes: Sequence[QuestionOutcome]) -> list[tuple[int, int]]:
    """For each answerable question, its quotations graded 2 and all its quotations."""
    return [
        (sum(quote.grade == ANSWERS for quote in outcome.quotes), len(outcome.quotes))
        for outcome in outcomes
        if outcome.answerable
    ]


def quote_precision(outcomes: Sequence[QuestionOutcome]) -> float | None:
    """The share of quotations on answerable questions graded 2, or ``None`` without quotations."""
    counts = precision_counts(outcomes)
    total = sum(quotes for _, quotes in counts)
    return sum(good for good, _ in counts) / total if total else None


def ungraded(outcomes: Sequence[QuestionOutcome]) -> int:
    """How many quotations on answerable questions have no grade."""
    return sum(
        quote.grade is None
        for outcome in outcomes
        if outcome.answerable
        for quote in outcome.quotes
    )


def _check_paired(first: Sequence[QuestionOutcome], second: Sequence[QuestionOutcome]) -> None:
    if [outcome.question_id for outcome in first] != [outcome.question_id for outcome in second]:
        raise ValueError("A paired comparison needs the same questions in the same order.")


def _percentiles(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return float(low), float(high)


def precision_difference(
    first: Sequence[QuestionOutcome],
    second: Sequence[QuestionOutcome],
    seed: int,
    resamples: int = RESAMPLES,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """The difference in quote precision, first minus second, with its paired interval.

    Each resample draws the answerable questions with replacement, the same
    draw for both selectors, and computes each one's precision over its
    quotations on the questions drawn. A resample in which a selector has no
    quotation has no precision and is left out of the interval.

    Args:
        first: One selector's outcomes.
        second: The other's, question by question.
        seed: Decides the resamples.
        resamples: How many resamples.
        level: The interval's coverage.

    Returns:
        The difference on all questions, and the interval's lower and upper bound.

    Raises:
        ValueError: If the questions differ, or either selector gives no quotation.
    """
    _check_paired(first, second)
    a = np.array(precision_counts(first), dtype=np.float64)
    b = np.array(precision_counts(second), dtype=np.float64)
    if a.size == 0 or a[:, 1].sum() == 0 or b[:, 1].sum() == 0:
        raise ValueError("Both selectors need at least one quotation on an answerable question.")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(a), size=(resamples, len(a)))
    with np.errstate(invalid="ignore", divide="ignore"):
        drawn_a = a[picks, 0].sum(axis=1) / a[picks, 1].sum(axis=1)
        drawn_b = b[picks, 0].sum(axis=1) / b[picks, 1].sum(axis=1)
    point = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    low, high = _percentiles(drawn_a - drawn_b, level)
    return float(point), low, high


def accuracy_difference(
    first: Sequence[QuestionOutcome],
    second: Sequence[QuestionOutcome],
    seed: int,
    resamples: int = RESAMPLES,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """The difference in balanced accuracy, first minus second, with its paired interval.

    Each resample draws the answerable and the unanswerable questions apart,
    each with replacement and at its own size, the same draw for both selectors.

    Args:
        first: One selector's outcomes.
        second: The other's, question by question.
        seed: Decides the resamples.
        resamples: How many resamples.
        level: The interval's coverage.

    Returns:
        The difference on all questions, and the interval's lower and upper bound.

    Raises:
        ValueError: If the questions differ, or a kind of question is missing.
    """
    _check_paired(first, second)
    answerable_a, unanswerable_a = _split(first)
    answerable_b, unanswerable_b = _split(second)

    def answered(outcomes: list[QuestionOutcome]) -> np.ndarray:
        return np.array([not outcome.abstained for outcome in outcomes], dtype=np.float64)

    def abstained(outcomes: list[QuestionOutcome]) -> np.ndarray:
        return np.array([outcome.abstained for outcome in outcomes], dtype=np.float64)

    rng = np.random.default_rng(seed)
    picks_answerable = rng.integers(0, len(answerable_a), size=(resamples, len(answerable_a)))
    picks_unanswerable = rng.integers(0, len(unanswerable_a), size=(resamples, len(unanswerable_a)))
    a_answerable, b_answerable = answered(answerable_a), answered(answerable_b)
    a_unanswerable, b_unanswerable = abstained(unanswerable_a), abstained(unanswerable_b)
    drawn_a = (
        a_answerable[picks_answerable].mean(axis=1)
        + a_unanswerable[picks_unanswerable].mean(axis=1)
    ) / 2
    drawn_b = (
        b_answerable[picks_answerable].mean(axis=1)
        + b_unanswerable[picks_unanswerable].mean(axis=1)
    ) / 2
    point = abstention(first).balanced_accuracy - abstention(second).balanced_accuracy
    low, high = _percentiles(drawn_a - drawn_b, level)
    return float(point), low, high


def party_shares(parties: Iterable[str]) -> dict[str, float]:
    """The share of each party, in the order of :data:`PARTY_ORDER`, parties absent at 0."""
    counts = Counter(parties)
    total = sum(counts.values())
    return {party: (counts[party] / total if total else 0.0) for party in PARTY_ORDER}


def choose_round(rounds: Sequence[tuple[int, float, float]]) -> int:
    """The round that goes to the test run: the highest mean of precision and balanced accuracy.

    Args:
        rounds: Each round's number, quote precision and balanced accuracy.

    Returns:
        The round's number. The means are compared to nine decimals, so that
        two equal means are not told apart by the rounding of floats, and a
        tie goes to the later round.

    Raises:
        ValueError: If there is no round.
    """
    if not rounds:
        raise ValueError("There is no round to choose.")
    return max(rounds, key=lambda item: (round((item[1] + item[2]) / 2, 9), item[0]))[0]


def choose_round_by_accuracy(rounds: Sequence[tuple[int, float]]) -> int:
    """The round that goes to the test run when the judge's kappa falls short of 0.6.

    Args:
        rounds: Each round's number and balanced accuracy.

    Returns:
        The number of the round with the highest balanced accuracy, compared to
        nine decimals; a tie goes to the later round.

    Raises:
        ValueError: If there is no round.
    """
    if not rounds:
        raise ValueError("There is no round to choose.")
    return max(rounds, key=lambda item: (round(item[1], 9), item[0]))[0]


def choose_model(
    scores: Mapping[str, tuple[float | None, float]],
    by_price: Sequence[str],
    margin: float = MODEL_MARGIN,
) -> str:
    """The model that goes on after round 1, as the addendum of 2026-11-23 sets out.

    A model's score is the mean of its quote precision and its balanced
    accuracy. A model qualifies if its score is at most *margin* below the
    highest score and its quote precision at most *margin* below the highest
    precision, and the cheapest that qualifies is chosen. A model that gives
    no quotation to any of the answerable questions has no precision, so it
    cannot qualify. The addendum leaves open the case in which no model
    qualifies, and then no model is chosen. Values are compared to nine
    decimals, as in :func:`choose_round`.

    Args:
        scores: Each model's quote precision, ``None`` when it gave no quotation
            to any of the answerable questions, and its balanced accuracy.
        by_price: The models from the cheapest to the most expensive.
        margin: How far below the best a model may come.

    Returns:
        The chosen model.

    Raises:
        ValueError: If a model has no place in *by_price*, no model has a
            quote precision, or no model qualifies.
    """
    missing = set(scores) - set(by_price)
    if missing:
        raise ValueError(f"No price order for {sorted(missing)}.")
    rated: dict[str, tuple[float, float]] = {}
    for model in by_price:
        precision, accuracy = scores.get(model, (None, 0.0))
        if precision is not None:
            rated[model] = (precision, accuracy)
    if not rated:
        raise ValueError("No model gave a quotation to any of the answerable questions.")

    def score(model: str) -> float:
        return round(sum(rated[model]) / 2, 9)

    def within(best: float, value: float) -> bool:
        return round(best - value, 9) <= margin

    top_score = max(score(model) for model in rated)
    top_precision = max(precision for precision, _ in rated.values())
    qualified = [
        model
        for model in rated
        if within(top_score, score(model)) and within(top_precision, rated[model][0])
    ]
    if not qualified:
        raise ValueError("No model qualifies, a case the addendum of 2026-11-23 leaves open.")
    return qualified[0]
