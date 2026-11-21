"""The BM25 baseline: tuning on the development split, and the run on it.

BM25 gets a fair chance before it is compared with anything
(``docs/search/PREREGISTRATION.md``): the analyzer (stemming, stop words), ``k1``
and ``b`` are tuned on the development split. The configuration chosen is the
one with the highest recall@10 on the natural questions; a tie goes to the
higher MRR@10, then to the configuration nearest Lucene's defaults (``k1`` 1.2,
``b`` 0.75, by the sum of the two distances), then to the analyzer listed first
in :data:`search.index.analyzer.ANALYZERS`.

Every set is then run once with that configuration. A filtered question is
searched among the passages its filter allows; every other question among all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.corpus.table import PassageTable
from search.eval.metrics import (
    DEPTH,
    METRIC_NAMES,
    mean_metrics,
    question_metrics,
    recall_at,
    reciprocal_rank,
    speech_ranking,
)
from search.eval.questions import EvalQuestion
from search.eval.stats import bootstrap_interval
from search.index.analyzer import ANALYZERS
from search.index.lexical import DEFAULT_PARAMS, PASSAGE_WINDOW, Bm25Params, LexicalIndex

#: Passages ranked per question while tuning, enough for ten distinct speeches.
TUNING_WINDOW = 200


class GridPoint(BaseModel):
    """One configuration tried in the tuning, and how it did.

    Attributes:
        analyzer: The analyzer's name.
        k1: BM25's ``k1``.
        b: BM25's ``b``.
        recall_10: Mean recall@10 over the tuning questions.
        mrr_10: Mean MRR@10 over them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyzer: str
    k1: float
    b: float
    recall_10: float
    mrr_10: float


def choose(grid: Sequence[GridPoint]) -> GridPoint:
    """Pick the configuration by the rule in the module's docstring."""
    order = [analyzer.name for analyzer in ANALYZERS]

    def key(point: GridPoint) -> tuple[float, float, float, int]:
        distance = abs(point.k1 - DEFAULT_PARAMS.k1) + abs(point.b - DEFAULT_PARAMS.b)
        return (-point.recall_10, -point.mrr_10, distance, order.index(point.analyzer))

    return min(grid, key=key)


class FilterMasks:
    """The passages each filter allows, computed once per filter."""

    def __init__(self, table: PassageTable) -> None:
        self.table = table
        self._masks: dict[tuple[str, str], np.ndarray] = {}

    def of(self, question: EvalQuestion) -> np.ndarray | None:
        """The mask of a question's filter, or ``None`` when it has none."""
        if question.filter is None:
            return None
        if question.filter.party is not None:
            key = ("party", question.filter.party)
        else:
            key = ("period", question.filter.period or "")
        if key not in self._masks:
            if key[0] == "party":
                self._masks[key] = self.table.party_mask(key[1])
            else:
                self._masks[key] = self.table.period_mask(key[1])
        return self._masks[key]


def rank(
    index: LexicalIndex,
    table: PassageTable,
    question: EvalQuestion,
    params: Bm25Params,
    masks: FilterMasks,
    depth: int = DEPTH,
    window: int = PASSAGE_WINDOW,
) -> list[str]:
    """The speeches BM25 ranks for a question, best first, under its filter."""
    hits = index.search(question.text, params, limit=window, allowed=masks.of(question))
    return speech_ranking((hit.passage for hit in hits), table.anf_ids, depth)


def tune(
    indexes: Mapping[str, LexicalIndex],
    table: PassageTable,
    questions: Sequence[EvalQuestion],
    k1_values: Sequence[float],
    b_values: Sequence[float],
) -> list[GridPoint]:
    """Try every analyzer and every pair of ``k1`` and ``b`` on the tuning questions.

    Args:
        indexes: One index per analyzer, by the analyzer's name.
        table: The passages.
        questions: The tuning questions.
        k1_values: The values of ``k1`` to try.
        b_values: The values of ``b`` to try.

    Returns:
        Every configuration and its scores, in the order tried.
    """
    masks = FilterMasks(table)
    grid = []
    for name, index in indexes.items():
        for k1 in k1_values:
            for b in b_values:
                params = Bm25Params(k1=k1, b=b)
                recalls, reciprocal = [], []
                for question in questions:
                    ranking = rank(index, table, question, params, masks, 10, TUNING_WINDOW)
                    recalls.append(recall_at(ranking, question.gold, 10))
                    reciprocal.append(reciprocal_rank(ranking, question.gold, 10))
                grid.append(
                    GridPoint(
                        analyzer=name,
                        k1=k1,
                        b=b,
                        recall_10=round(float(np.mean(recalls)), 6),
                        mrr_10=round(float(np.mean(reciprocal)), 6),
                    )
                )
    return grid


class QuestionResult(BaseModel):
    """What the run gave for one question.

    Attributes:
        question_id: The question.
        period: The period of its gold speech, when it has one.
        party: The party of its passage, for a question written from one.
        ranking: The speeches ranked, best first, up to the deepest rank measured.
        metrics: Each metric's value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    period: str | None
    party: str | None
    ranking: tuple[str, ...]
    metrics: dict[str, float]


class Summary(BaseModel):
    """The metrics of a group of questions, with 95% bootstrap intervals.

    Attributes:
        questions: How many questions.
        means: Each metric's mean.
        intervals: Each metric's interval.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: int
    means: dict[str, float]
    intervals: dict[str, tuple[float, float]]


class SetResult(BaseModel):
    """One question set's run: overall, by period and by party.

    Attributes:
        overall: Every question of the set.
        by_period: The questions of each period.
        by_party: The questions of each party, for sets written from a passage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: Summary
    by_period: dict[str, Summary]
    by_party: dict[str, Summary]


def speech_years(table: PassageTable) -> dict[str, int]:
    """The year of every speech, from its passages."""
    years: dict[str, int] = {}
    for speech, year in zip(table.anf_ids, table.years.tolist(), strict=True):
        years.setdefault(speech, int(year))
    return years


def run_set(
    index: LexicalIndex,
    table: PassageTable,
    questions: Sequence[EvalQuestion],
    params: Bm25Params,
    years: Mapping[str, int],
) -> list[QuestionResult]:
    """Run every question of a set and score it.

    Args:
        index: The index of the chosen analyzer.
        table: The passages.
        questions: The questions; each needs at least one gold speech.
        params: ``k1`` and ``b``.
        years: The year of every speech.

    Returns:
        One result per question, in the order given.
    """
    masks = FilterMasks(table)
    results = []
    for question in questions:
        ranking = rank(index, table, question, params, masks)
        party = None
        if question.passage_id is not None:
            party = str(table.parties[table.position(question.passage_id)])
        results.append(
            QuestionResult(
                question_id=question.question_id,
                period=table.period_of_year(years[question.gold[0]]),
                party=party,
                ranking=tuple(ranking),
                metrics=question_metrics(ranking, question.gold),
            )
        )
    return results


def summarize(results: Sequence[QuestionResult], seed: int, resamples: int) -> Summary:
    """The mean of each metric over some questions, with its bootstrap interval."""
    means = mean_metrics(result.metrics for result in results)
    intervals: dict[str, tuple[float, float]] = {}
    for name in METRIC_NAMES:
        low, high = bootstrap_interval(
            [result.metrics[name] for result in results], seed, resamples
        )
        intervals[name] = (round(low, 6), round(high, 6))
    return Summary(
        questions=len(results),
        means={name: round(value, 6) for name, value in means.items()},
        intervals=intervals,
    )


def summarize_set(results: Sequence[QuestionResult], seed: int, resamples: int) -> SetResult:
    """Summarize a set overall, by period and by party."""
    periods = sorted({result.period for result in results if result.period is not None})
    parties = sorted({result.party for result in results if result.party is not None})
    return SetResult(
        overall=summarize(results, seed, resamples),
        by_period={
            period: summarize([r for r in results if r.period == period], seed, resamples)
            for period in periods
        },
        by_party={
            party: summarize([r for r in results if r.party == party], seed, resamples)
            for party in parties
        },
    )
