"""Retrieval metrics at speech level, as the pre-registration defines them.

A retrieved passage counts as its speech, and a speech counts once, at its
best rank. Where a question has several gold speeches, recall is the share of
them found.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

#: The metrics every retrieval run reports, by name.
METRIC_NAMES: tuple[str, ...] = ("recall@5", "recall@10", "recall@50", "mrr@10")

#: The deepest rank any metric looks at.
DEPTH = 50


def speech_ranking(
    passages: Iterable[int], speech_of: Sequence[str], depth: int = DEPTH
) -> list[str]:
    """Turn a ranking of passages into a ranking of speeches.

    Args:
        passages: Passage positions, best first.
        speech_of: The speech of each passage position.
        depth: How many speeches to keep.

    Returns:
        The speeches, best first, each at the rank of its best passage.
    """
    ranking: list[str] = []
    seen: set[str] = set()
    for position in passages:
        speech = speech_of[position]
        if speech not in seen:
            seen.add(speech)
            ranking.append(speech)
            if len(ranking) == depth:
                break
    return ranking


def recall_at(ranking: Sequence[str], gold: Sequence[str], k: int) -> float:
    """The share of the gold speeches among the first *k* of a ranking."""
    if not gold:
        raise ValueError("Recall needs at least one gold speech.")
    found = set(ranking[:k])
    return sum(speech in found for speech in gold) / len(gold)


def reciprocal_rank(ranking: Sequence[str], gold: Sequence[str], k: int) -> float:
    """One over the rank of the first gold speech within the first *k*, or 0."""
    wanted = set(gold)
    for rank, speech in enumerate(ranking[:k], start=1):
        if speech in wanted:
            return 1 / rank
    return 0.0


def question_metrics(ranking: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    """Every metric of :data:`METRIC_NAMES` for one question."""
    return {
        "recall@5": recall_at(ranking, gold, 5),
        "recall@10": recall_at(ranking, gold, 10),
        "recall@50": recall_at(ranking, gold, 50),
        "mrr@10": reciprocal_rank(ranking, gold, 10),
    }


def mean_metrics(per_question: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """The mean of each metric over questions."""
    rows = list(per_question)
    if not rows:
        return {}
    return {name: sum(row[name] for row in rows) / len(rows) for name in METRIC_NAMES}
