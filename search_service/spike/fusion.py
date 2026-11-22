"""Reciprocal rank fusion of two rankings, as option B merges its two legs.

An item at rank ``r`` of a ranking, counted from 1, gets ``1 / (k + r)``, and an
item's score is the sum over the rankings it is in. LanceDB's ``RRFReranker``
computes the same sum, so both options fuse alike.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[int]], k: int, limit: int) -> list[tuple[int, float]]:
    """Merge rankings by reciprocal rank fusion.

    Args:
        rankings: Each a list of items, best first. An item appears at most once in one.
        k: The constant added to every rank.
        limit: The most items to return.

    Returns:
        ``(item, score)`` pairs, best first. Equal scores are ordered by item.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered[:limit]
