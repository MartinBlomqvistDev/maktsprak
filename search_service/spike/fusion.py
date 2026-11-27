"""Reciprocal rank fusion of two rankings, as option B merges its two legs.

An item at rank ``r`` of a ranking, counted from 1, gets ``1 / (k + r)``, and an
item's score is the sum over the rankings it is in. LanceDB's ``RRFReranker``
computes the same sum, so both options fuse alike.

Equal scores are common at a small ``k``: two ranks of 12 sum to exactly the
score of rank 1 alone at ``k`` = 10. They are ordered as
``search.index.fuse.reciprocal_rank_fusion`` orders them, by the better of the
item's ranks and then by item, so the service ranks as the hybrid the test
split measured.
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
        ``(item, score)`` pairs, best first. Equal scores are ordered by the
        item's best rank, then by item.
    """
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], best[item], item))
    return [(item, scores[item]) for item in ordered[:limit]]
