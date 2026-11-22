"""Reciprocal rank fusion of the lexical and the vector ranking.

Hybrid retrieval means the BM25 ranking and the vector ranking merged by
reciprocal rank fusion, with any filter applied to both before they are merged
(``docs/search/PREREGISTRATION.md``). A passage scores ``1 / (k + rank)`` for
each ranking it appears in, with rank counted from 1, and nothing for a ranking
it is absent from. The score depends only on the order of each ranking, so the
two rankings need no common scale.

Two passages with the same score are ordered by the better of their ranks, and
then by their position in the passages file, so the result never depends on
the order the rankings are given in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedHit:
    """One passage of a fused ranking.

    Attributes:
        passage: The passage's position in the passages file.
        score: Its fused score.
        ranks: Its rank in each ranking, from 1, or ``None`` where it is absent.
    """

    passage: int
    score: float
    ranks: tuple[int | None, ...]

    @property
    def best_rank(self) -> int:
        """The best of the passage's ranks."""
        return min(rank for rank in self.ranks if rank is not None)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: float, limit: int | None = None
) -> list[FusedHit]:
    """Merge rankings of passages by reciprocal rank fusion.

    Args:
        rankings: Each ranking's passages, best first, as positions in the
            passages file.
        k: The RRF constant; a larger ``k`` flattens the advantage of the top ranks.
        limit: The most passages to return, or ``None`` for every passage ranked.

    Returns:
        The fused ranking, best first.

    Raises:
        ValueError: If ``k`` is not above 0, or a passage occurs twice in one ranking.
    """
    if k <= 0:
        raise ValueError(f"The RRF constant must be above 0, not {k}.")
    ranks: dict[int, list[int | None]] = {}
    for number, ranking in enumerate(rankings):
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"Ranking {number} holds a passage twice.")
        for rank, passage in enumerate(ranking, start=1):
            ranks.setdefault(passage, [None] * len(rankings))[number] = rank
    hits = [
        FusedHit(
            passage=passage,
            score=sum(1.0 / (k + rank) for rank in held if rank is not None),
            ranks=tuple(held),
        )
        for passage, held in ranks.items()
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.best_rank, hit.passage))
    return hits if limit is None else hits[:limit]
