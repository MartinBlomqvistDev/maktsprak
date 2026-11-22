"""The subset of step A: every gold speech of the development split, filled up by a stratified draw.

The embedding candidates are compared on a subset of the corpus, as the
addendum of 2026-11-21 in ``docs/search/PREREGISTRATION.md`` fixes. The subset
holds every speech that answers a question of the development split. The rest
of it is drawn from the other speeches, grouped by year and party: each group
gets a quota of the passages still to fill, in proportion to its share of the
corpus's passages, and its speeches are taken in an order fixed by the seed
until the quota is reached. Whole speeches are taken, so the speech that reaches
a group's quota may take it a few passages over, and the subset ends a little
above its size.

Quotas are whole numbers that add up to the passages still to fill: each group
gets the whole part of its share, and the passages left over go one each to the
groups with the largest remainders, the earlier group first on a tie.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from search.corpus.table import PassageTable

Stratum = tuple[int, str]


@dataclass(frozen=True)
class SubsetDraw:
    """The passages of the subset, and how they were drawn.

    Attributes:
        positions: The passages, as positions in the passages file, ascending.
        gold_speeches: How many gold speeches the subset holds.
        gold_passages: How many passages they have.
        drawn_speeches: How many speeches the draw added.
        drawn_passages: How many passages those have.
        quotas: The passages each group of year and party was to get.
        taken: The passages each group got.
    """

    positions: np.ndarray
    gold_speeches: int
    gold_passages: int
    drawn_speeches: int
    drawn_passages: int
    quotas: dict[Stratum, int]
    taken: dict[Stratum, int]


def speech_passages(table: PassageTable) -> dict[str, list[int]]:
    """Each speech's passages, as positions in the passages file, in order."""
    passages: dict[str, list[int]] = {}
    for position, speech in enumerate(table.anf_ids):
        passages.setdefault(speech, []).append(position)
    return passages


def proportional_quotas(sizes: dict[Stratum, int], total: int) -> dict[Stratum, int]:
    """Split *total* among groups in proportion to their sizes, by largest remainder.

    Args:
        sizes: Each group's size, by key.
        total: What is to be split.

    Returns:
        Each group's whole share; the shares add up to *total*.
    """
    whole = sum(sizes.values())
    if whole == 0 or total <= 0:
        return {key: 0 for key in sizes}
    keys = sorted(sizes)
    exact = {key: total * sizes[key] / whole for key in keys}
    quotas = {key: int(exact[key]) for key in keys}
    left = total - sum(quotas.values())
    by_remainder = sorted(keys, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in by_remainder[:left]:
        quotas[key] += 1
    return quotas


def draw_subset(table: PassageTable, gold: Iterable[str], size: int, seed: int) -> SubsetDraw:
    """Draw the subset: the gold speeches, filled up by a stratified draw of whole speeches.

    Args:
        table: The passages.
        gold: The speeches that must be in the subset.
        size: The passages the subset is filled up to.
        seed: Decides the order speeches are drawn in.

    Returns:
        The subset.

    Raises:
        KeyError: If a gold speech has no passage in the table.
    """
    passages = speech_passages(table)
    gold_set = set(gold)
    missing = sorted(gold_set - passages.keys())
    if missing:
        raise KeyError(f"{len(missing)} gold speeches have no passage, the first {missing[0]}.")
    chosen = [position for speech in sorted(gold_set) for position in passages[speech]]
    gold_passages = len(chosen)

    strata: dict[Stratum, list[str]] = {}
    for speech, held in passages.items():
        if speech not in gold_set:
            first = held[0]
            key = (int(table.years[first]), str(table.parties[first]))
            strata.setdefault(key, []).append(speech)
    sizes = {key: sum(len(passages[s]) for s in speeches) for key, speeches in strata.items()}
    quotas = proportional_quotas(sizes, size - gold_passages)

    rng = np.random.default_rng(seed)
    taken: dict[Stratum, int] = {}
    drawn_speeches = 0
    for key in sorted(strata):
        speeches = sorted(strata[key])
        order = rng.permutation(len(speeches))
        count = 0
        for index in order:
            if count >= quotas[key]:
                break
            held = passages[speeches[int(index)]]
            chosen.extend(held)
            count += len(held)
            drawn_speeches += 1
        taken[key] = count
    positions = np.array(sorted(chosen), dtype=np.int64)
    return SubsetDraw(
        positions=positions,
        gold_speeches=len(gold_set),
        gold_passages=gold_passages,
        drawn_speeches=drawn_speeches,
        drawn_passages=len(positions) - gold_passages,
        quotas=quotas,
        taken=taken,
    )
