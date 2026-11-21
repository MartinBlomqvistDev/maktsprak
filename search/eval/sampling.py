"""Drawing the passages that generated questions are written from, and splitting them.

The passages are drawn from a frame: every passage, sorted by the year of its
speech, its party, its length band and a random key. A systematic sample from
that order, one passage at a fixed interval from a random start, spreads the
draw over the strata in proportion to their size. The draw is decided by the
seed alone.

Each drawn passage fills a *slot*. A slot belongs to one question set and one
split, both fixed at the draw, before any question is written. When the model
finds no question in a slot's passage, the slot moves on to the next passage of
the same stratum, in the frame's random order, so a replacement comes from the
same year, party and length band as the passage it replaces.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.corpus.table import PassageTable
from search.eval.questions import Split

#: How many length bands the frame is stratified on: the terciles of passage length.
LENGTH_BANDS = 3

#: Of every five slots of a set, in frame order, the ones at these places go to the
#: development split: two of five, which is the pre-registered 40%.
DEV_PLACES = (0, 2)

SlotSet = Literal["natural", "filtered", "stance"]
FilterKind = Literal["party", "period"]


@dataclass(frozen=True)
class Frame:
    """Every passage, in the order the draw walks them.

    Attributes:
        order: Passage positions in index order, sorted by year, party, length
            band and random key.
        stratum_start: For each frame position, where its stratum starts in ``order``.
        stratum_end: For each frame position, where its stratum ends, exclusive.
        band_edges: The largest passage length, in words, of each band but the last.
    """

    order: np.ndarray
    stratum_start: np.ndarray
    stratum_end: np.ndarray
    band_edges: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.order)


def band_edges(words: np.ndarray, bands: int = LENGTH_BANDS) -> tuple[int, ...]:
    """Cut passage lengths into bands of about equal size.

    Args:
        words: The length of every passage, in words.
        bands: How many bands.

    Returns:
        The upper edge of each band but the last: band ``i`` holds the passages
        longer than edge ``i - 1`` and at most edge ``i`` words long.
    """
    ordered = np.sort(words)
    return tuple(int(ordered[(len(ordered) * cut) // bands]) for cut in range(1, bands))


def build_frame(table: PassageTable, seed: int) -> Frame:
    """Sort every passage by year, party, length band and a random key.

    Args:
        table: The passages.
        seed: Decides the random keys.

    Returns:
        The frame.
    """
    edges = band_edges(table.words)
    bands = np.searchsorted(np.array(edges), table.words, side="left")
    keys = np.random.default_rng(seed).random(len(table))
    order = np.lexsort((keys, bands, table.parties, table.years))
    strata = np.stack([table.years[order], bands[order]], axis=1)
    parties = table.parties[order]
    changes = np.ones(len(order), dtype=bool)
    changes[1:] = (
        (strata[1:, 0] != strata[:-1, 0])
        | (strata[1:, 1] != strata[:-1, 1])
        | (parties[1:] != parties[:-1])
    )
    starts = np.flatnonzero(changes)
    ends = np.append(starts[1:], len(order))
    stratum_of = np.cumsum(changes) - 1
    return Frame(
        order=order,
        stratum_start=starts[stratum_of],
        stratum_end=ends[stratum_of],
        band_edges=edges,
    )


def systematic_positions(size: int, draws: int, rng: random.Random) -> list[int]:
    """Pick *draws* positions out of *size* at a fixed interval from a random start.

    Args:
        size: How many positions there are.
        draws: How many to pick; at most *size*.
        rng: Decides the start.

    Returns:
        The positions, ascending and distinct.

    Raises:
        ValueError: If more positions are asked for than there are.
    """
    if not 0 < draws <= size:
        raise ValueError(f"Cannot draw {draws} of {size} positions.")
    interval = size / draws
    start = rng.random() * interval
    return [int(start + number * interval) for number in range(draws)]


def interleave(sizes: Mapping[str, int]) -> list[str]:
    """Spread several groups over one sequence in proportion to their sizes.

    At each step the group furthest behind its share takes the next place; a
    tie goes to the group named first. ``{"a": 2, "b": 1}`` gives
    ``["a", "b", "a"]``.

    Args:
        sizes: How many places each group gets.

    Returns:
        The group of each place.
    """
    total = sum(sizes.values())
    taken = dict.fromkeys(sizes, 0)
    sequence = []
    for place in range(total):
        behind = {name: (place + 1) * size / total - taken[name] for name, size in sizes.items()}
        name = max(sizes, key=lambda group: (behind[group], -list(sizes).index(group)))
        taken[name] += 1
        sequence.append(name)
    return sequence


def split_pattern(count: int, offset: int) -> list[Split]:
    """Assign *count* items, in order, so that two of every five go to the development split.

    Args:
        count: How many items.
        offset: Shifts the pattern, from 0 to 4.

    Returns:
        The split of each item.
    """
    return ["dev" if (place + offset) % 5 in DEV_PLACES else "test" for place in range(count)]


class Slot(BaseModel):
    """A place for one generated question, fixed at the draw.

    Attributes:
        slot_id: The key; the question written in the slot gets the same number.
        question_set: Which set the slot's question belongs to.
        split: Which split it belongs to.
        frame_position: Where the drawn passage sits in the frame.
        filter_kind: For a filtered question, whether it is searched under its
            party or its period.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot_id: str
    question_set: SlotSet
    split: Split
    frame_position: int
    filter_kind: FilterKind | None = None


def draw_slots(frame: Frame, sizes: Mapping[SlotSet, int], seed: int) -> list[Slot]:
    """Draw one passage per slot, and fix each slot's set, split and filter.

    The sets are interleaved over one systematic sample, so each set is spread
    over the whole frame. Within a set, two of every five slots in frame order
    go to the development split, and a filtered question alternates between a
    party and a period filter.

    Args:
        frame: The frame.
        sizes: How many slots each set gets, in the order the sets are named.
        seed: Decides the start of the sample and the offset of the split.

    Returns:
        The slots, in frame order.
    """
    rng = random.Random(seed)
    positions = systematic_positions(len(frame), sum(sizes.values()), rng)
    sets = interleave(sizes)
    offset = rng.randrange(5)
    slots: list[Slot] = []
    for name, size in sizes.items():
        mine = [position for position, group in zip(positions, sets, strict=True) if group == name]
        width = len(str(size))
        for number, (position, split) in enumerate(
            zip(mine, split_pattern(len(mine), offset), strict=True)
        ):
            filter_kind: FilterKind | None = None
            if name == "filtered":
                filter_kind = "party" if number % 2 == 0 else "period"
            slots.append(
                Slot(
                    slot_id=f"{name}-{number + 1:0{max(width, 4)}d}",
                    question_set=name,
                    split=split,
                    frame_position=position,
                    filter_kind=filter_kind,
                )
            )
    return sorted(slots, key=lambda slot: slot.frame_position)


def candidates(frame: Frame, frame_position: int) -> Iterator[int]:
    """Walk a slot's stratum from its drawn passage, wrapping around at the end.

    Args:
        frame: The frame.
        frame_position: Where the slot's passage sits in the frame.

    Yields:
        Passage positions in index order: the drawn passage first, then the
        rest of its stratum in the frame's order.
    """
    start = int(frame.stratum_start[frame_position])
    end = int(frame.stratum_end[frame_position])
    size = end - start
    for step in range(size):
        yield int(frame.order[start + (frame_position - start + step) % size])


def stratum_counts(table: PassageTable, positions: Sequence[int]) -> dict[str, int]:
    """Count passages per party, for the draw's report."""
    counts: dict[str, int] = {}
    for position in positions:
        party = str(table.parties[position])
        counts[party] = counts.get(party, 0) + 1
    return dict(sorted(counts.items()))
