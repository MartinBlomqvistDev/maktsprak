"""Tests for search/eval/sampling.py."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from search.corpus.table import PassageTable
from search.eval.sampling import (
    band_edges,
    build_frame,
    candidates,
    draw_slots,
    interleave,
    split_pattern,
    systematic_positions,
)


def table_of(rows: int, seed: int = 1) -> PassageTable:
    """A passage table of *rows* passages over six years, three parties and varied lengths."""
    rng = random.Random(seed)
    years = [2004 + rng.randrange(6) for _ in range(rows)]
    parties = [rng.choice(["S", "M", "V"]) for _ in range(rows)]
    return PassageTable(
        passage_ids=[f"H{number}-1:0-0" for number in range(rows)],
        anf_ids=[f"H{number}-1" for number in range(rows)],
        parties=np.array(parties, dtype="<U2"),
        years=np.array(years, dtype=np.int16),
        words=np.array([rng.randrange(20, 300) for _ in range(rows)], dtype=np.int32),
    )


class TestFrame:
    def test_band_edges_cut_the_lengths_in_thirds(self):
        assert band_edges(np.arange(1, 10)) == (4, 7)

    def test_the_frame_is_sorted_by_year_party_and_band(self):
        table = table_of(300)
        frame = build_frame(table, seed=3)
        order = frame.order
        keys = list(zip(table.years[order].tolist(), table.parties[order].tolist(), strict=True))
        assert keys == sorted(keys)
        assert sorted(order.tolist()) == list(range(300))

    def test_each_position_knows_its_stratum(self):
        table = table_of(300)
        frame = build_frame(table, seed=3)
        edges = np.array(frame.band_edges)
        for position in range(len(frame)):
            start, end = frame.stratum_start[position], frame.stratum_end[position]
            assert start <= position < end
            members = frame.order[start:end]
            assert len(set(table.years[members].tolist())) == 1
            assert len(set(table.parties[members].tolist())) == 1
            bands = np.searchsorted(edges, table.words[members], side="left")
            assert len(set(bands.tolist())) == 1

    def test_the_seed_decides_the_order_within_strata(self):
        table = table_of(300)
        assert build_frame(table, 3).order.tolist() == build_frame(table, 3).order.tolist()
        assert build_frame(table, 3).order.tolist() != build_frame(table, 4).order.tolist()


class TestDraw:
    def test_systematic_positions_are_spread_and_distinct(self):
        positions = systematic_positions(100, 10, random.Random(1))
        assert len(positions) == len(set(positions)) == 10
        assert all(0 <= position < 100 for position in positions)
        assert all(9 <= b - a <= 11 for a, b in zip(positions, positions[1:], strict=False))

    def test_more_draws_than_positions_are_refused(self):
        with pytest.raises(ValueError):
            systematic_positions(5, 6, random.Random(1))

    def test_interleave_keeps_each_group_near_its_share(self):
        assert interleave({"a": 2, "b": 1}) == ["a", "b", "a"]
        sequence = interleave({"natural": 300, "filtered": 100, "stance": 50})
        assert Counter(sequence) == {"natural": 300, "filtered": 100, "stance": 50}
        assert Counter(sequence[:9]) == {"natural": 6, "filtered": 2, "stance": 1}

    def test_two_of_every_five_go_to_development(self):
        assert split_pattern(5, 0) == ["dev", "test", "dev", "test", "test"]
        assert Counter(split_pattern(300, 3))["dev"] == 120
        assert split_pattern(5, 1) == ["test", "dev", "test", "test", "dev"]

    def test_slots_get_their_sets_splits_and_filters(self):
        frame = build_frame(table_of(900), seed=5)
        slots = draw_slots(frame, {"natural": 30, "filtered": 10, "stance": 5}, seed=7)
        assert Counter(slot.question_set for slot in slots) == {
            "natural": 30,
            "filtered": 10,
            "stance": 5,
        }
        natural = [slot for slot in slots if slot.question_set == "natural"]
        assert Counter(slot.split for slot in natural)["dev"] == 12
        filters = [slot.filter_kind for slot in slots if slot.question_set == "filtered"]
        assert Counter(filters) == {"party": 5, "period": 5}
        assert {slot.slot_id for slot in natural} >= {"natural-0001", "natural-0030"}
        assert len({slot.frame_position for slot in slots}) == 45

    def test_the_same_seed_draws_the_same_slots(self):
        frame = build_frame(table_of(900), seed=5)
        sizes = {"natural": 30, "stance": 5}
        assert draw_slots(frame, sizes, seed=7) == draw_slots(frame, sizes, seed=7)


class TestCandidates:
    def test_the_walk_starts_at_the_drawn_passage_and_stays_in_its_stratum(self):
        table = table_of(300)
        frame = build_frame(table, seed=3)
        position = 37
        walk = list(candidates(frame, position))
        start, end = frame.stratum_start[position], frame.stratum_end[position]
        assert walk[0] == frame.order[position]
        assert sorted(walk) == sorted(frame.order[start:end].tolist())
