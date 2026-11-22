"""Tests for search/index/subset.py."""

from __future__ import annotations

import numpy as np
import pytest

from search.corpus.table import PassageTable
from search.index.subset import draw_subset, proportional_quotas, speech_passages


def table_of(speeches: list[tuple[str, int, str, int]]) -> PassageTable:
    """A passage table from (speech, year, party, passages) rows."""
    ids, anf, parties, years, words = [], [], [], [], []
    for speech, year, party, count in speeches:
        for number in range(count):
            ids.append(f"{speech}:{number}-{number}")
            anf.append(speech)
            parties.append(party)
            years.append(year)
            words.append(100)
    return PassageTable(
        passage_ids=ids,
        anf_ids=anf,
        parties=np.array(parties, dtype="<U2"),
        years=np.array(years, dtype=np.int16),
        words=np.array(words, dtype=np.int32),
    )


def corpus(seed: int = 0) -> PassageTable:
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2005, 2012, 2020):
        for party in ("S", "M", "SD"):
            for number in range(40):
                rows.append((f"H{year}{party}-{number}", year, party, int(rng.integers(1, 4))))
    return table_of(rows)


class TestQuotas:
    def test_quotas_add_up_and_follow_the_sizes(self):
        quotas = proportional_quotas({(2005, "S"): 30, (2005, "M"): 10}, 8)
        assert quotas == {(2005, "M"): 2, (2005, "S"): 6}

    def test_leftovers_go_to_the_largest_remainders(self):
        quotas = proportional_quotas({(1, "a"): 1, (1, "b"): 1, (1, "c"): 1}, 2)
        assert sum(quotas.values()) == 2
        assert quotas == {(1, "a"): 1, (1, "b"): 1, (1, "c"): 0}

    def test_nothing_to_fill_gives_nothing(self):
        assert proportional_quotas({(1, "a"): 5}, 0) == {(1, "a"): 0}


class TestDraw:
    def test_every_gold_speech_is_in_the_subset_whole(self):
        table = corpus()
        gold = ["H2005S-3", "H2020SD-7"]
        draw = draw_subset(table, gold, size=60, seed=1)
        held = {table.anf_ids[position] for position in draw.positions}
        assert set(gold) <= held
        passages = speech_passages(table)
        for speech in held:
            assert set(passages[speech]) <= set(draw.positions.tolist())

    def test_the_subset_reaches_its_size_and_goes_over_by_less_than_a_speech_a_group(self):
        table = corpus()
        draw = draw_subset(table, [], size=90, seed=1)
        assert 90 <= len(draw.positions) < 90 + 3 * len(draw.quotas)
        assert all(draw.taken[key] >= quota for key, quota in draw.quotas.items())

    def test_the_draw_is_spread_over_years_and_parties(self):
        table = corpus()
        draw = draw_subset(table, [], size=90, seed=2)
        assert set(draw.quotas) == {(y, p) for y in (2005, 2012, 2020) for p in ("S", "M", "SD")}
        assert all(count > 0 for count in draw.taken.values())

    def test_the_same_seed_draws_the_same_subset(self):
        table = corpus()
        first = draw_subset(table, ["H2012M-1"], size=70, seed=5)
        second = draw_subset(table, ["H2012M-1"], size=70, seed=5)
        other = draw_subset(table, ["H2012M-1"], size=70, seed=6)
        assert first.positions.tolist() == second.positions.tolist()
        assert first.positions.tolist() != other.positions.tolist()

    def test_positions_are_ascending_and_distinct(self):
        draw = draw_subset(corpus(), ["H2005S-1", "H2005S-2"], size=50, seed=3)
        assert (np.diff(draw.positions) > 0).all()

    def test_gold_alone_can_fill_the_subset(self):
        table = corpus()
        gold = sorted(set(table.anf_ids))[:30]
        draw = draw_subset(table, gold, size=5, seed=1)
        assert draw.drawn_passages == 0 and draw.gold_speeches == 30

    def test_a_gold_speech_without_passages_raises(self):
        with pytest.raises(KeyError, match="no passage"):
            draw_subset(corpus(), ["absent"], size=10, seed=1)
