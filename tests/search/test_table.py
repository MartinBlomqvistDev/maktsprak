"""Tests for search/corpus/table.py."""

from __future__ import annotations

import numpy as np
import pytest

from search.corpus.table import PassageTable, load_passage_table
from tests.search.corpus_kit import write_corpus


@pytest.fixture
def table(tmp_path) -> PassageTable:
    """The synthetic corpus's passages."""
    _, passages_path = write_corpus(tmp_path)
    return load_passage_table(passages_path)


class TestTable:
    def test_one_row_per_passage_in_file_order(self, table):
        assert len(table) == 8
        assert table.passage_ids[0] == "H209100-1:0-1"
        assert table.anf_ids[0] == "H209100-1"
        assert table.position("H209100-1:0-1") == 0

    def test_parties_years_and_words(self, table):
        assert table.parties.tolist()[:3] == ["M", "MP", "S"]
        assert table.years.tolist()[:3] == [2015, 2015, 2004]
        assert int(table.words[0]) == 11

    def test_the_last_period_is_named_after_the_latest_year(self, table):
        assert table.period_labels() == ["2003-2009", "2010-2017", "2018-2023"]
        assert table.period_of_year(2015) == "2010-2017"

    def test_a_year_before_the_first_period_is_refused(self, table):
        with pytest.raises(ValueError):
            table.period_of_year(2001)

    def test_masks_mark_the_periods_and_parties(self, table):
        assert table.period_mask("2003-2009").tolist() == [
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        assert int(table.period_mask("2018-2023").sum()) == 2
        assert int(table.party_mask("M").sum()) == 4

    def test_an_unknown_period_is_refused(self, table):
        with pytest.raises(ValueError, match="not a period"):
            table.period_mask("1990-1999")

    def test_a_passage_id_twice_is_refused(self):
        with pytest.raises(ValueError, match="twice"):
            PassageTable(
                passage_ids=["a:0-0", "a:0-0"],
                anf_ids=["a", "a"],
                parties=np.array(["S", "S"]),
                years=np.array([2015, 2015]),
                words=np.array([1, 1]),
            )

    def test_columns_of_different_length_are_refused(self):
        with pytest.raises(ValueError, match="length"):
            PassageTable(
                passage_ids=["a:0-0"],
                anf_ids=["a", "b"],
                parties=np.array(["S"]),
                years=np.array([2015]),
                words=np.array([1]),
            )
