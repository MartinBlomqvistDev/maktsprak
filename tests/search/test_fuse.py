"""Tests for search/index/fuse.py."""

from __future__ import annotations

import pytest

from search.index.fuse import reciprocal_rank_fusion


class TestFusion:
    def test_scores_are_the_sum_of_reciprocal_ranks(self):
        hits = reciprocal_rank_fusion([[10, 20], [20, 30]], k=60)
        by_passage = {hit.passage: hit for hit in hits}
        assert by_passage[20].score == pytest.approx(1 / 62 + 1 / 61)
        assert by_passage[10].score == pytest.approx(1 / 61)
        assert by_passage[30].score == pytest.approx(1 / 62)
        assert by_passage[20].ranks == (2, 1)
        assert by_passage[30].ranks == (None, 2)

    def test_a_passage_in_both_rankings_rises(self):
        hits = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]], k=60)
        assert hits[0].passage == 3

    def test_a_tie_goes_to_the_better_rank_then_to_passage_order(self):
        hits = reciprocal_rank_fusion([[9, 4], [4, 9]], k=60)
        assert [hit.passage for hit in hits] == [4, 9]
        hits = reciprocal_rank_fusion([[7], [3]], k=60)
        assert [hit.passage for hit in hits] == [3, 7]

    def test_the_order_of_the_rankings_makes_no_difference(self):
        first = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=20)
        second = reciprocal_rank_fusion([[3, 1, 4], [1, 2, 3]], k=20)
        assert [hit.passage for hit in first] == [hit.passage for hit in second]

    def test_a_small_k_favours_a_top_rank_and_a_large_k_agreement(self):
        rankings = [[1, 2, 7], [8, 9, 7]]
        assert reciprocal_rank_fusion(rankings, k=0.5)[0].passage == 1
        assert reciprocal_rank_fusion(rankings, k=60)[0].passage == 7

    def test_the_limit_cuts_the_result(self):
        assert len(reciprocal_rank_fusion([[1, 2, 3], [4, 5]], k=60, limit=2)) == 2

    def test_an_empty_ranking_adds_nothing(self):
        hits = reciprocal_rank_fusion([[1, 2], []], k=60)
        assert [hit.passage for hit in hits] == [1, 2]

    @pytest.mark.parametrize("k", [0, -1])
    def test_a_constant_not_above_zero_raises(self, k):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[1]], k=k)

    def test_a_passage_twice_in_one_ranking_raises(self):
        with pytest.raises(ValueError, match="twice"):
            reciprocal_rank_fusion([[1, 1]], k=60)
