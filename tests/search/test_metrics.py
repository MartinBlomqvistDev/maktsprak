"""Tests for search/eval/metrics.py and search/eval/stats.py."""

from __future__ import annotations

import pytest

from search.eval.metrics import (
    mean_metrics,
    question_metrics,
    recall_at,
    reciprocal_rank,
    speech_ranking,
)
from search.eval.stats import bootstrap_interval, paired_bootstrap

SPEECH_OF = ["a", "a", "b", "c", "b", "d"]


class TestMetrics:
    def test_a_speech_counts_once_at_its_best_rank(self):
        assert speech_ranking([0, 1, 2, 3, 4, 5], SPEECH_OF) == ["a", "b", "c", "d"]
        assert speech_ranking([4, 0], SPEECH_OF) == ["b", "a"]
        assert speech_ranking([0, 2, 3], SPEECH_OF, depth=2) == ["a", "b"]

    def test_recall_is_the_share_of_gold_found(self):
        ranking = ["a", "b", "c"]
        assert recall_at(ranking, ["c", "x"], 3) == 0.5
        assert recall_at(ranking, ["c"], 2) == 0.0
        with pytest.raises(ValueError):
            recall_at(ranking, [], 3)

    def test_the_reciprocal_rank_of_the_first_gold_within_k(self):
        assert reciprocal_rank(["a", "b", "c"], ["c", "b"], 10) == 0.5
        assert reciprocal_rank(["a", "b", "c"], ["c"], 2) == 0.0

    def test_every_metric_of_a_question_and_their_means(self):
        one = question_metrics(["g"], ["g"])
        assert one == {"recall@5": 1.0, "recall@10": 1.0, "recall@50": 1.0, "mrr@10": 1.0}
        none = question_metrics([], ["g"])
        assert mean_metrics([one, none])["recall@10"] == 0.5
        assert mean_metrics([]) == {}


class TestBootstrap:
    def test_the_interval_holds_the_mean_and_follows_the_seed(self):
        values = [0, 1] * 50
        low, high = bootstrap_interval(values, seed=1, resamples=2000)
        assert low < 0.5 < high
        assert (low, high) == bootstrap_interval(values, seed=1, resamples=2000)

    def test_constant_values_have_no_width(self):
        assert bootstrap_interval([0.3] * 10, seed=1, resamples=100) == pytest.approx((0.3, 0.3))

    def test_a_paired_difference_resamples_the_same_questions(self):
        first, second = [1, 1, 0, 1] * 25, [1, 0, 0, 1] * 25
        mean, low, high = paired_bootstrap(first, second, seed=2, resamples=2000)
        assert mean == pytest.approx(0.25)
        assert 0 < low < mean < high

    def test_empty_or_unequal_input_is_refused(self):
        with pytest.raises(ValueError):
            bootstrap_interval([], seed=1)
        with pytest.raises(ValueError):
            paired_bootstrap([1], [1, 0], seed=1)
