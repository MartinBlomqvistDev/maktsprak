"""Tests for src/maktsprak_pipeline/nlp/distinctiveness.py (Fightin' Words)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.maktsprak_pipeline.nlp.distinctiveness import (
    distinctive_words,
    group_token_counts,
    tokenize,
    weighted_log_odds,
    wordcloud_frequencies,
)


class TestTokenize:
    def test_lowercases_and_keeps_swedish_letters(self):
        assert tokenize("Skolan ÄR Viktig", min_length=1, stopwords=frozenset()) == [
            "skolan",
            "är",
            "viktig",
        ]

    def test_drops_short_and_stopwords(self):
        assert tokenize("vi ska satsa på skolan", min_length=4, stopwords={"satsa"}) == [
            "skolan"
        ]

    def test_strips_digits_and_punctuation(self):
        assert tokenize("2024: klimatet, skatten!", stopwords=frozenset()) == [
            "klimatet",
            "skatten",
        ]

    def test_keeps_accented_latin_letters(self):
        # é must survive: otherwise "moské" truncates to "mosk" and every
        # inflected form collapses to the same wrong stem.
        assert tokenize("moské moskén idé", min_length=1, stopwords=frozenset()) == [
            "moské",
            "moskén",
            "idé",
        ]

    def test_non_string_returns_empty(self):
        assert tokenize(None) == []  # type: ignore[arg-type]


class TestGroupTokenCounts:
    def test_counts_per_group(self):
        # Use an explicit empty stop-word set so the test doesn't depend on the
        # curated list (which drops common political words like "skatt").
        df = pd.DataFrame(
            {
                "party": ["S", "S", "M"],
                "text": ["skola skola", "skola välfärd", "kärnkraft kärnkraft"],
            }
        )
        counts = group_token_counts(df, min_length=3, stopwords=frozenset())
        assert counts["S"]["skola"] == 3
        assert counts["M"]["kärnkraft"] == 2
        assert "skola" not in counts["M"]

    def test_party_self_names_filtered_by_default(self):
        # By default, party self-references are dropped so the clouds show
        # rhetoric, not letterhead.
        df = pd.DataFrame(
            {
                "party": ["SD", "V"],
                "text": ["sverigedemokraterna vill migranter", "vänsterpartiet vill gaza"],
            }
        )
        counts = group_token_counts(df)
        assert "sverigedemokraterna" not in counts["SD"]
        assert "vänsterpartiet" not in counts["V"]
        assert counts["SD"]["migranter"] == 1
        assert counts["V"]["gaza"] == 1


class TestWeightedLogOdds:
    def _counts(self):
        # Two parties: A leans hard on "skatt", B on "klimat"; "välfärd" shared.
        return {
            "A": Counter({"skatt": 100, "välfärd": 40, "klimat": 2}),
            "B": Counter({"klimat": 100, "välfärd": 40, "skatt": 2}),
        }

    def test_distinctive_word_scores_positive_for_owner(self):
        scores = weighted_log_odds(self._counts(), min_count=1)
        assert scores["A"]["skatt"] > 0
        assert scores["B"]["klimat"] > 0

    def test_distinctive_word_scores_negative_for_other(self):
        scores = weighted_log_odds(self._counts(), min_count=1)
        # "skatt" is under-represented in B → negative z there.
        assert scores["B"]["skatt"] < 0
        assert scores["A"]["klimat"] < 0

    def test_shared_word_near_zero(self):
        scores = weighted_log_odds(self._counts(), min_count=1)
        # "välfärd" is used equally → z close to zero, and much smaller in
        # magnitude than the distinctive words.
        assert abs(scores["A"]["välfärd"]) < abs(scores["A"]["skatt"])
        assert abs(scores["B"]["välfärd"]) < abs(scores["B"]["klimat"])

    def test_two_groups_are_mirror_images(self):
        scores = weighted_log_odds(self._counts(), min_count=1)
        # With exactly two groups, A-vs-rest is the negation of B-vs-rest.
        assert scores["A"]["skatt"] == -scores["B"]["skatt"]

    def test_min_count_filters_vocab(self):
        counts = {
            "A": Counter({"vanlig": 50, "annan": 50, "sällsynt": 1}),
            "B": Counter({"vanlig": 50, "annan": 50}),
        }
        scores = weighted_log_odds(counts, min_count=5)
        assert "sällsynt" not in scores["A"]
        assert "vanlig" in scores["A"]

    def test_empty_input_returns_empty_per_group(self):
        scores = weighted_log_odds({"A": Counter(), "B": Counter()}, min_count=5)
        assert scores == {"A": {}, "B": {}}


class TestDistinctiveWordsAndCloud:
    def _scores(self):
        counts = {
            "A": Counter({"skatt": 100, "företag": 60, "välfärd": 40}),
            "B": Counter({"klimat": 100, "rättvisa": 60, "välfärd": 40}),
        }
        return weighted_log_odds(counts, min_count=1)

    def test_top_words_are_the_distinctive_ones(self):
        top = distinctive_words(self._scores(), "A", top_n=2)
        words = [w for w, _ in top]
        assert "skatt" in words
        assert "företag" in words

    def test_only_positive_scores_returned(self):
        top = distinctive_words(self._scores(), "A", top_n=100)
        assert all(z > 0 for _, z in top)

    def test_wordcloud_frequencies_positive_floats(self):
        freqs = wordcloud_frequencies(self._scores(), "A", top_n=10)
        assert freqs
        assert all(isinstance(v, float) and v > 0 for v in freqs.values())
