"""Tests for src/maktsprak_pipeline/nlp/cleaning.py."""

from __future__ import annotations

from src.maktsprak_pipeline.nlp.cleaning import (
    clean_text,
    combined_stopwords,
    political_stopwords,
    swedish_stopwords,
)


class TestStopwords:
    def test_combined_is_union(self):
        assert combined_stopwords == swedish_stopwords | political_stopwords

    def test_combined_is_frozenset(self):
        assert isinstance(combined_stopwords, frozenset)

    def test_political_stopwords_is_frozenset(self):
        assert isinstance(political_stopwords, frozenset)

    def test_no_concatenated_words(self):
        """Ensure the missing-comma bug is fixed: 'sverigestackt' must not appear."""
        bad_concat = "sverigestackt"
        assert bad_concat not in combined_stopwords, (
            f"'{bad_concat}' found in combined_stopwords — "
            "likely caused by a missing comma in the set literal."
        )

    def test_individual_words_present(self):
        """The words around the old bug site should each be present individually."""
        assert "sverige" in combined_stopwords or "sveriges" in combined_stopwords
        assert "tack" in combined_stopwords

    def test_known_political_terms_present(self):
        for word in ("talman", "riksdagen", "anförande", "replik"):
            assert word in political_stopwords, f"'{word}' missing from political_stopwords"

    def test_no_empty_strings(self):
        assert "" not in combined_stopwords


class TestCleanText:
    def test_hyphen_newline_joined(self):
        assert clean_text("demo-\ncrat") == "democrat"

    def test_multiple_spaces_collapsed(self):
        assert clean_text("hello   world") == "hello world"

    def test_newlines_collapsed_to_space(self):
        result = clean_text("line one\nline two")
        assert "\n" not in result
        assert "line one" in result

    def test_strip_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]
        assert clean_text(42) == ""  # type: ignore[arg-type]

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_normal_text_unchanged(self):
        text = "Sverige är ett demokratiskt land."
        assert clean_text(text) == text
