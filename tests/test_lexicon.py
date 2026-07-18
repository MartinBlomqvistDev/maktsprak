"""Tests for src/maktsprak_pipeline/nlp/lexicon.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.maktsprak_pipeline.nlp.lexicon import apply_ton_lexicon


@pytest.fixture
def minimal_lexicon(tmp_path: Path) -> Path:
    """Write a tiny lexicon CSV and return its path."""
    content = "ord,kategori,vikt\nkatastrof,Negativt,1.0\nbra,Positivt,1.0\nutmärkt,Positivt,2.0\n"
    lexicon_file = tmp_path / "lexikon.csv"
    lexicon_file.write_text(content, encoding="utf-8")
    return lexicon_file


class TestApplyTonLexicon:
    def test_returns_dataframe(self, sample_speeches_df, minimal_lexicon):
        result = apply_ton_lexicon(sample_speeches_df, lexicon_path=minimal_lexicon)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, sample_speeches_df, minimal_lexicon):
        original_cols = list(sample_speeches_df.columns)
        apply_ton_lexicon(sample_speeches_df, lexicon_path=minimal_lexicon)
        assert list(sample_speeches_df.columns) == original_cols

    def test_adds_category_columns(self, sample_speeches_df, minimal_lexicon):
        result = apply_ton_lexicon(sample_speeches_df, lexicon_path=minimal_lexicon)
        assert "Negativt" in result.columns
        assert "Positivt" in result.columns

    def test_scores_are_non_negative(self, sample_speeches_df, minimal_lexicon):
        result = apply_ton_lexicon(sample_speeches_df, lexicon_path=minimal_lexicon)
        assert (result["Negativt"] >= 0).all()
        assert (result["Positivt"] >= 0).all()

    def test_missing_lexicon_path_returns_copy(self, sample_speeches_df):
        result = apply_ton_lexicon(sample_speeches_df, lexicon_path=None)
        assert list(result.columns) == list(sample_speeches_df.columns)

    def test_nonexistent_file_returns_copy(self, sample_speeches_df, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        result = apply_ton_lexicon(sample_speeches_df, lexicon_path=missing)
        assert list(result.columns) == list(sample_speeches_df.columns)

    def test_higher_weight_gives_higher_score(self, minimal_lexicon):
        # "utmärkt" has weight 2.0, "bra" has weight 1.0
        df = pd.DataFrame({"text": ["utmärkt utmärkt", "bra bra"]})
        result = apply_ton_lexicon(df, lexicon_path=minimal_lexicon)
        assert result.loc[0, "Positivt"] > result.loc[1, "Positivt"]

    def test_empty_text_gives_zero_score(self, minimal_lexicon):
        df = pd.DataFrame({"text": [""]})
        result = apply_ton_lexicon(df, lexicon_path=minimal_lexicon)
        assert result["Positivt"].iloc[0] == 0.0

    def test_matching_is_case_insensitive(self, tmp_path):
        # Regression: all-caps lexicon entries used to never match because
        # tokens are lower-cased. Matching must be case-folded on both sides.
        lex = tmp_path / "lex.csv"
        lex.write_text("ord,kategori,vikt\nFAKTA,Sak,1.0\n", encoding="utf-8")
        df = pd.DataFrame({"text": ["Det här handlar om fakta och siffror."]})
        result = apply_ton_lexicon(df, lexicon_path=lex)
        assert result["Sak"].iloc[0] > 0.0

    def test_multiword_phrase_matches(self, tmp_path):
        # Regression: multi-word entries used to never match because the text
        # was split into single tokens. Phrases must match as a unit.
        lex = tmp_path / "lex.csv"
        lex.write_text("ord,kategori,vikt\nvi mot dem,Pop,2.0\n", encoding="utf-8")
        df = pd.DataFrame(
            {"text": ["Det är alltid vi mot dem i den här debatten.", "Vi samarbetar."]}
        )
        result = apply_ton_lexicon(df, lexicon_path=lex)
        assert result["Pop"].iloc[0] > 0.0
        assert result["Pop"].iloc[1] == 0.0

    def test_single_word_does_not_match_as_substring(self, tmp_path):
        # "hot" must not fire inside "hotell", single words match whole tokens.
        lex = tmp_path / "lex.csv"
        lex.write_text("ord,kategori,vikt\nhot,Agg,1.0\n", encoding="utf-8")
        df = pd.DataFrame({"text": ["Vi bokade ett hotell i staden."]})
        result = apply_ton_lexicon(df, lexicon_path=lex)
        assert result["Agg"].iloc[0] == 0.0
