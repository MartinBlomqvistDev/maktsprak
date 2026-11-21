"""Tests for search/index/analyzer.py."""

from __future__ import annotations

from search.index.analyzer import (
    ANALYZERS,
    STEMMING,
    Analyzer,
    stem,
    stem_all,
    stopwords,
    tokenize,
)


class TestTokenize:
    def test_text_is_lower_cased_and_split_on_everything_but_letters_and_digits(self):
        assert tokenize("EU-medlemskapet, år 2003!") == ["eu", "medlemskapet", "år", "2003"]

    def test_the_underscore_separates_tokens(self):
        assert tokenize("a_b") == ["a", "b"]

    def test_swedish_letters_are_kept(self):
        assert tokenize("Åtgärder för äldre i Örebro") == [
            "åtgärder",
            "för",
            "äldre",
            "i",
            "örebro",
        ]

    def test_empty_text_has_no_tokens(self):
        assert tokenize(" -- ") == []


class TestStopwords:
    def test_the_snowball_list_is_read_whole(self):
        words = stopwords()
        assert len(words) == 114
        assert {"och", "att", "vilkas", "sitt"} <= words

    def test_comment_lines_are_not_words(self):
        assert not any(word.startswith("#") or " " in word for word in stopwords())


class TestStem:
    def test_a_compound_keeps_its_parts_and_loses_its_ending(self):
        assert stem("kärnkraftsavvecklingen") == "kärnkraftsavveckling"
        assert stem("kärnkraften") == "kärnkraft"

    def test_many_tokens_stem_as_one_at_a_time(self):
        tokens = ["regeringen", "skolorna", "lärarna"]
        assert stem_all(tokens) == [stem(token) for token in tokens]

    def test_digits_pass_unchanged(self):
        assert stem("2003") == "2003"


class TestAnalyzer:
    def test_stop_words_are_dropped_before_stemming(self):
        analyzer = Analyzer(stem=True, stopwords=True)
        assert analyzer.terms("Och regeringen sitter kvar") == ["regering", "sitt", "kvar"]

    def test_without_stop_words_every_token_stays(self):
        analyzer = Analyzer(stem=False, stopwords=False)
        assert analyzer.terms("Och det är så") == ["och", "det", "är", "så"]

    def test_a_dropped_stop_word_has_no_term(self):
        assert Analyzer(stem=False, stopwords=True).term_of("och") is None

    def test_the_four_analyzers_have_four_names(self):
        names = [analyzer.name for analyzer in ANALYZERS]
        assert names == ["stem-stop", "stem-nostop", "nostem-stop", "nostem-nostop"]

    def test_the_rare_word_check_stems_and_keeps_stop_words(self):
        assert Analyzer(stem=True, stopwords=False) == STEMMING
