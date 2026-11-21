"""Tests for search/index/lexical.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from search.index.analyzer import Analyzer
from search.index.lexical import (
    DEFAULT_PARAMS,
    Bm25Params,
    IndexManifest,
    LexicalIndex,
    build_index,
    file_sha256,
    passage_ids_sha256,
    passage_texts,
    tokenize_corpus,
)
from tests.search.corpus_kit import SPEECHES, speech, write_corpus

PLAIN = Analyzer(stem=False, stopwords=False)

TEXTS = [
    ("a:0-0", "röd röd blå"),
    ("b:0-0", "blå grön"),
    ("c:0-0", "gul"),
    ("d:0-0", "röd gul gul gul gul"),
]


@pytest.fixture
def index() -> LexicalIndex:
    """An index of four short texts, without stemming or stop words."""
    return build_index(tokenize_corpus(TEXTS), PLAIN)


def bm25(tf: int, df: int, length: int, mean_length: float, passages: int, k1=1.2, b=0.75):
    """BM25 of one term in one passage, written out by hand."""
    idf = math.log(1 + (passages - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / mean_length))


class TestTokenizedCorpus:
    def test_every_passage_has_its_run_of_tokens(self):
        corpus = tokenize_corpus(TEXTS)
        assert corpus.passages == 4
        assert corpus.offsets.tolist() == [0, 3, 5, 6, 11]
        assert corpus.vocabulary == ["röd", "blå", "grön", "gul"]
        assert corpus.tokens[:3].tolist() == [0, 0, 1]


class TestBuild:
    def test_lengths_and_document_frequencies(self, index):
        assert index.doc_lengths.tolist() == [3, 2, 1, 5]
        assert index.document_frequency("röd") == 2
        assert index.document_frequency("gul") == 2
        assert index.document_frequency("lila") == 0

    def test_stop_words_are_left_out_of_the_lengths(self):
        corpus = tokenize_corpus([("a:0-0", "och skolan och"), ("b:0-0", "skolan")])
        index = build_index(corpus, Analyzer(stem=False, stopwords=True))
        assert index.doc_lengths.tolist() == [1, 1]
        assert index.document_frequency("och") == 0

    def test_stemming_folds_inflections_into_one_term(self):
        corpus = tokenize_corpus([("a:0-0", "skolorna"), ("b:0-0", "skolor")])
        index = build_index(corpus, Analyzer(stem=True, stopwords=False))
        assert index.document_frequency("skol") == 2


class TestScore:
    def test_a_score_is_bm25_written_out(self, index):
        scores = index.scores("röd", DEFAULT_PARAMS)
        mean = 11 / 4
        assert scores[0] == pytest.approx(bm25(2, 2, 3, mean, 4))
        assert scores[3] == pytest.approx(bm25(1, 2, 5, mean, 4))
        assert scores[1] == scores[2] == 0

    def test_terms_add_up(self, index):
        both = index.scores("röd blå", DEFAULT_PARAMS)
        apart = index.scores("röd", DEFAULT_PARAMS) + index.scores("blå", DEFAULT_PARAMS)
        assert both == pytest.approx(apart)

    def test_a_repeated_query_term_counts_once(self, index):
        once = index.scores("röd", DEFAULT_PARAMS)
        assert index.scores("röd röd röd", DEFAULT_PARAMS) == pytest.approx(once)

    def test_k1_and_b_change_the_scores_without_a_new_index(self, index):
        before = index.scores("gul", DEFAULT_PARAMS)
        after = index.scores("gul", Bm25Params(k1=0.5, b=0.0))
        assert after[2] == pytest.approx(bm25(1, 2, 1, 11 / 4, 4, k1=0.5, b=0.0))
        assert not np.allclose(before, after)

    def test_bounds_on_the_parameters(self):
        with pytest.raises(ValueError):
            Bm25Params(k1=-1, b=0.5)
        with pytest.raises(ValueError):
            Bm25Params(k1=1, b=1.5)


class TestSearch:
    def test_hits_come_best_first_and_none_without_a_query_term(self, index):
        hits = index.search("gul", DEFAULT_PARAMS)
        assert [hit.passage for hit in hits] == [3, 2]
        assert hits[0].score > hits[1].score

    def test_a_query_of_unknown_words_finds_nothing(self, index):
        assert index.search("lila", DEFAULT_PARAMS) == []

    def test_the_mask_keeps_out_what_it_does_not_allow(self, index):
        allowed = np.array([False, True, True, True])
        hits = index.search("röd", DEFAULT_PARAMS, allowed=allowed)
        assert [hit.passage for hit in hits] == [3]

    def test_the_limit_cuts_the_ranking(self, index):
        assert len(index.search("röd blå gul grön", DEFAULT_PARAMS, limit=2)) == 2

    def test_equal_scores_rank_in_index_order(self):
        corpus = tokenize_corpus([("a:0-0", "x y"), ("b:0-0", "x y"), ("c:0-0", "x y")])
        index = build_index(corpus, PLAIN)
        assert [hit.passage for hit in index.search("x", DEFAULT_PARAMS)] == [0, 1, 2]


class TestDisk:
    def manifest(self, index: LexicalIndex) -> IndexManifest:
        return IndexManifest(
            analyzer=index.analyzer,
            passages=index.passages,
            passage_ids_sha256=passage_ids_sha256([key for key, _ in TEXTS]),
            passages_sha256="0" * 64,
            speeches_sha256="0" * 64,
            terms=len(index.terms),
            postings=len(index.doc_ids),
            tokens=int(index.doc_lengths.sum()),
            mean_passage_terms=index.mean_length,
        )

    def test_an_index_reads_back_as_written(self, index, tmp_path):
        index.save(tmp_path, self.manifest(index))
        loaded, manifest = LexicalIndex.load(tmp_path)
        assert manifest.analyzer == PLAIN
        query = "röd gul"
        assert loaded.scores(query, DEFAULT_PARAMS) == pytest.approx(
            index.scores(query, DEFAULT_PARAMS)
        )

    def test_an_index_that_does_not_match_its_manifest_is_refused(self, index, tmp_path):
        manifest = self.manifest(index).model_copy(update={"postings": 1})
        index.save(tmp_path, manifest)
        with pytest.raises(ValueError, match="does not match"):
            LexicalIndex.load(tmp_path)

    def test_a_file_hash_is_the_hash_of_its_bytes(self, tmp_path):
        path = tmp_path / "x.bin"
        path.write_bytes(b"abc")
        assert file_sha256(path) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )


class TestPassageTexts:
    def test_each_passage_holds_its_paragraphs_in_order(self, tmp_path):
        speeches_path, passages_path = write_corpus(tmp_path)
        texts = dict(passage_texts(speeches_path, passages_path))
        whole = "\n".join(SPEECHES[0].paragraphs)
        assert texts[f"{SPEECHES[0].anf_id}:0-1"] == whole
        assert len(texts) == len(SPEECHES)

    def test_files_out_of_order_are_refused(self, tmp_path):
        first, second = SPEECHES[0], SPEECHES[2]
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        speeches_path, _ = write_corpus(tmp_path / "a", [first, second])
        _, passages_path = write_corpus(tmp_path / "b", [second, first])
        with pytest.raises(ValueError, match="same order"):
            list(passage_texts(speeches_path, passages_path))

    def test_a_speech_the_passages_skip_is_passed_over(self, tmp_path):
        extra = speech("GR0951", 1, ["Ett anförande utan passager."], party="C")
        (tmp_path / "a").mkdir()
        speeches_path, _ = write_corpus(tmp_path / "a", [SPEECHES[0], extra, SPEECHES[2]])
        (tmp_path / "b").mkdir()
        _, passages_path = write_corpus(tmp_path / "b", [SPEECHES[0], SPEECHES[2]])
        assert len(list(passage_texts(speeches_path, passages_path))) == 2
