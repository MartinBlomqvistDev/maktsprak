"""Tests for search_service/spike/probe_fts.py, on the kit's corpus. Skipped without LanceDB."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("lancedb")

import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

from search.index.lexical import (  # noqa: E402
    Bm25Params,
    LexicalIndex,
    lexical_dir,
)
from search_service.spike.probe_fts import (  # noqa: E402
    _fts,
    check_folding,
    check_int8,
    check_long_words,
    check_questions,
    check_stemming,
    counted_each_time,
    fit_one_word,
)
from search_service.spike.settings import ProbeSettings  # noqa: E402
from tests.search_service.spike_kit import (  # noqa: E402
    CHOSEN,
    RAW,
    kit_settings,
    write_spike_corpus,
)

PROBE = ProbeSettings(
    seed=1,
    common_tokens=20,
    drawn_tokens=20,
    passage_step=1,
    one_word_queries=("skolan",),
    hits_per_query=50,
    k1_grid=(0.6, 1.8, 0.3),
    b_grid=(0.0, 1.0, 0.25),
    questions=3,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_spike_corpus(tmp_path_factory.mktemp("spike"))


@pytest.fixture
def database(tmp_path):
    return lancedb.connect(tmp_path / "probe")


@pytest.fixture(scope="module")
def vocabulary(corpus):
    directory = lexical_dir(corpus.search_dir, RAW)
    words = json.loads((directory / "terms.json").read_text(encoding="utf-8"))
    return words, np.diff(np.load(directory / "term_ptr.npy"))


@pytest.fixture(scope="module")
def indexed(corpus, tmp_path_factory):
    index, _ = LexicalIndex.load(lexical_dir(corpus.search_dir, CHOSEN))
    database = lancedb.connect(tmp_path_factory.mktemp("passages"))
    table = database.create_table(
        "passages", pa.table({"id": list(range(len(corpus.texts))), "text": corpus.texts})
    )
    table.create_index("text", config=_fts(kit_settings().lance.fts))
    return table, index


def test_the_stems_group_the_kits_words_as_snowball_does(database, vocabulary):
    words, frequency = vocabulary
    result = check_stemming(database, words, frequency, PROBE, kit_settings().lance.fts)
    assert result.words > 20
    assert (result.unmatched, result.differing) == (0, 0)


def test_folding_makes_far_and_far_with_a_ring_one_word(database):
    result = check_folding(database, kit_settings().lance.fts)
    assert result.folded["far"] == ("far", "får")
    assert result.unfolded["far"] == ("far",)


def test_a_word_beyond_the_default_limit_is_dropped(database, vocabulary):
    words, frequency = vocabulary
    longer = [*words, "x" * 41]
    result = check_long_words(database, longer, np.append(frequency, 1), kit_settings().lance.fts)
    assert result.corpus_words_over == 1
    assert (result.found_at_default, result.found_without_limit) == (False, True)


def test_one_word_fits_lucenes_defaults(indexed):
    table, index = indexed
    fit = fit_one_word(table, index, "skolan", PROBE)
    assert (fit.k1, fit.b) == (1.2, 0.75)
    assert fit.max_deviation < 1e-5
    assert fit.implied_idf == pytest.approx(fit.lucene_idf, rel=1e-5)


def test_a_repeated_word_counts_each_time(indexed):
    table, index = indexed
    params = Bm25Params(k1=1.2, b=0.75)
    result = check_questions(table, index, ["skolan skolan lärare", "el el el", "vården"], params)
    assert result.questions_with_a_repeated_term == 2
    assert result.agree_counted_each_time == 3
    assert result.agree_counted_once == 1


def test_counting_each_time_doubles_a_doubled_word(indexed):
    _, index = indexed
    params = Bm25Params(k1=1.2, b=0.75)
    np.testing.assert_allclose(
        counted_each_time(index, "skolan skolan", params), 2 * index.scores("skolan", params)
    )


def test_an_int8_column_cannot_be_searched_with_a_float_query(database):
    assert "Int8" in check_int8(database)
