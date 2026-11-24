"""Tests for search_service/spike/numpy_store.py and inputs.py, on the kit's corpus."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from search.index.lexical import LexicalIndex, file_sha256, lexical_dir, passage_ids_sha256
from search.parties import PARTY_ORDER
from search_service.spike.contracts import PartyViewRequest, SearchFilter, SearchRequest
from search_service.spike.inputs import (
    CorpusFiles,
    day_number,
    day_of,
    read_bm25_choice,
    read_passage_columns,
)
from search_service.spike.numpy_store import NumpyStore, build_numpy_store, real_vectors
from search_service.spike.vectors import VectorMatrix, query_vector, top_positions
from tests.search_service.spike_kit import CHOSEN, SPEECHES, kit_settings, write_spike_corpus

SHAPES = [(8, "float32"), (8, "float16"), (8, "int8")]

#: A question with a word from the passages every filter of the tests allows.
EVERY_FILTER = "skolan skolor jobb el sjukvården vårdköerna"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_spike_corpus(tmp_path_factory.mktemp("spike"))


@pytest.fixture(scope="module")
def built(corpus, tmp_path_factory):
    directory = tmp_path_factory.mktemp("numpy")
    files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
    manifest = build_numpy_store(directory, files, SHAPES, kit_settings().vectors)
    return directory, manifest


@pytest.fixture(scope="module", params=["float32", "float16", "int8"])
def store(built, request) -> NumpyStore:
    return NumpyStore(built[0], 8, request.param, kit_settings().search)


def speech_of(passage_id: str):
    return next(s for s in SPEECHES if s.anf_id == passage_id.split(":")[0])


class TestInputs:
    def test_the_tuning_report_names_the_bm25(self, corpus):
        choice = read_bm25_choice(corpus.tuning)
        assert choice.analyzer == CHOSEN
        assert (choice.params.k1, choice.params.b) == (1.2, 0.9)

    def test_an_unknown_analyzer_is_refused(self, tmp_path):
        path = tmp_path / "tuning.json"
        path.write_text(json.dumps({"chosen": {"analyzer": "x", "k1": 1.0, "b": 0.5}}))
        with pytest.raises(ValueError, match="analyzer"):
            read_bm25_choice(path)

    def test_the_columns_are_in_index_order(self, corpus):
        files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
        columns = read_passage_columns(files.passages)
        assert columns.passage_ids == corpus.passage_ids
        assert day_of(int(columns.days[0])) == speech_of(columns.passage_ids[0]).date
        assert len(read_passage_columns(files.passages, limit=4)) == 4

    def test_day_numbers_round_trip(self):
        assert day_number(date(1970, 1, 2)) == 1
        assert day_of(day_number(date(2023, 2, 14))) == date(2023, 2, 14)


class TestBuild:
    def test_the_manifest_counts_what_was_built(self, built, corpus):
        _, manifest = built
        assert manifest.passages == len(corpus.passage_ids)
        assert manifest.persons == len({s.person_id for s in SPEECHES if s.person_id})
        assert {(v.dim, v.format) for v in manifest.vectors} == set(SHAPES)
        assert (manifest.params.k1, manifest.params.b) == (1.2, 0.9)

    def test_the_bm25_index_is_the_tunings_own(self, built, corpus):
        copied, _ = LexicalIndex.load(built[0] / "lexical")
        original, _ = LexicalIndex.load(lexical_dir(corpus.search_dir, CHOSEN))
        np.testing.assert_array_equal(copied.doc_ids, original.doc_ids)
        assert copied.terms == original.terms

    def test_an_index_of_another_corpus_is_refused(self, corpus, tmp_path):
        files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
        manifest_path = files.lexical / "manifest.json"
        saved = manifest_path.read_text(encoding="utf-8")
        try:
            manifest_path.write_text(saved.replace('"passages": 12', '"passages": 13'))
            with pytest.raises(ValueError, match="passages"):
                build_numpy_store(tmp_path, files, SHAPES[:1], kit_settings().vectors)
        finally:
            manifest_path.write_text(saved, encoding="utf-8")

    def test_a_limited_build_indexes_only_the_passages_kept(self, corpus, tmp_path):
        files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
        manifest = build_numpy_store(tmp_path, files, SHAPES[:1], kit_settings().vectors, limit=5)
        assert manifest.passages == 5
        assert NumpyStore(tmp_path, 8, "float32", kit_settings().search).lexical.passages == 5

    def test_a_shape_that_was_not_built_is_refused(self, built):
        with pytest.raises(ValueError, match="no 16-dimensional"):
            NumpyStore(built[0], 16, "float32", kit_settings().search)


class TestSearch:
    def test_hits_carry_their_text_and_fields(self, store, corpus):
        response = store.search(SearchRequest(text="kärnkraft", vector_seed=1, mode="lexical"))
        assert response.hits
        for hit in response.hits:
            assert hit.text == corpus.texts[corpus.passage_ids.index(hit.passage_id)]
            assert hit.party == speech_of(hit.passage_id).party
            assert hit.date == speech_of(hit.passage_id).date

    def test_the_lexical_leg_is_the_tuned_bm25(self, store):
        response = store.search(SearchRequest(text="kärnkraften", vector_seed=1, mode="lexical"))
        expected = store.lexical.search("kärnkraften", store.params, limit=3)
        assert [hit.passage_id for hit in response.hits] == [
            store.passage_ids[hit.passage] for hit in expected
        ]

    def test_the_vector_leg_is_exact(self, store):
        response = store.search(SearchRequest(text="x", vector_seed=4, mode="vector"))
        exact = top_positions(store.vectors.scores(query_vector(4, 8)), 3)
        assert [hit.passage_id for hit in response.hits] == [store.passage_ids[i] for i in exact]

    @pytest.mark.parametrize(
        "search_filter",
        [
            SearchFilter(party="M"),
            SearchFilter(start=date(2010, 1, 1), end=date(2017, 12, 31)),
            SearchFilter(person_id="0303"),
            SearchFilter(party="S", start=date(2018, 1, 1)),
        ],
    )
    def test_every_hit_passes_the_filter_in_every_mode(self, store, search_filter):
        for mode in ("hybrid", "lexical", "vector"):
            request = SearchRequest(
                text=EVERY_FILTER, vector_seed=2, filter=search_filter, mode=mode
            )
            response = store.search(request)
            assert response.hits
            for hit in response.hits:
                speech = speech_of(hit.passage_id)
                assert search_filter.party in (None, speech.party)
                assert search_filter.person_id in (None, speech.person_id)
                assert search_filter.start is None or speech.date >= search_filter.start
                assert search_filter.end is None or speech.date <= search_filter.end

    def test_the_filter_counts_what_it_allows(self, store):
        response = store.search(
            SearchRequest(text="el", vector_seed=1, filter=SearchFilter(party="M"))
        )
        assert response.allowed == sum(s.party == "M" for s in SPEECHES)

    def test_an_unknown_speaker_allows_nothing(self, store):
        request = SearchRequest(text="el", vector_seed=1, filter=SearchFilter(person_id="9999"))
        response = store.search(request)
        assert (response.allowed, response.hits) == (0, ())

    def test_a_hybrid_search_returns_at_most_the_hits_asked_for(self, store):
        response = store.search(SearchRequest(text="skolan", vector_seed=3))
        assert len(response.hits) == 3
        assert response.timings.lexical_ms is not None and response.timings.vector_ms is not None


class TestPartyView:
    def test_each_party_gets_its_own_best_passage(self, store):
        response = store.party_view(PartyViewRequest(text="kärnkraft el", vector_seed=5))
        assert list(response.hits) == list(PARTY_ORDER)
        for party, hit in response.hits.items():
            assert hit is not None and hit.party == party

    def test_a_party_with_nothing_under_the_filter_gets_none(self, store):
        request = PartyViewRequest(
            text="skolan", vector_seed=5, filter=SearchFilter(end=date(2009, 12, 31))
        )
        response = store.party_view(request)
        assert response.hits["SD"] is None
        assert response.hits["S"] is not None and response.hits["S"].date.year == 2004

    def test_the_view_chooses_the_party_itself(self):
        with pytest.raises(ValueError, match="chooses the party"):
            PartyViewRequest(text="x", vector_seed=1, filter=SearchFilter(party="M"))


def test_the_store_and_the_matrix_agree_on_the_vectors(built):
    directory, _ = built
    matrix = VectorMatrix.load(directory, 8, "float32", chunk_rows=5)
    assert len(matrix) == 12


def write_index(directory, ids, matrix):
    """A vector index as search/index/vector.py writes it, with the manifest fields read back."""
    directory.mkdir()
    np.save(directory / "vectors.npy", matrix)
    np.save(directory / "positions.npy", np.arange(len(ids), dtype=np.int64))
    manifest = {
        "rows": len(ids),
        "dimension": matrix.shape[1],
        "vectors_sha256": file_sha256(directory / "vectors.npy"),
        "positions_sha256": file_sha256(directory / "positions.npy"),
        "inputs": {"passage_ids_sha256": passage_ids_sha256(ids)},
    }
    (directory / "index.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


class TestRealVectors:
    """A store of real vectors, from an index whose loading is stood in for."""

    @pytest.fixture
    def real(self, corpus, tmp_path):
        files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
        ids = list(read_passage_columns(files.passages).passage_ids)
        matrix = np.random.default_rng(7).standard_normal((len(ids), 8)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        index = write_index(tmp_path / "index", ids, matrix)
        queries = tmp_path / "queries"
        queries.mkdir()
        np.save(queries / "queries.npy", np.stack([matrix[3], matrix[3], matrix[5]]))
        texts = {"query_ids": ["q1", "q2", "q3"], "texts": ["fråga ett", "fråga ett", "fråga tre"]}
        (queries / "query_texts.json").write_text(json.dumps(texts), encoding="utf-8")
        directory = tmp_path / "store"
        manifest = build_numpy_store(
            directory,
            files,
            [(8, "float32")],
            kit_settings().vectors,
            vectors_from=index,
            queries_from=queries,
        )
        return directory, manifest, ids, matrix, index

    def test_the_store_holds_the_real_vectors_of_its_passages(self, real):
        directory, manifest, _, matrix, index = real
        np.testing.assert_array_equal(np.load(directory / "vectors-8-float32.npy"), matrix)
        assert manifest.vectors_source_sha256 == file_sha256(index / "vectors.npy")
        assert manifest.query_vectors == 2

    def test_an_index_of_other_passages_or_altered_files_is_refused(self, real):
        _, _, ids, matrix, index = real
        with pytest.raises(ValueError, match="other passages"):
            real_vectors(write_index(index.parent / "other", ids[::-1], matrix), ids)
        np.save(index / "vectors.npy", matrix * 2)
        with pytest.raises(ValueError, match="does not match"):
            real_vectors(index, ids)

    def test_a_question_is_searched_with_its_own_vector(self, real):
        directory, _, ids, _, _ = real
        store = NumpyStore(directory, 8, "float32", kit_settings().search)
        first = store.search(SearchRequest(text="fråga ett", vector_seed=0, mode="vector"))
        third = store.search(SearchRequest(text="fråga tre", vector_seed=0, mode="vector"))
        assert first.hits[0].passage_id == ids[3]
        assert third.hits[0].passage_id == ids[5]

    def test_a_question_the_store_holds_no_vector_for_is_refused(self, real):
        store = NumpyStore(real[0], 8, "float32", kit_settings().search)
        with pytest.raises(ValueError, match="no vector"):
            store.search(SearchRequest(text="en okänd fråga", vector_seed=0, mode="vector"))

    def test_real_vectors_take_one_shape_and_the_questions(self, corpus, tmp_path):
        files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
        wrong = [([(8, "float32"), (8, "int8")], tmp_path), ([(8, "float32")], None)]
        for shapes, queries in wrong:
            with pytest.raises(ValueError, match="one shape"):
                build_numpy_store(
                    tmp_path / "store",
                    files,
                    shapes,
                    kit_settings().vectors,
                    vectors_from=tmp_path,
                    queries_from=queries,
                )
