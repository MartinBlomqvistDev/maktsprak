"""Tests for search_service/spike/lance_store.py, on the kit's corpus. Skipped without LanceDB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lancedb")

from search.parties import PARTY_ORDER  # noqa: E402
from search_service.spike.contracts import (  # noqa: E402
    PartyViewRequest,
    SearchFilter,
    SearchRequest,
)
from search_service.spike.inputs import CorpusFiles  # noqa: E402
from search_service.spike.lance_store import (  # noqa: E402
    LanceStore,
    build_lance_store,
    where_clause,
)
from search_service.spike.numpy_store import NumpyStore, build_numpy_store  # noqa: E402
from search_service.spike.settings import AnnParams  # noqa: E402
from tests.search_service.spike_kit import (  # noqa: E402
    SPEECHES,
    kit_settings,
    write_spike_corpus,
)

#: A question with a word from the passages every filter of the tests allows.
EVERY_FILTER = "skolan skolor jobb el sjukvården vårdköerna"


@dataclass(frozen=True)
class Built:
    store: LanceStore
    directory: Path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_spike_corpus(tmp_path_factory.mktemp("spike"))


@pytest.fixture(scope="module", params=["lance-8-float32", "lance-8-float16"])
def built(corpus, tmp_path_factory, request) -> Built:
    settings = kit_settings()
    spec = settings.store(request.param)
    directory = tmp_path_factory.mktemp(request.param)
    files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
    build_lance_store(directory, files, spec, settings)
    return Built(LanceStore(directory, spec, settings), directory)


@pytest.fixture(scope="module")
def numpy_store(corpus, tmp_path_factory) -> NumpyStore:
    settings = kit_settings()
    directory = tmp_path_factory.mktemp("numpy")
    files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
    build_numpy_store(directory, files, [(8, "float32")], settings.vectors)
    return NumpyStore(directory, 8, "float32", settings.search)


def speech_of(passage_id: str):
    return next(s for s in SPEECHES if s.anf_id == passage_id.split(":")[0])


class TestWhereClause:
    def test_no_filter_gives_no_predicate(self):
        assert where_clause(SearchFilter()) is None

    def test_every_field_becomes_a_condition(self):
        search_filter = SearchFilter(
            party="M", start=date(2010, 1, 1), end=date(2017, 12, 31), person_id="0202"
        )
        assert where_clause(search_filter) == (
            "party = 'M' AND day >= DATE '2010-01-01' AND day <= DATE '2017-12-31' "
            "AND person_id = '0202'"
        )

    def test_a_person_id_must_be_digits(self):
        with pytest.raises(ValueError):
            SearchFilter(person_id="0' OR '1'='1")


class TestBuild:
    def test_the_manifest_records_the_indexes(self, built):
        names = {index.name for index in built.store.manifest.indices}
        assert {"text_idx", "party_idx", "day_idx", "person_id_idx"} <= names
        fts = next(index for index in built.store.manifest.indices if index.name == "text_idx")
        assert fts.details["language"] == "Swedish"
        assert fts.details["ascii_folding"] is False

    def test_a_store_opened_as_another_is_refused(self, built):
        other = "lance-8-float16" if built.store.spec.format == "float32" else "lance-8-float32"
        with pytest.raises(ValueError, match="not built"):
            LanceStore(built.directory, kit_settings().store(other), kit_settings())

    def test_the_vectors_are_those_of_the_numpy_store(self, built, numpy_store):
        rows = built.store.table.to_arrow().select(["passage_id", "vector"]).to_pylist()
        assert [row["passage_id"] for row in rows] == [
            numpy_store.passage_ids[i] for i in range(numpy_store.passages)
        ]
        vectors = np.array([row["vector"] for row in rows], dtype=np.float32)
        tolerance = 0 if built.store.spec.format == "float32" else 1e-3
        np.testing.assert_allclose(vectors, np.asarray(numpy_store.vectors.matrix), atol=tolerance)


class TestSearch:
    def test_the_swedish_stemmer_matches_inflected_forms(self, built):
        store = built.store
        response = store.search(
            SearchRequest(text="kärnkraftsavvecklingen", vector_seed=1, mode="lexical")
        )
        assert [hit.passage_id for hit in response.hits] == ["H109050-1:0-1"]
        response = store.search(SearchRequest(text="avvecklingen", vector_seed=1, mode="lexical"))
        assert "H109050-2:0-1" in [hit.passage_id for hit in response.hits]

    def test_hits_carry_their_text_and_fields(self, built, corpus):
        for hit in built.store.search(SearchRequest(text="skolan", vector_seed=1)).hits:
            speech = speech_of(hit.passage_id)
            assert hit.text == corpus.texts[corpus.passage_ids.index(hit.passage_id)]
            assert (hit.party, hit.date) == (speech.party, speech.date)

    def test_a_ranking_reads_no_text_and_the_hits_keep_its_order(self, built):
        store = built.store
        vector = store.query(1, None)
        for mode in ("hybrid", "lexical", "vector"):
            ranked = store._run("skolan", vector, mode, None, AnnParams(), 50)
            assert "text" not in ranked.column_names
            hits = store.search(SearchRequest(text="skolan", vector_seed=1, mode=mode)).hits
            expected = ranked["passage_id"].to_pylist()[: len(hits)]
            assert [hit.passage_id for hit in hits] == expected

    @pytest.mark.parametrize(
        "search_filter",
        [
            SearchFilter(party="M"),
            SearchFilter(start=date(2010, 1, 1), end=date(2017, 12, 31)),
            SearchFilter(person_id="0303"),
        ],
    )
    def test_every_hit_passes_the_filter_in_every_mode(self, built, search_filter):
        for mode in ("hybrid", "lexical", "vector"):
            request = SearchRequest(
                text=EVERY_FILTER, vector_seed=2, filter=search_filter, mode=mode
            )
            response = built.store.search(request)
            assert response.hits
            for hit in response.hits:
                speech = speech_of(hit.passage_id)
                assert search_filter.party in (None, speech.party)
                assert search_filter.person_id in (None, speech.person_id)
                assert search_filter.start is None or speech.date >= search_filter.start
                assert search_filter.end is None or speech.date <= search_filter.end

    def test_the_exact_vector_search_agrees_with_numpy(self, built, numpy_store):
        request = SearchRequest(text="x", vector_seed=8, mode="vector")
        ours = [hit.passage_id for hit in numpy_store.search(request).hits]
        theirs = [hit.passage_id for hit in built.store.search(request).hits]
        if built.store.spec.format == "float32":
            assert theirs == ours
        else:
            assert set(theirs) & set(ours)

    def test_the_party_view_gives_each_party_its_own(self, built):
        response = built.store.party_view(PartyViewRequest(text="kärnkraft el", vector_seed=5))
        assert list(response.hits) == list(PARTY_ORDER)
        for party, hit in response.hits.items():
            assert hit is not None and hit.party == party
