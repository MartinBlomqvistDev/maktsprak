"""Tests for search/index/vector.py: exact search and the index of step 07."""

from __future__ import annotations

import numpy as np
import pytest

from search.index.embed import HashEmbedder, embed_inputs, read_inputs, write_inputs
from search.index.vector import (
    QUERY_BLOCK,
    VectorIndex,
    build_vector_index,
    load_vector_index,
    passages_path,
)

UNIT = np.eye(4, dtype=np.float32)


def index_of(rows, positions=None) -> VectorIndex:
    vectors = np.asarray(rows, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return VectorIndex(vectors, np.arange(len(vectors)) if positions is None else positions)


class TestSearch:
    def test_the_nearest_passages_come_first(self):
        index = index_of([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]])
        hits = index.search(UNIT[0], limit=3)
        assert [hit.passage for hit in hits] == [0, 1, 2]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score == pytest.approx(1 / np.sqrt(2))

    def test_the_limit_cuts_the_ranking(self):
        index = index_of([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]])
        assert [hit.passage for hit in index.search(UNIT[0], limit=1)] == [0]

    def test_rows_report_their_passages_positions(self):
        index = index_of([[0, 1, 0, 0], [1, 0, 0, 0]], positions=np.array([4, 9]))
        assert [hit.passage for hit in index.search(UNIT[0], limit=2)] == [9, 4]

    def test_a_mask_over_the_passages_file_keeps_others_out(self):
        index = index_of([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]], positions=np.array([1, 3, 5]))
        allowed = np.zeros(6, dtype=bool)
        allowed[[3, 5]] = True
        assert [hit.passage for hit in index.search(UNIT[0], 3, allowed)] == [3, 5]

    def test_an_empty_mask_returns_nothing(self):
        index = index_of([[1, 0, 0, 0]])
        assert index.search(UNIT[0], 5, np.zeros(1, dtype=bool)) == []

    def test_equal_scores_go_in_passage_order(self):
        index = index_of([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        assert [hit.passage for hit in index.search(UNIT[0], 3)] == [0, 2, 1]

    def test_positions_out_of_order_are_refused(self):
        with pytest.raises(ValueError, match="ascending"):
            index_of([[0, 1, 0, 0], [1, 0, 0, 0]], positions=np.array([7, 5]))

    def test_many_questions_match_one_at_a_time(self):
        rng = np.random.default_rng(3)
        index = index_of(rng.normal(size=(50, 4)))
        queries = rng.normal(size=(QUERY_BLOCK + 5, 4)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        together = index.search_many(queries, 7, [None] * len(queries))
        apart = [index.search(query, 7) for query in queries]
        for batch, single in zip(together, apart, strict=True):
            assert [hit.passage for hit in batch] == [hit.passage for hit in single]
            np.testing.assert_allclose(
                [hit.score for hit in batch], [hit.score for hit in single], atol=1e-6
            )

    def test_the_ranking_is_the_exact_one(self):
        rng = np.random.default_rng(4)
        index = index_of(rng.normal(size=(200, 8)))
        query = rng.normal(size=8).astype(np.float32)
        query /= np.linalg.norm(query)
        expected = np.argsort(-(index.vectors @ query), kind="stable")[:20]
        assert [hit.passage for hit in index.search(query, 20)] == expected.tolist()

    def test_questions_of_another_size_are_refused(self):
        with pytest.raises(ValueError):
            index_of([[1, 0, 0, 0]]).search_many(np.ones((1, 3), dtype=np.float32), 1, [None])


PASSAGES = [(0, "a:0-0", "röd blå"), (1, "b:0-0", "grön gul"), (3, "d:0-0", "röd grön")]
QUERIES = [("natural-0001", "natural", "röd")]


@pytest.fixture
def run(tmp_path):
    inputs = tmp_path / "inputs"
    write_inputs(inputs, "subset-200", 200, PASSAGES, QUERIES, {})
    run_dir = tmp_path / "run"
    embed_inputs(read_inputs(inputs), HashEmbedder(16), run_dir, shard_rows=2)
    return inputs, run_dir


class TestIndex:
    def test_an_index_holds_the_runs_vectors_and_positions(self, run, tmp_path):
        inputs, run_dir = run
        manifest = build_vector_index(run_dir, inputs, tmp_path / "index")
        index, loaded = load_vector_index(tmp_path / "index", ["a:0-0", "b:0-0", "c:0-0", "d:0-0"])
        assert loaded == manifest and manifest.rows == 3
        assert index.positions.tolist() == [0, 1, 3]
        query = HashEmbedder(16).embed(["röd"], "query")[0]
        assert {hit.passage for hit in index.search(query, 2)} == {0, 3}

    def test_an_index_against_other_passages_is_refused(self, run, tmp_path):
        inputs, run_dir = run
        build_vector_index(run_dir, inputs, tmp_path / "index")
        with pytest.raises(ValueError, match="other passages"):
            load_vector_index(tmp_path / "index", ["a:0-0", "x:0-0", "c:0-0", "d:0-0"])
        with pytest.raises(ValueError, match="reaches past"):
            load_vector_index(tmp_path / "index", ["a:0-0", "b:0-0"])

    def test_a_changed_matrix_is_caught(self, run, tmp_path):
        inputs, run_dir = run
        build_vector_index(run_dir, inputs, tmp_path / "index")
        np.save(tmp_path / "index" / "vectors.npy", np.zeros((3, 16), dtype=np.float32))
        with pytest.raises(ValueError, match="does not match"):
            load_vector_index(tmp_path / "index")

    def test_a_run_that_is_not_whole_is_refused(self, tmp_path):
        inputs = tmp_path / "inputs"
        write_inputs(inputs, "subset-200", 200, PASSAGES, QUERIES, {})
        run_dir = tmp_path / "run"
        embed_inputs(read_inputs(inputs), HashEmbedder(16), run_dir, shard_rows=2, max_shards=1)
        with pytest.raises(ValueError, match="not complete"):
            build_vector_index(run_dir, inputs, tmp_path / "index")

    def test_a_run_of_other_inputs_is_refused(self, run, tmp_path):
        _, run_dir = run
        other = tmp_path / "other"
        write_inputs(other, "subset-200", 200, PASSAGES[:2], QUERIES, {})
        with pytest.raises(ValueError, match="not made from"):
            build_vector_index(run_dir, other, tmp_path / "index")


def test_the_corpus_own_size_has_the_plain_file_name(tmp_path):
    assert passages_path(tmp_path, 200, 200).name == "passages.parquet"
    assert passages_path(tmp_path, 120, 200).name == "passages-120.parquet"
