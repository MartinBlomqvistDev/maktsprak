"""Tests for search_service/spike/vectors.py."""

from __future__ import annotations

import numpy as np
import pytest

from search_service.spike.vectors import (
    VectorMatrix,
    encode,
    int8_scales,
    query_vector,
    scales_name,
    synthetic_chunks,
    top_positions,
    write_vectors,
)


def all_vectors(rows: int, dim: int, seed: int = 1, chunk_rows: int = 4) -> np.ndarray:
    return np.concatenate(list(synthetic_chunks(rows, dim, seed, chunk_rows)))


class TestSyntheticVectors:
    def test_they_are_unit_vectors_of_the_shape_asked_for(self):
        vectors = all_vectors(10, 6)
        assert vectors.shape == (10, 6)
        assert vectors.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)

    def test_the_same_seed_gives_the_same_vectors(self):
        np.testing.assert_array_equal(all_vectors(10, 6), all_vectors(10, 6))
        assert not np.array_equal(all_vectors(10, 6, seed=1), all_vectors(10, 6, seed=2))

    def test_a_chunk_does_not_depend_on_how_many_rows_follow(self):
        np.testing.assert_array_equal(all_vectors(9, 6)[:8], all_vectors(20, 6)[:8])

    def test_a_query_vector_is_a_unit_vector_of_its_own(self):
        first = query_vector(5, 6)
        assert first.shape == (6,)
        assert abs(float(np.linalg.norm(first)) - 1) < 1e-6
        np.testing.assert_array_equal(first, query_vector(5, 6))
        assert not np.array_equal(first, query_vector(6, 6))


class TestFormats:
    def test_float16_keeps_the_values_to_half_precision(self):
        block = all_vectors(5, 6)
        stored = encode(block, "float16")
        assert stored.dtype == np.float16
        np.testing.assert_allclose(stored.astype(np.float32), block, atol=1e-3)

    def test_int8_maps_each_dimensions_largest_component_to_127(self):
        scales = int8_scales(10, 6, 1, 4)
        codes = encode(all_vectors(10, 6), "int8", scales)
        assert codes.dtype == np.int8
        assert np.all(np.abs(codes).max(axis=0) == 127)

    def test_int8_decodes_to_within_half_a_step(self):
        vectors = all_vectors(10, 6)
        scales = int8_scales(10, 6, 1, 4)
        decoded = encode(vectors, "int8", scales).astype(np.float32) * scales
        assert np.all(np.abs(decoded - vectors) <= scales / 2 + 1e-7)

    def test_int8_needs_scales(self):
        with pytest.raises(ValueError, match="scale"):
            encode(all_vectors(2, 6), "int8")


class TestTopPositions:
    def test_best_first_and_ties_in_order(self):
        scores = np.array([0.1, 0.5, 0.5, 0.9, 0.2])
        assert top_positions(scores, 3).tolist() == [3, 1, 2]

    def test_positions_stand_for_other_rows(self):
        scores = np.array([0.1, 0.9])
        assert top_positions(scores, 1, np.array([40, 70])).tolist() == [70]

    def test_fewer_scores_than_asked_for(self):
        assert top_positions(np.array([0.3, 0.4]), 5).tolist() == [1, 0]


@pytest.fixture(params=["float32", "float16", "int8"])
def matrix(request, tmp_path) -> tuple[VectorMatrix, np.ndarray]:
    write_vectors(tmp_path, 30, 8, request.param, seed=3, chunk_rows=7)
    return VectorMatrix.load(tmp_path, 8, request.param, chunk_rows=4), all_vectors(30, 8, 3, 7)


class TestVectorMatrix:
    def test_scores_are_the_dot_products(self, matrix):
        store, exact = matrix
        query = query_vector(9, 8)
        tolerance = 1e-6 if store.matrix.dtype == np.float32 else 3e-2
        np.testing.assert_allclose(store.scores(query), exact @ query, atol=tolerance)

    def test_scoring_some_rows_gives_their_scores(self, matrix):
        store, _ = matrix
        query = query_vector(9, 8)
        rows = np.array([2, 5, 29])
        np.testing.assert_allclose(store.scores(query, rows), store.scores(query)[rows], rtol=1e-5)

    def test_a_search_keeps_to_the_mask(self, matrix):
        store, _ = matrix
        allowed = np.zeros(30, dtype=bool)
        allowed[[1, 4, 7, 20]] = True
        found = store.search(query_vector(9, 8), 3, allowed, gather_below=0.0)
        assert set(found.tolist()) <= {1, 4, 7, 20}
        assert len(found) == 3

    def test_scoring_only_the_allowed_rows_finds_the_same(self, matrix):
        store, _ = matrix
        allowed = np.arange(30) % 3 == 0
        query = query_vector(11, 8)
        masked = store.search(query, 4, allowed, gather_below=0.0)
        gathered = store.search(query, 4, allowed, gather_below=1.0)
        assert masked.tolist() == gathered.tolist()

    def test_nothing_allowed_finds_nothing(self, matrix):
        store, _ = matrix
        assert store.search(query_vector(1, 8), 3, np.zeros(30, dtype=bool), 0.5).tolist() == []

    def test_no_mask_searches_every_row(self, matrix):
        store, _ = matrix
        query = query_vector(2, 8)
        expected = top_positions(store.scores(query), 5)
        assert store.search(query, 5, None, 0.5).tolist() == expected.tolist()


def test_an_int8_matrix_needs_its_scales():
    with pytest.raises(ValueError, match="scale"):
        VectorMatrix(np.zeros((2, 4), dtype=np.int8), None, 2)


class TestRealVectors:
    def test_real_vectors_are_written_as_they_are_given(self, tmp_path):
        source = np.random.default_rng(3).standard_normal((40, 6)).astype(np.float32)
        path = write_vectors(tmp_path, 40, 6, "float32", seed=1, chunk_rows=16, source=source)
        np.testing.assert_array_equal(np.load(path), source)

    def test_the_int8_scales_come_from_the_real_vectors(self, tmp_path):
        source = np.random.default_rng(4).standard_normal((40, 6)).astype(np.float32)
        write_vectors(tmp_path, 40, 6, "int8", seed=1, chunk_rows=16, source=source)
        scales = np.load(tmp_path / scales_name(6))
        np.testing.assert_allclose(scales, np.abs(source).max(axis=0) / 127, rtol=1e-6)

    def test_a_source_of_another_shape_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="source"):
            write_vectors(tmp_path, 40, 6, "float32", 1, 16, source=np.zeros((39, 6), np.float32))
