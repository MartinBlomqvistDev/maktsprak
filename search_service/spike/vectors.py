"""Synthetic vectors, and a matrix of them searched exactly with numpy.

No embedding model has been chosen yet, so the stores are filled with synthetic
vectors: unit vectors drawn from a seeded generator. They cost what real vectors
of the same shape and format cost to store, load and scan. They have none of the
structure of real embeddings, so nothing is read from them about what a search
finds.

A vector is stored in one of three formats: float32; float16; or int8 with one
scale per dimension, so that a component is its code times the scale of its
dimension.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from search_service.spike.settings import VectorFormat

#: Mixed into the seed of a query vector, so that no query shares a stream with a chunk.
QUERY_STREAM = 0x5157

#: The largest code of an int8 component; -127 is the smallest, so the codes are symmetric.
INT8_LIMIT = 127

_DTYPES: dict[str, type[np.generic]] = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
}


def _unit_rows(block: np.ndarray) -> np.ndarray:
    """Scale every row of *block* to length 1, in place, and return it."""
    block /= np.linalg.norm(block, axis=1, keepdims=True)
    return block


def synthetic_chunks(rows: int, dim: int, seed: int, chunk_rows: int) -> Iterator[np.ndarray]:
    """Yield the vectors of *rows* passages, a chunk at a time, as float32 unit vectors.

    Chunk ``n`` is drawn from a generator seeded with ``(seed, dim, n)``, so a
    chunk does not depend on the others.

    Args:
        rows: How many vectors in all.
        dim: The dimension.
        seed: The seed.
        chunk_rows: How many vectors one chunk holds; the last may hold fewer.

    Yields:
        Arrays of shape ``(chunk, dim)``.
    """
    for number, start in enumerate(range(0, rows, chunk_rows)):
        size = min(chunk_rows, rows - start)
        generator = np.random.default_rng([seed, dim, number])
        yield _unit_rows(generator.standard_normal((size, dim), dtype=np.float32))


def query_vector(seed: int, dim: int) -> np.ndarray:
    """The synthetic vector of one query, as a float32 unit vector.

    Args:
        seed: The query's own seed.
        dim: The dimension.

    Returns:
        An array of shape ``(dim,)``.
    """
    generator = np.random.default_rng([QUERY_STREAM, seed, dim])
    vector = generator.standard_normal(dim, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def int8_scales(rows: int, dim: int, seed: int, chunk_rows: int) -> np.ndarray:
    """The scale of each dimension that maps its largest absolute component to 127.

    Args:
        rows: How many vectors in all.
        dim: The dimension.
        seed: The seed of the vectors.
        chunk_rows: The size of their chunks.

    Returns:
        A float32 array of shape ``(dim,)``.
    """
    peak = np.zeros(dim, dtype=np.float32)
    for block in synthetic_chunks(rows, dim, seed, chunk_rows):
        np.maximum(peak, np.abs(block).max(axis=0), out=peak)
    return np.where(peak > 0, peak / INT8_LIMIT, 1.0).astype(np.float32)


def encode(block: np.ndarray, fmt: VectorFormat, scales: np.ndarray | None = None) -> np.ndarray:
    """Store a chunk of float32 vectors in one of the three formats.

    Args:
        block: The vectors.
        fmt: The format.
        scales: The scale of each dimension; needed for int8 only.

    Returns:
        The stored chunk.

    Raises:
        ValueError: If int8 is asked for without scales.
    """
    if fmt == "float32":
        return block
    if fmt == "float16":
        return block.astype(np.float16)
    if scales is None:
        raise ValueError("int8 needs the scale of each dimension.")
    codes = np.rint(block / scales)
    return np.clip(codes, -INT8_LIMIT, INT8_LIMIT).astype(np.int8)


def vectors_name(dim: int, fmt: VectorFormat) -> str:
    """The file a numpy store keeps its vectors of one shape and format in."""
    return f"vectors-{dim}-{fmt}.npy"


def scales_name(dim: int) -> str:
    """The file the int8 scales of one dimension are kept in."""
    return f"vectors-{dim}-int8-scales.npy"


def write_vectors(
    directory: Path, rows: int, dim: int, fmt: VectorFormat, seed: int, chunk_rows: int
) -> Path:
    """Write the synthetic vectors of *rows* passages to a ``.npy`` file in *directory*.

    Args:
        directory: Where to write; created if missing.
        rows: How many vectors.
        dim: The dimension.
        fmt: The format.
        seed: The seed.
        chunk_rows: The size of a chunk.

    Returns:
        The file written. For int8 the scales are written beside it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    scales = None
    if fmt == "int8":
        scales = int8_scales(rows, dim, seed, chunk_rows)
        np.save(directory / scales_name(dim), scales)
    path = directory / vectors_name(dim, fmt)
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=_DTYPES[fmt], shape=(rows, dim))
    start = 0
    for block in synthetic_chunks(rows, dim, seed, chunk_rows):
        matrix[start : start + len(block)] = encode(block, fmt, scales)
        start += len(block)
    matrix.flush()
    del matrix
    return path


def top_positions(scores: np.ndarray, k: int, positions: np.ndarray | None = None) -> np.ndarray:
    """The *k* best of *scores*, best first; equal scores keep their order.

    Args:
        scores: The scores to rank.
        k: How many to keep.
        positions: What each score stands for; ``None`` means its own index.

    Returns:
        The positions of the best scores, as int64.
    """
    if positions is None:
        positions = np.arange(len(scores))
    if len(scores) > k:
        kept = np.argpartition(-scores, k - 1)[:k]
        scores, positions = scores[kept], positions[kept]
    order = np.lexsort((positions, -scores))
    return positions[order].astype(np.int64)


class VectorMatrix:
    """Vectors of one shape and format, searched exactly by dot product.

    A float32 matrix is multiplied in one call. A float16 or int8 matrix, and
    any set of rows picked out, is turned into float32 a chunk at a time, so no
    copy holds more than one chunk.

    Attributes:
        matrix: The vectors, one row per passage, usually memory-mapped.
        scales: The scale of each dimension of an int8 matrix, else ``None``.
        chunk_rows: How many rows are turned into float32 at a time.
    """

    def __init__(self, matrix: np.ndarray, scales: np.ndarray | None, chunk_rows: int) -> None:
        if matrix.dtype == np.int8 and scales is None:
            raise ValueError("An int8 matrix needs the scale of each dimension.")
        self.matrix = matrix
        self.scales = scales
        self.chunk_rows = chunk_rows

    @classmethod
    def load(cls, directory: Path, dim: int, fmt: VectorFormat, chunk_rows: int) -> VectorMatrix:
        """Open the vectors written by :func:`write_vectors`, memory-mapped.

        Args:
            directory: The store's directory.
            dim: The dimension.
            fmt: The format.
            chunk_rows: How many rows are turned into float32 at a time.

        Returns:
            The matrix.
        """
        matrix = np.load(directory / vectors_name(dim, fmt), mmap_mode="r")
        scales = np.load(directory / scales_name(dim)) if fmt == "int8" else None
        return cls(matrix, scales, chunk_rows)

    def __len__(self) -> int:
        return len(self.matrix)

    def _weights(self, query: np.ndarray) -> np.ndarray:
        """The query, with an int8 matrix's scales folded in."""
        if self.scales is None:
            return query.astype(np.float32, copy=False)
        return (query * self.scales).astype(np.float32)

    def scores(self, query: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """The dot product of *query* with every vector, or with the vectors of *rows*.

        Args:
            query: A float32 vector of the matrix's dimension.
            rows: The rows to score, ascending; ``None`` scores all.

        Returns:
            One float32 score per row scored, in the order of the rows.
        """
        weights = self._weights(query)
        if rows is None and self.matrix.dtype == np.float32:
            return np.asarray(self.matrix @ weights, dtype=np.float32)
        count = len(self.matrix) if rows is None else len(rows)
        out = np.empty(count, dtype=np.float32)
        for start in range(0, count, self.chunk_rows):
            end = min(start + self.chunk_rows, count)
            block = self.matrix[start:end] if rows is None else self.matrix[rows[start:end]]
            np.matmul(block.astype(np.float32, copy=False), weights, out=out[start:end])
        return out

    def search(
        self, query: np.ndarray, k: int, allowed: np.ndarray | None, gather_below: float
    ) -> np.ndarray:
        """The *k* rows nearest *query* among those *allowed*, best first.

        Args:
            query: A float32 unit vector.
            k: How many rows to return.
            allowed: A mask over the rows, or ``None`` for all.
            gather_below: Below this share of rows allowed, only those rows are scored.

        Returns:
            The rows, as int64.
        """
        if allowed is None:
            return top_positions(self.scores(query), k)
        rows = np.flatnonzero(allowed)
        if len(rows) == 0:
            return rows.astype(np.int64)
        if len(rows) < gather_below * len(self.matrix):
            return top_positions(self.scores(query, rows), k, rows)
        scores = self.scores(query)
        return top_positions(scores[rows], k, rows)
