"""Exact vector search over the passages, and the index that step 07 builds from a run.

A vector index is a matrix of unit vectors, one row per passage, together with
the position of each row's passage in the passages file. A question's vector is
compared with every row by dot product, which for unit vectors is the cosine,
and the best rows are returned. The search is exact: every row is scored. An
approximate index is a question for the storage spike, which judges it against
this one. Questions are scored a block at a time, and the matrix product of a
block can round a score differently in its last digits than a single question's
product would.

Step 07 turns a whole run of step 06 into an index. The run's shards, checked
against their hashes, become one matrix on disk, with a manifest that names the
run, the embedder and the passages. Loading the index checks the files against
that manifest again, so an index that no longer matches is never searched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.index.embed import (
    RUN_MANIFEST,
    EmbedderSpec,
    InputsManifest,
    ids_sha256,
    now,
    read_inputs,
    read_run_manifest,
    read_shards,
    save_array,
    write_json,
)
from search.index.lexical import file_sha256

VECTORS_NAME = "vectors.npy"
POSITIONS_NAME = "positions.npy"
INDEX_MANIFEST = "index.json"

#: The directory under the search data that holds input sets, runs and indexes.
VECTORS_DIR = "vectors"

#: How many questions are scored against the whole matrix at once.
QUERY_BLOCK = 32


def inputs_path(search_dir: Path, inputs: str) -> Path:
    """The directory of one input set, for example ``subset-200``."""
    return search_dir / VECTORS_DIR / "inputs" / inputs


def run_path(search_dir: Path, inputs: str, candidate: str) -> Path:
    """The directory of one candidate's run of step 06 over one input set."""
    return search_dir / VECTORS_DIR / "runs" / inputs / candidate


def index_path(search_dir: Path, inputs: str, candidate: str) -> Path:
    """The directory of one candidate's index of step 07 over one input set."""
    return search_dir / VECTORS_DIR / "index" / inputs / candidate


def passages_path(search_dir: Path, target_words: int, corpus_words: int) -> Path:
    """The passages file of one passage size.

    Args:
        search_dir: The search data directory.
        target_words: The passage size.
        corpus_words: The corpus's own size, whose file has the plain name.

    Returns:
        ``passages.parquet`` for the corpus's own size, ``passages-<size>.parquet`` otherwise.
    """
    if target_words == corpus_words:
        return search_dir / "passages.parquet"
    return search_dir / f"passages-{target_words}.parquet"


class VectorIndexManifest(BaseModel):
    """What a vector index on disk was built from.

    Attributes:
        embedder: What produced the vectors.
        inputs: The input set the run embedded.
        run_manifest_sha256: The hash of the run's manifest file.
        rows: How many passages the index holds.
        dimension: The size of each vector.
        vectors_sha256: The hash of the matrix file.
        positions_sha256: The hash of the positions file.
        built: When the index was built.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedder: EmbedderSpec
    inputs: InputsManifest
    run_manifest_sha256: str
    rows: int
    dimension: int
    vectors_sha256: str
    positions_sha256: str
    built: datetime


@dataclass(frozen=True)
class VectorHit:
    """One passage a vector search returned.

    Attributes:
        passage: The passage's position in the passages file.
        score: Its cosine with the question.
    """

    passage: int
    score: float


class VectorIndex:
    """Unit vectors of passages, searched exactly.

    Attributes:
        vectors: One row of ``float32`` per passage, each of length 1.
        positions: Each row's passage, as its position in the passages file,
            in ascending order.
    """

    def __init__(self, vectors: np.ndarray, positions: np.ndarray) -> None:
        if vectors.ndim != 2 or positions.ndim != 1 or len(vectors) != len(positions):
            raise ValueError("The vectors and the positions of an index do not fit together.")
        if len(positions) > 1 and not (np.diff(positions) > 0).all():
            raise ValueError("The positions of an index must be ascending and distinct.")
        self.vectors = vectors
        self.positions = np.asarray(positions, dtype=np.int64)

    @property
    def rows(self) -> int:
        """How many passages the index holds."""
        return len(self.positions)

    @property
    def dimension(self) -> int:
        """The size of each vector."""
        return int(self.vectors.shape[1])

    def _top(self, scores: np.ndarray, limit: int, allowed: np.ndarray | None) -> list[VectorHit]:
        """The best rows of one question's scores, best first, under a mask over the passages."""
        candidates = np.arange(self.rows)
        if allowed is not None:
            candidates = np.flatnonzero(allowed[self.positions])
        if len(candidates) > limit:
            kept = np.argpartition(-scores[candidates], limit - 1)[:limit]
            candidates = candidates[kept]
        order = np.lexsort((self.positions[candidates], -scores[candidates]))
        return [
            VectorHit(
                passage=int(self.positions[candidates[i]]), score=float(scores[candidates[i]])
            )
            for i in order
        ]

    def search(
        self, query: np.ndarray, limit: int, allowed: np.ndarray | None = None
    ) -> list[VectorHit]:
        """Rank the passages for one question's vector, best first.

        Two passages with the same score are ranked in passage order.

        Args:
            query: The question's unit vector.
            limit: The most passages to return.
            allowed: A mask over the passages file; only passages marked ``True``
                can be returned. ``None`` allows every passage in the index.

        Returns:
            The hits, best first.
        """
        return self.search_many(query[np.newaxis, :], limit, [allowed])[0]

    def search_many(
        self, queries: np.ndarray, limit: int, allowed: Sequence[np.ndarray | None]
    ) -> list[list[VectorHit]]:
        """Rank the passages for many questions at once, a block of questions at a time.

        Args:
            queries: One unit vector per question.
            limit: The most passages to return for each.
            allowed: One mask per question, or ``None`` for no filter, as in :meth:`search`.

        Returns:
            Each question's hits, in the order of the questions.

        Raises:
            ValueError: If the vectors' size differs from the index's, or the
                number of masks from the number of questions.
        """
        data = np.asarray(queries, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dimension or len(allowed) != len(data):
            raise ValueError("The questions do not fit the index or their masks.")
        results: list[list[VectorHit]] = []
        for start in range(0, len(data), QUERY_BLOCK):
            block = data[start : start + QUERY_BLOCK] @ self.vectors.T
            for offset, scores in enumerate(block):
                results.append(self._top(scores, limit, allowed[start + offset]))
        return results


def build_vector_index(run_dir: Path, inputs_dir: Path, out_dir: Path) -> VectorIndexManifest:
    """Step 07: turn a whole run into an index on disk.

    Args:
        run_dir: The run of step 06.
        inputs_dir: The input set the run embedded.
        out_dir: Where to write the index; created if missing.

    Returns:
        The index's manifest.

    Raises:
        ValueError: If the run is not whole, does not match its input set, or a
            vector is not of length 1.
    """
    manifest = read_run_manifest(run_dir)
    inputs = read_inputs(inputs_dir)
    if manifest.inputs.passage_ids_sha256 != inputs.manifest.passage_ids_sha256:
        raise ValueError(f"The run in {run_dir} was not made from the inputs in {inputs_dir}.")
    vectors = read_shards(run_dir, manifest)
    norms = np.linalg.norm(vectors, axis=1)
    if len(vectors) and np.abs(norms - 1).max() > 1e-3:
        raise ValueError(f"The run in {run_dir} holds vectors that are not of length 1.")
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_sha256 = save_array(out_dir / VECTORS_NAME, vectors.astype(np.float32))
    positions_sha256 = save_array(out_dir / POSITIONS_NAME, inputs.positions.astype(np.int64))
    index_manifest = VectorIndexManifest(
        embedder=manifest.embedder,
        inputs=manifest.inputs,
        run_manifest_sha256=file_sha256(run_dir / RUN_MANIFEST),
        rows=len(vectors),
        dimension=manifest.embedder.dimension,
        vectors_sha256=vectors_sha256,
        positions_sha256=positions_sha256,
        built=now(),
    )
    write_json(out_dir / INDEX_MANIFEST, index_manifest)
    return index_manifest


def load_vector_index(
    index_dir: Path, passage_ids: Sequence[str] | None = None
) -> tuple[VectorIndex, VectorIndexManifest]:
    """Read an index written by :func:`build_vector_index`; the matrix is memory-mapped.

    Args:
        index_dir: The index's directory.
        passage_ids: The ids of the passages file the index is to be searched
            against, in its order. When given, the index's passages must be the
            ones at its positions in that file.

    Returns:
        The index and its manifest.

    Raises:
        ValueError: If a file does not match the manifest, or the passages do
            not match the passages file.
    """
    manifest = VectorIndexManifest.model_validate_json(
        (index_dir / INDEX_MANIFEST).read_text(encoding="utf-8")
    )
    for name, recorded in (
        (VECTORS_NAME, manifest.vectors_sha256),
        (POSITIONS_NAME, manifest.positions_sha256),
    ):
        if file_sha256(index_dir / name) != recorded:
            raise ValueError(f"{name} in {index_dir} does not match the index's manifest.")
    vectors = np.load(index_dir / VECTORS_NAME, mmap_mode="r", allow_pickle=False)
    positions = np.load(index_dir / POSITIONS_NAME, allow_pickle=False)
    if vectors.shape != (manifest.rows, manifest.dimension) or len(positions) != manifest.rows:
        raise ValueError(f"The index in {index_dir} does not have the shape of its manifest.")
    if passage_ids is not None:
        if len(positions) and int(positions.max()) >= len(passage_ids):
            raise ValueError(f"The index in {index_dir} reaches past the passages file.")
        held = [passage_ids[int(position)] for position in positions]
        if ids_sha256(held) != manifest.inputs.passage_ids_sha256:
            raise ValueError(f"The index in {index_dir} holds other passages than the file.")
    return VectorIndex(vectors, positions), manifest
