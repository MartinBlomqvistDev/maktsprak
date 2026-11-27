"""A package of the kit's corpus, and a search service over it, for the service's tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from search.answer.models import Candidate, Selected
from search.answer.selector import selection_output
from search_service.package import (
    STORE_DIR,
    AnswerConfig,
    Embedding,
    PackageManifest,
    Retrieval,
    SpeechTable,
    build_package,
)
from search_service.service import RateLimiter, SearchService, ServiceStore
from search_service.spike.inputs import CorpusFiles, read_passage_columns
from search_service.spike.settings import SearchSettings
from tests.search_service.spike_kit import kit_settings, write_index

RETRIEVAL = Retrieval(
    analyzer="stem-nostop",
    k1=1.2,
    b=0.9,
    target_words=200,
    rrf_k=10,
    depth=6,
    hits=3,
    gather_below=0.5,
    chunk_rows=5,
    vector_wait_s=30,
)
EMBEDDING = Embedding(
    name="kit",
    model="kit/model",
    revision="r1",
    provider="provider",
    quantization="bf16",
    dimension=8,
    query_prompt="Query:",
)
ANSWER = AnswerConfig(
    model="claude-opus-5-5",
    effort="low",
    max_tokens=8000,
    prompt_sha256="a" * 64,
    threshold=0.2,
    monthly_budget_usd=2.0,
    input_price=4.0,
    output_price=20.0,
)
HASHES = {"bm25_tuning": "b" * 64, "hybrid_tuning": "c" * 64, "answer_run": "d" * 64}


def unit_matrix(rows: int, dim: int, seed: int = 7) -> np.ndarray:
    """Random vectors of length one."""
    matrix = np.random.default_rng(seed).standard_normal((rows, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build_kit_package(
    corpus, root: Path, embedding: Embedding = EMBEDDING, answer: AnswerConfig = ANSWER
):
    """A package of the kit's corpus, with random unit vectors of the embedding's dimension.

    Returns:
        The package's directory, its manifest, the passage ids, the vectors and
        the corpus files.
    """
    files = CorpusFiles.under(corpus.search_dir, corpus.tuning)
    ids = list(read_passage_columns(files.passages).passage_ids)
    matrix = unit_matrix(len(ids), embedding.dimension)
    index = write_index(root / "index", ids, matrix)
    manifest = build_package(
        root / "package",
        name="kit",
        files=files,
        vectors_from=index,
        vectors=kit_settings().vectors,
        retrieval=RETRIEVAL,
        embedding=embedding,
        answer=answer,
        report_hashes=HASHES,
    )
    return root / "package", manifest, ids, matrix, files


class FakeSelector:
    """Quotes the first paragraph of the first candidate, or what it is told to answer."""

    def __init__(self, output: str | None = None, fail: bool = False, delay_s: float = 0.0) -> None:
        self.output = output
        self.fail = fail
        self.delay_s = delay_s
        self.asked: list[tuple[str, list[str]]] = []

    def select(self, question: str, candidates: Sequence[Candidate]) -> Selected:
        self.asked.append((question, [candidate.passage_id for candidate in candidates]))
        time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("the selector broke")
        output = self.output
        if output is None:
            output = selection_output(candidates[0].passage_id, 1, 1)
        return Selected(selector="claude", output=output, model="claude-opus-5-5")


def kit_service(
    directory: Path,
    manifest: PackageManifest,
    vectors: dict[str, np.ndarray],
    *,
    selector=None,
    delay_s: float = 0.0,
    wait_s: float | None = None,
    limits: dict[str, int] | None = None,
    answer_deadline_s: float = 45.0,
) -> SearchService:
    """A service over a kit package whose embedder knows the vectors given, by question."""
    if wait_s is not None:
        manifest = manifest.model_copy(
            update={"retrieval": manifest.retrieval.model_copy(update={"vector_wait_s": wait_s})}
        )
    retrieval = manifest.retrieval
    store = ServiceStore(
        directory / STORE_DIR,
        manifest.embedding.dimension,
        "float32",
        SearchSettings(
            depth=retrieval.depth,
            hits=retrieval.hits,
            rrf_k=retrieval.rrf_k,
            gather_below=retrieval.gather_below,
        ),
    )

    async def embed(text: str) -> np.ndarray:
        await asyncio.sleep(delay_s)
        if text not in vectors:
            raise RuntimeError("the API failed")
        return vectors[text]

    return SearchService(
        manifest,
        store,
        SpeechTable(directory, store.texts),
        embed,
        selector or FakeSelector(),
        RateLimiter(3600, limits or {"search": 100, "answer": 100}),
        question_chars=400,
        answer_deadline_s=answer_deadline_s,
    )
