"""A BM25 index over the passages, built with numpy.

The index is an inverted list: for every term, the passages that hold it and
how often. Scores are computed at query time from those counts, so ``k1`` and
``b`` can be tuned without building the index again.

A passage's score for a query is the sum, over the distinct terms of the query,
of ``idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``, where
``tf`` is the term's count in the passage, ``dl`` the passage's length in terms
and ``avgdl`` the mean length. ``idf(t)`` is ``ln(1 + (N - df + 0.5) / (df + 0.5))``,
the form Lucene uses, which never goes below zero. A term repeated in the query
counts once.

Passages are numbered in the order of ``passages.parquet``. That order is part
of the index, and the manifest records the hash of the file it came from.
"""

from __future__ import annotations

import hashlib
import json
import math
from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from search.index.analyzer import Analyzer, stem_all, stopwords, tokenize

LEXICAL_DIR_NAME = "lexical"
MANIFEST_NAME = "manifest.json"
TERMS_NAME = "terms.json"
ARRAY_NAMES = ("term_ptr", "doc_ids", "tfs", "doc_lengths")

#: How many passages a search ranks before it is cut to the hits asked for.
#: Enough for 50 distinct speeches even when a speech has many passages among them.
PASSAGE_WINDOW = 1000

_READ_BATCH_ROWS = 2000


class Bm25Params(BaseModel):
    """The two free parameters of BM25.

    Attributes:
        k1: How quickly the weight of a term saturates with its count.
        b: How strongly a passage's length is normalised, from 0 (not at all) to 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    k1: float = Field(ge=0)
    b: float = Field(ge=0, le=1)


#: Lucene's defaults, the starting point of the tuning.
DEFAULT_PARAMS = Bm25Params(k1=1.2, b=0.75)


def file_sha256(path: Path) -> str:
    """Hash a file in chunks.

    Args:
        path: The file.

    Returns:
        Its SHA-256 in hex.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def passage_texts(speeches_path: Path, passages_path: Path) -> Iterator[tuple[str, str]]:
    """Yield every passage's id and text, in the order of the passages file.

    A passage's text is its paragraphs joined by a line feed. The two files are
    read side by side, which works because the passages were packed speech by
    speech in the order of the speeches file; a passage whose speech is not the
    next one in that file stops the read.

    Args:
        speeches_path: ``anforanden.parquet``.
        passages_path: ``passages.parquet``.

    Yields:
        ``(passage_id, text)`` pairs.

    Raises:
        ValueError: If the two files are not in the same order.
    """
    speeches = (
        row
        for batch in pq.ParquetFile(speeches_path).iter_batches(
            batch_size=_READ_BATCH_ROWS, columns=["anf_id", "paragraphs"]
        )
        for row in batch.to_pylist()
    )
    current_id: str | None = None
    current_paragraphs: list[str] = []
    for batch in pq.ParquetFile(passages_path).iter_batches(
        batch_size=_READ_BATCH_ROWS * 10, columns=["passage_id", "anf_id", "first", "last"]
    ):
        for row in batch.to_pylist():
            while row["anf_id"] != current_id:
                speech = next(speeches, None)
                if speech is None:
                    raise ValueError(
                        f"Passage {row['passage_id']} has no speech in the speeches file, "
                        "or the two files are not in the same order."
                    )
                current_id, current_paragraphs = speech["anf_id"], speech["paragraphs"]
            yield row["passage_id"], "\n".join(current_paragraphs[row["first"] : row["last"] + 1])


@dataclass
class TokenizedCorpus:
    """Every passage as a run of token ids, before any analyzer is applied.

    Attributes:
        passage_ids: The passages, in index order.
        vocabulary: The distinct tokens, lower-case, in order of first appearance.
        tokens: The token ids of all passages, one after the other.
        offsets: Where each passage's tokens start in ``tokens``, and the end of the last.
    """

    passage_ids: list[str]
    vocabulary: list[str]
    tokens: np.ndarray
    offsets: np.ndarray

    @property
    def passages(self) -> int:
        """How many passages there are."""
        return len(self.passage_ids)


def tokenize_corpus(texts: Iterable[tuple[str, str]]) -> TokenizedCorpus:
    """Split every passage into tokens and number the distinct tokens.

    Args:
        texts: ``(passage_id, text)`` pairs, in index order.

    Returns:
        The corpus as token ids.
    """
    vocabulary: dict[str, int] = {}
    tokens = array("i")
    offsets = array("q", [0])
    passage_ids: list[str] = []
    for passage, text in texts:
        tokens.extend([vocabulary.setdefault(token, len(vocabulary)) for token in tokenize(text)])
        offsets.append(len(tokens))
        passage_ids.append(passage)
    return TokenizedCorpus(
        passage_ids=passage_ids,
        vocabulary=list(vocabulary),
        tokens=np.frombuffer(tokens, dtype=np.int32).copy(),
        offsets=np.frombuffer(offsets, dtype=np.int64).copy(),
    )


class IndexManifest(BaseModel):
    """What an index on disk was built from, checked when it is loaded.

    Attributes:
        analyzer: How text became terms.
        passages: How many passages the index holds.
        passage_ids_sha256: The hash of the passage ids in index order, one per line.
        passages_sha256: The hash of the passages file the index was built from.
        speeches_sha256: The hash of the speeches file the text came from.
        terms: How many distinct terms the index holds.
        postings: How many (term, passage) pairs it holds.
        tokens: How many terms the passages hold in all, repeats included.
        mean_passage_terms: The mean length of a passage, in terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyzer: Analyzer
    passages: int
    passage_ids_sha256: str
    passages_sha256: str
    speeches_sha256: str
    terms: int
    postings: int
    tokens: int
    mean_passage_terms: float


def passage_ids_sha256(passage_ids: Sequence[str]) -> str:
    """Hash passage ids in order, one per line, as the manifest records them."""
    return hashlib.sha256("\n".join(passage_ids).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Hit:
    """One passage a search returned.

    Attributes:
        passage: The passage's number in index order.
        score: Its BM25 score.
    """

    passage: int
    score: float


class LexicalIndex:
    """A BM25 index: the terms, their postings and the length of every passage.

    Attributes:
        analyzer: How text became terms, for passages and queries alike.
        terms: The distinct terms, by id.
        term_ptr: Where each term's postings start in ``doc_ids`` and ``tfs``,
            and the end of the last.
        doc_ids: The passage of each posting, grouped by term, ascending within a term.
        tfs: The count of the term in that passage.
        doc_lengths: The length of every passage, in terms.
    """

    def __init__(
        self,
        analyzer: Analyzer,
        terms: Sequence[str],
        term_ptr: np.ndarray,
        doc_ids: np.ndarray,
        tfs: np.ndarray,
        doc_lengths: np.ndarray,
    ) -> None:
        if len(term_ptr) != len(terms) + 1 or len(doc_ids) != len(tfs):
            raise ValueError("The index arrays do not fit together.")
        self.analyzer = analyzer
        self.terms = list(terms)
        self.term_ids = {term: number for number, term in enumerate(self.terms)}
        self.term_ptr = term_ptr
        self.doc_ids = doc_ids
        self.tfs = tfs
        self.doc_lengths = doc_lengths
        total = float(doc_lengths.sum())
        self.mean_length = total / len(doc_lengths) if len(doc_lengths) else 0.0
        self._norms: dict[Bm25Params, np.ndarray] = {}

    @property
    def passages(self) -> int:
        """How many passages the index holds."""
        return len(self.doc_lengths)

    def document_frequency(self, term: str) -> int:
        """Count the passages that hold *term*; 0 for a term the index lacks."""
        number = self.term_ids.get(term)
        if number is None:
            return 0
        return int(self.term_ptr[number + 1] - self.term_ptr[number])

    def idf(self, term: str) -> float:
        """The inverse document frequency of *term*, in Lucene's form."""
        df = self.document_frequency(term)
        return math.log(1 + (self.passages - df + 0.5) / (df + 0.5))

    def _norm(self, params: Bm25Params) -> np.ndarray:
        """``k1 * (1 - b + b * dl / avgdl)`` for every passage, computed once per parameters."""
        norm = self._norms.get(params)
        if norm is None:
            relative = self.doc_lengths / self.mean_length if self.mean_length else 0.0
            norm = params.k1 * (1 - params.b + params.b * relative)
            self._norms[params] = norm
        return norm

    def scores(self, query: str, params: Bm25Params) -> np.ndarray:
        """Score every passage for *query*.

        Args:
            query: The question, as written; the index's analyzer turns it into terms.
            params: ``k1`` and ``b``.

        Returns:
            One score per passage, in index order. A passage that holds none of
            the query's terms scores 0.
        """
        scores = np.zeros(self.passages, dtype=np.float64)
        norm = self._norm(params)
        for term in dict.fromkeys(self.analyzer.terms(query)):
            number = self.term_ids.get(term)
            if number is None:
                continue
            start, end = int(self.term_ptr[number]), int(self.term_ptr[number + 1])
            docs = self.doc_ids[start:end]
            tf = self.tfs[start:end].astype(np.float64)
            weight = math.log(1 + (self.passages - (end - start) + 0.5) / (end - start + 0.5))
            scores[docs] += weight * tf * (params.k1 + 1) / (tf + norm[docs])
        return scores

    def search(
        self,
        query: str,
        params: Bm25Params,
        limit: int = PASSAGE_WINDOW,
        allowed: np.ndarray | None = None,
    ) -> list[Hit]:
        """Rank the passages for *query*, best first.

        Passages that hold no term of the query are never returned. Two
        passages with the same score are ranked in index order.

        Args:
            query: The question, as written.
            params: ``k1`` and ``b``.
            limit: The most passages to return.
            allowed: A mask over the passages; only those marked ``True`` can be
                returned. ``None`` allows every passage.

        Returns:
            The hits, best first.
        """
        scores = self.scores(query, params)
        if allowed is not None:
            scores = np.where(allowed, scores, 0.0)
        candidates = np.flatnonzero(scores > 0)
        if len(candidates) > limit:
            kept = np.argpartition(-scores[candidates], limit - 1)[:limit]
            candidates = candidates[kept]
        order = np.lexsort((candidates, -scores[candidates]))
        return [Hit(passage=int(candidates[i]), score=float(scores[candidates[i]])) for i in order]

    def save(self, directory: Path, manifest: IndexManifest) -> None:
        """Write the index and its manifest to *directory*.

        Args:
            directory: Where to write; created if missing.
            manifest: What the index was built from.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for name in ARRAY_NAMES:
            np.save(directory / f"{name}.npy", getattr(self, name))
        (directory / TERMS_NAME).write_text(
            json.dumps(self.terms, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        (directory / MANIFEST_NAME).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, directory: Path) -> tuple[LexicalIndex, IndexManifest]:
        """Read an index written by :meth:`save`; the large arrays are memory-mapped.

        Args:
            directory: The index's directory.

        Returns:
            The index and its manifest.

        Raises:
            ValueError: If the arrays do not match the manifest.
        """
        manifest = IndexManifest.model_validate_json(
            (directory / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}
        terms = json.loads((directory / TERMS_NAME).read_text(encoding="utf-8"))
        index = cls(manifest.analyzer, terms, **arrays)
        if (
            index.passages != manifest.passages
            or len(index.terms) != manifest.terms
            or len(index.doc_ids) != manifest.postings
        ):
            raise ValueError(f"The index in {directory} does not match its manifest.")
        return index, manifest


def build_index(corpus: TokenizedCorpus, analyzer: Analyzer) -> LexicalIndex:
    """Build the inverted index of a tokenized corpus under one analyzer.

    Args:
        corpus: The passages as token ids.
        analyzer: Whether stop words are dropped and tokens stemmed.

    Returns:
        The index.

    Raises:
        ValueError: If a term occurs more often in one passage than the counts can hold.
    """
    stems = stem_all(corpus.vocabulary) if analyzer.stem else corpus.vocabulary
    dropped = stopwords() if analyzer.stopwords else frozenset()
    term_ids: dict[str, int] = {}
    token_to_term = np.empty(len(corpus.vocabulary), dtype=np.int32)
    for number, (token, term) in enumerate(zip(corpus.vocabulary, stems, strict=True)):
        token_to_term[number] = -1 if token in dropped else term_ids.setdefault(term, len(term_ids))

    passages = corpus.passages
    lengths = np.diff(corpus.offsets)
    docs = np.repeat(np.arange(passages, dtype=np.int32), lengths)
    mapped = token_to_term[corpus.tokens]
    kept = mapped >= 0
    docs, mapped = docs[kept], mapped[kept]
    doc_lengths = np.bincount(docs, minlength=passages).astype(np.int32)

    keys = mapped.astype(np.int64) * passages + docs
    del docs, mapped, kept
    pairs, counts = np.unique(keys, return_counts=True)
    del keys
    if len(counts) and counts.max() > np.iinfo(np.uint16).max:
        raise ValueError("A term occurs more often in one passage than a 16-bit count holds.")
    posting_terms = pairs // passages
    term_ptr = np.zeros(len(term_ids) + 1, dtype=np.int64)
    np.cumsum(np.bincount(posting_terms, minlength=len(term_ids)), out=term_ptr[1:])
    terms = [""] * len(term_ids)
    for term, number in term_ids.items():
        terms[number] = term
    return LexicalIndex(
        analyzer=analyzer,
        terms=terms,
        term_ptr=term_ptr,
        doc_ids=(pairs % passages).astype(np.int32),
        tfs=counts.astype(np.uint16),
        doc_lengths=doc_lengths,
    )


def lexical_dir(search_dir: Path, analyzer: Analyzer) -> Path:
    """The directory of one analyzer's index under the search-data directory."""
    return search_dir / LEXICAL_DIR_NAME / analyzer.name
