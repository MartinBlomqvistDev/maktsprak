"""What the stores are built from: the passages, their text, and the BM25 the tuning chose."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME
from search.index.analyzer import ANALYZERS, Analyzer
from search.index.lexical import LEXICAL_DIR_NAME, Bm25Params, passage_texts

#: The tuning report that names the BM25 of every comparison.
TUNING_NAME = "bm25_tuning_dev.json"

_EPOCH = date(1970, 1, 1)


def day_number(day: date) -> int:
    """A date as days since 1970-01-01, as the stores keep it."""
    return (day - _EPOCH).days


def day_of(number: int) -> date:
    """The date of a day number."""
    return _EPOCH + timedelta(days=number)


class Bm25Choice(BaseModel):
    """The analyzer and parameters the tuning on the development split chose.

    Attributes:
        analyzer: How text becomes terms.
        params: ``k1`` and ``b``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyzer: Analyzer
    params: Bm25Params


def read_bm25_choice(path: Path) -> Bm25Choice:
    """Read the configuration the tuning chose from its report.

    Args:
        path: ``bm25_tuning_dev.json``.

    Returns:
        The choice.

    Raises:
        ValueError: If the report names an analyzer that does not exist.
    """
    chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
    for analyzer in ANALYZERS:
        if analyzer.name == chosen["analyzer"]:
            return Bm25Choice(analyzer=analyzer, params=Bm25Params(k1=chosen["k1"], b=chosen["b"]))
    raise ValueError(f"The tuning report names no known analyzer: {chosen['analyzer']!r}.")


@dataclass(frozen=True)
class CorpusFiles:
    """The files a store is built from.

    Attributes:
        speeches: ``anforanden.parquet``.
        passages: ``passages.parquet``.
        lexical: The directory of the chosen analyzer's BM25 index.
        choice: The BM25 the tuning chose.
    """

    speeches: Path
    passages: Path
    lexical: Path
    choice: Bm25Choice

    @classmethod
    def under(cls, search_dir: Path, tuning_report: Path) -> CorpusFiles:
        """The files under the search-data directory, with the choice in *tuning_report*."""
        choice = read_bm25_choice(tuning_report)
        return cls(
            speeches=search_dir / SPEECHES_NAME,
            passages=search_dir / PASSAGES_NAME,
            lexical=search_dir / LEXICAL_DIR_NAME / choice.analyzer.name,
            choice=choice,
        )


@dataclass(frozen=True)
class PassageColumns:
    """The fields of every passage that a store keeps beside its text.

    Attributes:
        passage_ids: The key of each passage, in index order.
        parties: The party of each passage's speech.
        days: The date of each passage's speech, in days since 1970-01-01, as int32.
        person_ids: The Riksdag's id of each speaker, or ``None``.
    """

    passage_ids: list[str]
    parties: list[str]
    days: np.ndarray
    person_ids: list[str | None]

    def __len__(self) -> int:
        return len(self.passage_ids)


def read_passage_columns(passages_path: Path, limit: int | None = None) -> PassageColumns:
    """Read the passages' keys and filter fields, in index order.

    Args:
        passages_path: ``passages.parquet``.
        limit: Keep only the first *limit* passages; ``None`` keeps all.

    Returns:
        The columns.
    """
    table = pq.read_table(passages_path, columns=["passage_id", "party", "date", "person_id"])
    if limit is not None:
        table = table.slice(0, limit)
    days = table.column("date").cast(pa.int32()).to_numpy().astype(np.int32)
    return PassageColumns(
        passage_ids=table.column("passage_id").to_pylist(),
        parties=table.column("party").to_pylist(),
        days=days,
        person_ids=table.column("person_id").to_pylist(),
    )


def iter_texts(files: CorpusFiles, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield every passage's id and text, in index order, as the BM25 index saw them.

    Args:
        files: The corpus files.
        limit: Stop after *limit* passages; ``None`` reads all.

    Yields:
        ``(passage_id, text)`` pairs.
    """
    texts = passage_texts(files.speeches, files.passages)
    yield from (texts if limit is None else islice(texts, limit))
