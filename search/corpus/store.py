"""Reading and writing the corpus on disk, as Parquet."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from search.models import Anforande, Passage

SPEECHES_NAME = "anforanden.parquet"
PASSAGES_NAME = "passages.parquet"
BATCH_ROWS = 2000

Record = TypeVar("Record", bound=BaseModel)

SPEECH_SCHEMA = pa.schema(
    [
        ("anf_id", pa.string()),
        ("dok_id", pa.string()),
        ("rm", pa.string()),
        ("dok_nummer", pa.int32()),
        ("anf_nummer", pa.int32()),
        ("date", pa.date32()),
        ("speaker_raw", pa.string()),
        ("person_id", pa.string()),
        ("party", pa.string()),
        ("is_reply", pa.bool_()),
        ("title", pa.string()),
        ("activity", pa.string()),
        ("rel_dok_id", pa.string()),
        ("layout", pa.string()),
        ("paragraphs", pa.list_(pa.string())),
        ("notes", pa.list_(pa.string())),
        ("text_sha256", pa.string()),
        ("url", pa.string()),
    ]
)


PASSAGE_SCHEMA = pa.schema(
    [
        ("passage_id", pa.string()),
        ("anf_id", pa.string()),
        ("first", pa.int32()),
        ("last", pa.int32()),
        ("words", pa.int32()),
        ("party", pa.string()),
        ("date", pa.date32()),
        ("person_id", pa.string()),
    ]
)


def _row(speech: Anforande) -> dict[str, object]:
    """Turn a speech into a row Arrow can take: lists where the model has tuples."""
    row = speech.model_dump()
    row["paragraphs"] = list(speech.paragraphs)
    row["notes"] = list(speech.notes)
    return row


def _write_parquet(
    path: Path,
    schema: pa.Schema,
    batches: Iterable[Sequence[Record]],
    to_row: Callable[[Record], dict[str, object]],
) -> int:
    """Write records to a Parquet file beside *path*, then move it into place.

    Whatever stops the write, including a destination that cannot be replaced,
    the partial file is removed and the file already at *path* is left as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    written = 0
    try:
        with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
            for batch in batches:
                if not batch:
                    continue
                rows = [to_row(record) for record in batch]
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                written += len(rows)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return written


def write_speeches(path: Path, batches: Iterable[Sequence[Anforande]]) -> int:
    """Write speeches to a Parquet file, one row group per batch.

    The file is written beside its destination and moved into place only when
    every batch has been written, so a build that stops halfway leaves the
    previous file untouched and no partial one behind. The same holds when the
    move itself fails, for example on Windows while another reader has the
    destination open.

    Args:
        path: The Parquet file to write.
        batches: The speeches, in the order they should be stored.

    Returns:
        The number of speeches written.
    """
    return _write_parquet(path, SPEECH_SCHEMA, batches, _row)


def iter_speeches(path: Path) -> Iterator[Anforande]:
    """Read speeches back from a Parquet file, checking each one on the way in.

    The file stays open until the iterator is exhausted or closed, so a caller
    that stops early and then rewrites the same path must close the iterator
    first.

    Args:
        path: The Parquet file to read.

    Yields:
        Each speech, validated as an :class:`Anforande`.

    Raises:
        pydantic.ValidationError: If a row breaks a rule of the model.
    """
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
        for row in batch.to_pylist():
            yield Anforande.model_validate(row)


def write_passages(path: Path, batches: Iterable[Sequence[Passage]]) -> int:
    """Write passages to a Parquet file, one row group per batch.

    Like :func:`write_speeches`, the file is written beside its destination and
    moved into place only when every batch has been written. A batch source that
    raises while it is being consumed, or a move that fails, therefore leaves the
    previous file untouched and no partial one behind.

    Args:
        path: The Parquet file to write.
        batches: The passages, in the order they should be stored.

    Returns:
        The number of passages written.
    """
    return _write_parquet(path, PASSAGE_SCHEMA, batches, lambda passage: passage.model_dump())


def iter_passages(path: Path) -> Iterator[Passage]:
    """Read passages back from a Parquet file, checking each one on the way in.

    The file stays open until the iterator is exhausted or closed, as for
    :func:`iter_speeches`.

    Args:
        path: The Parquet file to read.

    Yields:
        Each passage, validated as a :class:`Passage`.

    Raises:
        pydantic.ValidationError: If a row breaks a rule of the model.
    """
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
        for row in batch.to_pylist():
            yield Passage.model_validate(row)


def write_report(report: BaseModel, path: Path) -> None:
    """Write a report as indented JSON with Unix line endings, so it diffs cleanly.

    Args:
        report: The report to write.
        path: Where to write it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
