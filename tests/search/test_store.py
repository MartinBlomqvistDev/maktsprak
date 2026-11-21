"""Tests for search/corpus/store.py. Everything is written to a temporary directory."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, ValidationError

from search.corpus.store import (
    PASSAGE_SCHEMA,
    SPEECH_SCHEMA,
    _row,
    iter_passages,
    iter_speeches,
    write_passages,
    write_report,
    write_speeches,
)
from search.models import Anforande, Passage, deep_link, passage_id, text_sha256

PARAGRAPHS = ("Fru talman! Tack för svaret.", "Jag yrkar bifall till reservation 3.")


def speech(number: int = 7, **changes: object) -> Anforande:
    """Build a valid speech numbered *number*, with any field replaced by *changes*."""
    fields: dict[str, object] = {
        "anf_id": f"H209100-{number}",
        "dok_id": "H209100",
        "rm": "2014/15",
        "dok_nummer": 100,
        "anf_nummer": number,
        "date": date(2015, 5, 12),
        "speaker_raw": "Anna Andersson (S)",
        "person_id": "0123456789012",
        "party": "S",
        "is_reply": False,
        "title": "Svar på interpellation 2014/15:1 om skolan",
        "activity": "interpellationsdebatt",
        "rel_dok_id": "H21001",
        "layout": "html",
        "paragraphs": PARAGRAPHS,
        "notes": ("(Applåder)",),
        "text_sha256": text_sha256(PARAGRAPHS),
        "url": deep_link("H209100", number),
    }
    fields.update(changes)
    return Anforande(**fields)


def passage(number: int = 7, **changes: object) -> Passage:
    """Build a valid passage of speech *number*, with any field replaced by *changes*."""
    anf_id = f"H209100-{number}"
    fields: dict[str, object] = {
        "passage_id": passage_id(anf_id, 0, 1),
        "anf_id": anf_id,
        "first": 0,
        "last": 1,
        "words": 12,
        "party": "S",
        "date": date(2015, 5, 12),
        "person_id": "0123456789012",
    }
    fields.update(changes)
    return Passage(**fields)


def refuse_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every move of a file onto another fail, as on Windows with the target open."""

    def refuse(self: Path, target: object) -> Path:
        raise PermissionError("the file is open elsewhere")

    monkeypatch.setattr(Path, "replace", refuse)


def names(directory: Path) -> list[str]:
    """List the file names in *directory*."""
    return sorted(path.name for path in directory.iterdir())


class TestSpeeches:
    def test_round_trips_through_the_file(self, tmp_path):
        path = tmp_path / "speeches.parquet"
        speeches = [speech(1), speech(2), speech(3)]
        assert write_speeches(path, [speeches[:2], speeches[2:]]) == 3
        assert list(iter_speeches(path)) == speeches

    def test_a_speech_without_the_optional_fields_round_trips(self, tmp_path):
        path = tmp_path / "speeches.parquet"
        bare = speech(person_id=None, title=None, activity=None, rel_dok_id=None, notes=())
        write_speeches(path, [[bare]])
        (read,) = iter_speeches(path)
        assert read == bare
        assert (read.person_id, read.title, read.activity, read.rel_dok_id) == (None,) * 4

    def test_one_row_group_is_written_per_batch_and_empty_batches_are_skipped(self, tmp_path):
        path = tmp_path / "speeches.parquet"
        write_speeches(path, [[speech(1)], [], [speech(2), speech(3)]])
        assert pq.ParquetFile(path).num_row_groups == 2

    def test_no_speeches_give_an_empty_file(self, tmp_path):
        path = tmp_path / "speeches.parquet"
        assert write_speeches(path, []) == 0
        assert list(iter_speeches(path)) == []

    def test_the_same_speeches_give_the_same_bytes(self, tmp_path):
        first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
        write_speeches(first, [[speech(1), speech(2)]])
        write_speeches(second, [[speech(1), speech(2)]])
        assert first.read_bytes() == second.read_bytes()

    def test_the_directory_is_created(self, tmp_path):
        path = tmp_path / "deeper" / "speeches.parquet"
        write_speeches(path, [[speech()]])
        assert path.exists()

    def test_a_source_that_fails_leaves_the_old_file_and_no_partial_one(self, tmp_path):
        path = tmp_path / "speeches.parquet"
        write_speeches(path, [[speech(1)]])
        before = path.read_bytes()

        def failing() -> Iterator[list[Anforande]]:
            yield [speech(2)]
            raise RuntimeError("the source stopped")

        with pytest.raises(RuntimeError, match="the source stopped"):
            write_speeches(path, failing())
        assert path.read_bytes() == before
        assert names(tmp_path) == ["speeches.parquet"]

    def test_a_move_that_fails_leaves_the_old_file_and_no_partial_one(self, tmp_path, monkeypatch):
        path = tmp_path / "speeches.parquet"
        write_speeches(path, [[speech(1)]])
        before = path.read_bytes()

        refuse_replace(monkeypatch)
        with pytest.raises(PermissionError):
            write_speeches(path, [[speech(2)]])
        monkeypatch.undo()

        assert path.read_bytes() == before
        assert names(tmp_path) == ["speeches.parquet"]

    def test_a_first_write_that_fails_to_move_leaves_nothing(self, tmp_path, monkeypatch):
        refuse_replace(monkeypatch)
        with pytest.raises(PermissionError):
            write_speeches(tmp_path / "speeches.parquet", [[speech()]])
        monkeypatch.undo()
        assert names(tmp_path) == []

    def test_a_row_that_breaks_a_rule_is_refused_on_the_way_in(self, tmp_path):
        path = tmp_path / "speeches.parquet"
        row = _row(speech()) | {"party": "FP"}
        pq.write_table(pa.Table.from_pylist([row], schema=SPEECH_SCHEMA), path)
        with pytest.raises(ValidationError, match="not one of the eight parties"):
            list(iter_speeches(path))


class TestPassages:
    def test_round_trips_through_the_file(self, tmp_path):
        path = tmp_path / "passages.parquet"
        passages = [passage(1), passage(2, person_id=None), passage(3)]
        assert write_passages(path, [passages[:1], passages[1:]]) == 3
        assert list(iter_passages(path)) == passages

    def test_one_row_group_is_written_per_batch_and_empty_batches_are_skipped(self, tmp_path):
        path = tmp_path / "passages.parquet"
        write_passages(path, [[passage(1)], [], [passage(2)]])
        assert pq.ParquetFile(path).num_row_groups == 2

    def test_a_source_that_fails_leaves_the_old_file_and_no_partial_one(self, tmp_path):
        path = tmp_path / "passages.parquet"
        write_passages(path, [[passage(1)]])
        before = path.read_bytes()

        def failing() -> Iterator[list[Passage]]:
            yield [passage(2)]
            raise RuntimeError("the source stopped")

        with pytest.raises(RuntimeError, match="the source stopped"):
            write_passages(path, failing())
        assert path.read_bytes() == before
        assert names(tmp_path) == ["passages.parquet"]

    def test_a_move_that_fails_leaves_the_old_file_and_no_partial_one(self, tmp_path, monkeypatch):
        path = tmp_path / "passages.parquet"
        write_passages(path, [[passage(1)]])
        before = path.read_bytes()

        refuse_replace(monkeypatch)
        with pytest.raises(PermissionError):
            write_passages(path, [[passage(2)]])
        monkeypatch.undo()

        assert path.read_bytes() == before
        assert names(tmp_path) == ["passages.parquet"]

    def test_a_row_that_breaks_a_rule_is_refused_on_the_way_in(self, tmp_path):
        path = tmp_path / "passages.parquet"
        row = passage().model_dump() | {"party": "FP"}
        pq.write_table(pa.Table.from_pylist([row], schema=PASSAGE_SCHEMA), path)
        with pytest.raises(ValidationError, match="not one of the eight parties"):
            list(iter_passages(path))


class Report(BaseModel):
    """A small report for the writer to take."""

    total: int
    label: str


class TestReport:
    def test_is_indented_json_with_unix_line_endings(self, tmp_path):
        path = tmp_path / "reports" / "report.json"
        write_report(Report(total=3, label="Åre"), path)
        raw = path.read_bytes()
        assert b"\r" not in raw
        assert raw.endswith(b"}\n")
        assert raw.decode("utf-8").splitlines()[1] == '  "total": 3,'
