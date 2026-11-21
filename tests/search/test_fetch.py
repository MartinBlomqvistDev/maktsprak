"""Tests for search/corpus/fetch.py. Nothing here reaches the network."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from search.corpus.fetch import (
    FetchedFile,
    Manifest,
    dataset_url,
    fetch_datasets,
    load_manifest,
    sessions_between,
    verify_raw_files,
    write_manifest,
)

TEMPLATE = "https://example.org/dataset/anforande-{session}.json.zip"
URL_FIRST = "https://example.org/dataset/anforande-200203.json.zip"
URL_SECOND = "https://example.org/dataset/anforande-200304.json.zip"
SESSIONS = ["2002/03", "2003/04"]
TODAY = date(2026, 11, 20)


def zip_bytes(speech_text: str) -> bytes:
    """Build a small zip in memory, shaped like a dataset file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("H00001-1.json", json.dumps({"anforande": {"text": speech_text}}))
    return buffer.getvalue()


PAYLOADS = {
    URL_FIRST: zip_bytes("first session"),
    URL_SECOND: zip_bytes("second session, longer"),
}


class DroppedConnection(httpx.AsyncByteStream):
    """A body that starts to arrive and then breaks off."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"PK partial bytes"
        raise httpx.ReadError("connection dropped")


class FakeHost:
    """A stand-in for the Riksdag's file host that counts what it is asked for."""

    def __init__(
        self,
        payloads: dict[str, bytes],
        failures_first: int = 0,
        dropping: frozenset[str] = frozenset(),
        failure_status: int = 503,
    ) -> None:
        self.payloads = payloads
        self.failures_first = failures_first
        self.failure_status = failure_status
        self.dropping = dropping
        self.requests: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests[url] += 1
        if self.requests[url] <= self.failures_first:
            return httpx.Response(self.failure_status)
        if url in self.dropping:
            return httpx.Response(200, stream=DroppedConnection())
        if url not in self.payloads:
            return httpx.Response(404)
        return httpx.Response(200, content=self.payloads[url])


def fetch(host: FakeHost, raw_dir: Path, known: Manifest, **options: object) -> Manifest:
    """Run one fetch against the fake host, without pauses unless ``backoff_seconds`` is given."""
    options.setdefault("backoff_seconds", 0)

    async def run() -> Manifest:
        async with httpx.AsyncClient(transport=httpx.MockTransport(host)) as client:
            return await fetch_datasets(
                options.pop("sessions", SESSIONS),
                url_template=TEMPLATE,
                raw_dir=raw_dir,
                known=known,
                client=client,
                today=TODAY,
                **options,
            )

    return asyncio.run(run())


def entry(session: str, file: str | None = None, url: str | None = None) -> FetchedFile:
    """Build a manifest entry for *session*."""
    compact = session.replace("/", "")
    name = f"anforande-{compact}.json.zip" if file is None else file
    return FetchedFile(
        session=session,
        file=name,
        url=f"https://example.org/dataset/{name}" if url is None else url,
        bytes=10,
        sha256="0" * 64,
        fetched=TODAY,
    )


class TestSessions:
    def test_lists_sessions_in_order(self):
        assert sessions_between("2008/09", "2010/11") == ["2008/09", "2009/10", "2010/11"]

    def test_one_session_is_a_valid_range(self):
        assert sessions_between("2026/27", "2026/27") == ["2026/27"]

    @pytest.mark.parametrize("label", ["2002-03", "2002/04", "02/03", "2002/2003", "1999/00", ""])
    def test_a_malformed_label_raises(self, label):
        with pytest.raises(ValueError):
            sessions_between(label, "2026/27")

    def test_a_reversed_range_raises(self):
        with pytest.raises(ValueError):
            sessions_between("2010/11", "2008/09")

    def test_the_file_is_named_after_the_label_without_its_slash(self):
        assert dataset_url(TEMPLATE, "2002/03") == (
            "https://example.org/dataset/anforande-200203.json.zip"
        )


class TestFetch:
    def test_writes_each_file_and_records_it(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())

        assert [entry.session for entry in manifest.files] == SESSIONS
        for item in manifest.files:
            payload = PAYLOADS[item.url]
            assert (tmp_path / item.file).read_bytes() == payload
            assert item.bytes == len(payload)
            assert item.sha256 == hashlib.sha256(payload).hexdigest()
            assert item.fetched == TODAY

    def test_leaves_no_partial_files_behind(self, tmp_path):
        fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "anforande-200203.json.zip",
            "anforande-200304.json.zip",
        ]

    def test_a_second_run_fetches_nothing(self, tmp_path):
        host = FakeHost(PAYLOADS)
        first = fetch(host, tmp_path, Manifest())
        second = fetch(host, tmp_path, first)

        assert second == first
        assert set(host.requests.values()) == {1}

    def test_force_fetches_again(self, tmp_path):
        host = FakeHost(PAYLOADS)
        first = fetch(host, tmp_path, Manifest())
        fetch(host, tmp_path, first, force=True)
        assert set(host.requests.values()) == {2}

    def test_a_file_the_manifest_does_not_vouch_for_fails_the_run(self, tmp_path):
        (tmp_path / "anforande-200203.json.zip").write_bytes(b"of unknown origin")
        with pytest.raises(ValueError, match="does not match the manifest"):
            fetch(FakeHost(PAYLOADS), tmp_path, Manifest())

    def test_a_file_the_manifest_does_not_vouch_for_does_not_stop_the_other_sessions(
        self, tmp_path
    ):
        (tmp_path / "anforande-200203.json.zip").write_bytes(b"of unknown origin")
        host = FakeHost(PAYLOADS)
        with pytest.raises(ValueError, match="does not match the manifest"):
            fetch(host, tmp_path, Manifest())
        assert (tmp_path / "anforande-200203.json.zip").read_bytes() == b"of unknown origin"
        assert (tmp_path / "anforande-200304.json.zip").read_bytes() == PAYLOADS[URL_SECOND]
        assert list(host.requests) == [URL_SECOND]

    def test_a_changed_file_fails_the_run(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        (tmp_path / "anforande-200203.json.zip").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="does not match the manifest"):
            fetch(FakeHost(PAYLOADS), tmp_path, manifest)

    def test_entries_for_other_sessions_are_kept(self, tmp_path):
        host = FakeHost(PAYLOADS)
        first = fetch(host, tmp_path, Manifest(), sessions=["2003/04"])
        both = fetch(host, tmp_path, first, sessions=["2002/03"])
        assert [item.session for item in both.files] == SESSIONS

    def test_a_server_error_is_retried(self, tmp_path):
        host = FakeHost(PAYLOADS, failures_first=2)
        manifest = fetch(host, tmp_path, Manifest())
        assert len(manifest.files) == 2
        assert set(host.requests.values()) == {3}

    def test_a_rate_limit_answer_is_retried(self, tmp_path):
        host = FakeHost(PAYLOADS, failures_first=1, failure_status=429)
        manifest = fetch(host, tmp_path, Manifest())
        assert len(manifest.files) == 2
        assert set(host.requests.values()) == {2}

    def test_the_pause_grows_with_each_failed_attempt(self, tmp_path, monkeypatch):
        pauses: list[float] = []

        async def record(seconds: float) -> None:
            pauses.append(seconds)

        monkeypatch.setattr("search.corpus.fetch.asyncio.sleep", record)
        host = FakeHost(PAYLOADS, failures_first=2)
        fetch(host, tmp_path, Manifest(), sessions=["2002/03"], backoff_seconds=3)
        assert pauses == [3, 6]

    def test_gives_up_after_the_last_attempt(self, tmp_path):
        host = FakeHost(PAYLOADS, failures_first=5)
        with pytest.raises(httpx.HTTPStatusError):
            fetch(host, tmp_path, Manifest(), attempts=3)

    def test_a_missing_dataset_is_not_retried(self, tmp_path):
        host = FakeHost({})
        with pytest.raises(httpx.HTTPStatusError):
            fetch(host, tmp_path, Manifest(), sessions=["2002/03"])
        assert set(host.requests.values()) == {1}

    def test_a_failed_session_does_not_lose_the_ones_that_succeeded(self, tmp_path):
        manifest_path = tmp_path / "reports" / "manifest.json"
        raw = tmp_path / "raw"
        host = FakeHost({URL_FIRST: PAYLOADS[URL_FIRST]})
        with pytest.raises(httpx.HTTPStatusError):
            fetch(
                host,
                raw,
                Manifest(),
                on_fetched=lambda manifest: write_manifest(manifest, manifest_path),
            )

        assert [item.session for item in load_manifest(manifest_path).files] == ["2002/03"]
        assert [path.name for path in raw.iterdir()] == ["anforande-200203.json.zip"]

        later = FakeHost(PAYLOADS)
        fetch(later, raw, load_manifest(manifest_path))
        assert list(later.requests) == [URL_SECOND]

    def test_reports_the_manifest_after_each_file(self, tmp_path):
        seen: list[Manifest] = []
        fetch(FakeHost(PAYLOADS), tmp_path, Manifest(), on_fetched=seen.append)
        assert [len(manifest.files) for manifest in seen] == [1, 2]
        assert [item.session for item in seen[-1].files] == SESSIONS

    def test_a_manifest_write_that_fails_removes_the_file_it_was_told_about(self, tmp_path):
        host = FakeHost(PAYLOADS)

        def refuse(manifest: Manifest) -> None:
            raise PermissionError("the manifest is locked")

        with pytest.raises(PermissionError):
            fetch(host, tmp_path, Manifest(), on_fetched=refuse, concurrency=1)

        assert list(tmp_path.iterdir()) == []
        assert list(host.requests) == [URL_FIRST]
        again = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        assert [item.session for item in again.files] == SESSIONS

    def test_a_manifest_write_that_fails_keeps_what_the_manifest_already_holds(self, tmp_path):
        first = fetch(FakeHost(PAYLOADS), tmp_path, Manifest(), sessions=["2002/03"])
        calls: list[Manifest] = []

        def refuse(manifest: Manifest) -> None:
            calls.append(manifest)
            raise PermissionError("the manifest is locked")

        with pytest.raises(PermissionError):
            fetch(FakeHost(PAYLOADS), tmp_path, first, on_fetched=refuse, concurrency=1)

        verify_raw_files(first, tmp_path)
        assert [path.name for path in tmp_path.iterdir()] == ["anforande-200203.json.zip"]
        assert len(calls) == 1

    def test_a_failed_refetch_leaves_the_old_file_and_entry_in_place(self, tmp_path):
        first = fetch(FakeHost(PAYLOADS), tmp_path, Manifest(), sessions=["2002/03"])
        with pytest.raises(httpx.HTTPStatusError):
            fetch(FakeHost({}), tmp_path, first, sessions=["2002/03"], force=True)
        verify_raw_files(first, tmp_path)
        assert [path.name for path in tmp_path.iterdir()] == ["anforande-200203.json.zip"]

    def test_a_download_that_breaks_off_leaves_no_partial_file(self, tmp_path):
        host = FakeHost(PAYLOADS, dropping=frozenset({URL_FIRST}))
        with pytest.raises(httpx.ReadError):
            fetch(host, tmp_path, Manifest(), sessions=["2002/03"], attempts=2)
        assert list(tmp_path.iterdir()) == []
        assert host.requests[URL_FIRST] == 2

    def test_a_body_that_is_not_a_zip_is_refused_and_not_recorded(self, tmp_path):
        host = FakeHost({URL_FIRST: b"<html>Underhall</html>"})
        seen: list[Manifest] = []
        with pytest.raises(ValueError, match="not a readable zip"):
            fetch(host, tmp_path, Manifest(), sessions=["2002/03"], on_fetched=seen.append)
        assert list(tmp_path.iterdir()) == []
        assert seen == []
        assert host.requests[URL_FIRST] == 1

    def test_a_zip_cut_short_is_refused(self, tmp_path):
        whole = PAYLOADS[URL_FIRST]
        host = FakeHost({URL_FIRST: whole[: len(whole) // 2]})
        with pytest.raises(ValueError, match="not a readable zip"):
            fetch(host, tmp_path, Manifest(), sessions=["2002/03"])
        assert list(tmp_path.iterdir()) == []

    def test_a_refused_body_leaves_the_file_already_on_disk_alone(self, tmp_path):
        first = fetch(FakeHost(PAYLOADS), tmp_path, Manifest(), sessions=["2002/03"])
        host = FakeHost({URL_FIRST: b"<html>Underhall</html>"})
        with pytest.raises(ValueError, match="not a readable zip"):
            fetch(host, tmp_path, first, sessions=["2002/03"], force=True)
        verify_raw_files(first, tmp_path)

    def test_a_session_given_twice_is_fetched_once(self, tmp_path):
        host = FakeHost(PAYLOADS)
        manifest = fetch(host, tmp_path, Manifest(), sessions=["2002/03", "2002/03"])
        assert [item.session for item in manifest.files] == ["2002/03"]
        assert host.requests[URL_FIRST] == 1

    def test_a_malformed_session_stops_the_run_before_any_request(self, tmp_path):
        host = FakeHost(PAYLOADS)
        with pytest.raises(ValueError, match="Not a session label"):
            fetch(host, tmp_path, Manifest(), sessions=["2002/03", "2003-04"])
        assert not host.requests
        assert list(tmp_path.iterdir()) == []


class TestManifest:
    def test_round_trips_through_the_file(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path / "raw", Manifest())
        path = tmp_path / "reports" / "manifest.json"
        write_manifest(manifest, path)
        assert load_manifest(path) == manifest

    def test_a_missing_manifest_reads_as_empty(self, tmp_path):
        assert load_manifest(tmp_path / "absent.json") == Manifest()

    def test_verification_passes_for_untouched_files(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        verify_raw_files(manifest, tmp_path)

    def test_verification_catches_a_changed_file(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        (tmp_path / "anforande-200304.json.zip").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="does not match the SHA-256"):
            verify_raw_files(manifest, tmp_path)

    def test_verification_catches_a_missing_file(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        (tmp_path / "anforande-200304.json.zip").unlink()
        with pytest.raises(FileNotFoundError):
            verify_raw_files(manifest, tmp_path)

    def test_entries_must_come_in_session_order(self):
        with pytest.raises(ValidationError, match="ordered by session"):
            Manifest(files=(entry("2003/04"), entry("2002/03")))

    def test_a_session_cannot_appear_twice_and_the_error_names_it(self):
        with pytest.raises(ValidationError, match="a session more than once: 2002/03"):
            Manifest(files=(entry("2002/03"), entry("2002/03", file="x200203.json.zip")))

    def test_every_session_that_appears_twice_is_named(self):
        entries = (
            entry("2002/03"),
            entry("2002/03", file="x200203.json.zip"),
            entry("2003/04"),
            entry("2004/05"),
            entry("2004/05", file="x200405.json.zip"),
        )
        with pytest.raises(ValidationError, match="a session more than once: 2002/03, 2004/05"):
            Manifest(files=entries)

    def test_a_file_cannot_appear_twice_and_the_error_names_it(self):
        with pytest.raises(
            ValidationError,
            match="a file more than once: anforande-200203-200304.json.zip",
        ):
            Manifest(
                files=(
                    entry("2002/03", file="anforande-200203-200304.json.zip"),
                    entry("2003/04", file="anforande-200203-200304.json.zip"),
                )
            )

    @pytest.mark.parametrize(
        "file",
        [
            "",
            ".",
            "..",
            "../outside.zip",
            "sub/inside.zip",
            "sub\\inside.zip",
            "D:evil.zip",
            "anforande-200203.json.zip:stream",
        ],
    )
    def test_an_entry_names_a_bare_file(self, file):
        with pytest.raises(ValidationError, match="bare file name"):
            entry("2002/03", file=file)

    def test_an_entry_names_the_file_its_address_ends_in(self):
        with pytest.raises(ValidationError, match="is not the file at"):
            entry("2002/03", url="https://example.org/dataset/anforande-200203.json.zip.old")

    def test_an_entry_names_a_file_of_its_own_session(self):
        with pytest.raises(ValidationError, match="is not named after session 2002/03"):
            entry("2002/03", file="anforande-209999.json.zip")

    @pytest.mark.parametrize("session", ["2002", "2002/04", "1999/00", "2002-03", ""])
    def test_an_entry_names_a_real_session(self, session):
        with pytest.raises(ValidationError, match="Not a session label|not supported"):
            entry(session, file="anforande.json.zip")

    def test_a_manifest_that_fails_its_rules_cannot_be_loaded(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest(Manifest(files=(entry("2002/03"), entry("2003/04"))), path)
        text = path.read_text(encoding="utf-8").replace('"2003/04"', '"2002/03"')
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_manifest(path)

    def test_a_failed_write_leaves_the_old_manifest_and_no_partial_file(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "manifest.json"
        old = Manifest(files=(entry("2002/03"),))
        write_manifest(old, path)

        def refuse(self: Path, target: object) -> Path:
            raise PermissionError("the manifest is open elsewhere")

        monkeypatch.setattr(Path, "replace", refuse)
        with pytest.raises(PermissionError):
            write_manifest(Manifest(files=(entry("2002/03"), entry("2003/04"))), path)
        monkeypatch.undo()

        assert load_manifest(path) == old
        assert [item.name for item in tmp_path.iterdir()] == ["manifest.json"]
