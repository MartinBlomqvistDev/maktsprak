"""Tests for search/corpus/fetch.py. Nothing here reaches the network."""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from datetime import date
from pathlib import Path

import httpx
import pytest

from search.corpus.fetch import (
    Manifest,
    dataset_url,
    fetch_datasets,
    load_manifest,
    sessions_between,
    verify_raw_files,
    write_manifest,
)

TEMPLATE = "https://example.org/dataset/anforande-{session}.json.zip"
PAYLOADS = {
    "https://example.org/dataset/anforande-200203.json.zip": b"first session",
    "https://example.org/dataset/anforande-200304.json.zip": b"second session, longer",
}
SESSIONS = ["2002/03", "2003/04"]
TODAY = date(2026, 11, 20)


class FakeHost:
    """A stand-in for the Riksdag's file host that counts what it is asked for."""

    def __init__(self, payloads: dict[str, bytes], failures_first: int = 0) -> None:
        self.payloads = payloads
        self.failures_first = failures_first
        self.requests: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests[url] += 1
        if self.requests[url] <= self.failures_first:
            return httpx.Response(503)
        if url not in self.payloads:
            return httpx.Response(404)
        return httpx.Response(200, content=self.payloads[url])


def fetch(host: FakeHost, raw_dir: Path, known: Manifest, **options: object) -> Manifest:
    """Run one fetch against the fake host."""

    async def run() -> Manifest:
        async with httpx.AsyncClient(transport=httpx.MockTransport(host)) as client:
            return await fetch_datasets(
                options.pop("sessions", SESSIONS),
                url_template=TEMPLATE,
                raw_dir=raw_dir,
                known=known,
                client=client,
                today=TODAY,
                backoff_seconds=0,
                **options,
            )

    return asyncio.run(run())


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
        for entry in manifest.files:
            payload = PAYLOADS[entry.url]
            assert (tmp_path / entry.file).read_bytes() == payload
            assert entry.bytes == len(payload)
            assert entry.sha256 == hashlib.sha256(payload).hexdigest()
            assert entry.fetched == TODAY

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

    def test_a_file_the_manifest_does_not_vouch_for_stops_the_run(self, tmp_path):
        (tmp_path / "anforande-200203.json.zip").write_bytes(b"of unknown origin")
        with pytest.raises(ValueError, match="does not match the manifest"):
            fetch(FakeHost(PAYLOADS), tmp_path, Manifest())

    def test_a_changed_file_stops_the_run(self, tmp_path):
        manifest = fetch(FakeHost(PAYLOADS), tmp_path, Manifest())
        (tmp_path / "anforande-200203.json.zip").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="does not match the manifest"):
            fetch(FakeHost(PAYLOADS), tmp_path, manifest)

    def test_entries_for_other_sessions_are_kept(self, tmp_path):
        host = FakeHost(PAYLOADS)
        first = fetch(host, tmp_path, Manifest(), sessions=["2003/04"])
        both = fetch(host, tmp_path, first, sessions=["2002/03"])
        assert [entry.session for entry in both.files] == SESSIONS

    def test_a_server_error_is_retried(self, tmp_path):
        host = FakeHost(PAYLOADS, failures_first=2)
        manifest = fetch(host, tmp_path, Manifest())
        assert len(manifest.files) == 2
        assert set(host.requests.values()) == {3}

    def test_gives_up_after_the_last_attempt(self, tmp_path):
        host = FakeHost(PAYLOADS, failures_first=5)
        with pytest.raises(httpx.HTTPStatusError):
            fetch(host, tmp_path, Manifest(), attempts=3)

    def test_a_missing_dataset_is_not_retried(self, tmp_path):
        host = FakeHost({})
        with pytest.raises(httpx.HTTPStatusError):
            fetch(host, tmp_path, Manifest(), sessions=["2002/03"])
        assert set(host.requests.values()) == {1}


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
