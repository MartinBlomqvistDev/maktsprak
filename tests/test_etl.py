"""Tests for the incremental ETL: retries, the protocol listing, the watermark and the run."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests

from src.maktsprak_pipeline.pipeline import extract, orchestrate, transform


def _listing(docs: list[tuple[str, str]], total: int) -> bytes:
    """Build a Riksdag protocol listing page holding *docs* as (dok_id, datum)."""
    body = "".join(
        f"<dokument><dok_id>{pid}</dok_id><datum>{day}</datum>"
        f"<filbilaga><fil><url>https://example.com/{pid}.pdf</url></fil></filbilaga>"
        "</dokument>"
        for pid, day in docs
    )
    return f'<dokumentlista traffar="{total}">{body}</dokumentlista>'.encode()


def _response(content: bytes) -> MagicMock:
    """A stand-in for a successful :class:`requests.Response`."""
    return MagicMock(content=content, status_code=200)


class TestCallWithBackoff:
    def test_returns_once_a_retry_succeeds(self):
        fn = MagicMock(side_effect=[requests.ConnectionError("reset"), "ok"])
        with patch.object(extract.time, "sleep") as sleep:
            result = extract.call_with_backoff(
                fn, retry_on=(requests.RequestException,), what="test", first_wait=5
            )
        assert result == "ok"
        assert fn.call_count == 2
        sleep.assert_called_once_with(5)

    def test_the_wait_doubles_until_it_gives_up(self):
        fn = MagicMock(side_effect=requests.ConnectionError("down"))
        with (
            patch.object(extract.time, "sleep") as sleep,
            pytest.raises(RuntimeError, match="after 6 attempts") as raised,
        ):
            extract.call_with_backoff(
                fn, retry_on=(requests.RequestException,), what="test", attempts=6, first_wait=5
            )
        assert fn.call_count == 6
        assert [c.args[0] for c in sleep.call_args_list] == [5, 10, 20, 40, 80]
        assert isinstance(raised.value.__cause__, requests.ConnectionError)

    def test_other_errors_are_not_retried(self):
        fn = MagicMock(side_effect=ValueError("a bug, not the network"))
        with patch.object(extract.time, "sleep") as sleep, pytest.raises(ValueError):
            extract.call_with_backoff(fn, retry_on=(requests.RequestException,), what="test")
        assert fn.call_count == 1
        sleep.assert_not_called()


class TestDownloadPdf:
    def test_a_reset_connection_is_retried(self, tmp_path):
        side_effect = [requests.ConnectionError("reset"), _response(b"%PDF-1.7")]
        with (
            patch.object(extract.requests, "get", side_effect=side_effect) as get,
            patch.object(extract.time, "sleep"),
        ):
            transform._download_pdf("https://example.com/HE0913.pdf", tmp_path / "HE0913.pdf")
        assert get.call_count == 2
        assert (tmp_path / "HE0913.pdf").read_bytes() == b"%PDF-1.7"


class TestFetchProtocolDocs:
    def test_pages_through_the_listing_and_saves_every_page(self, tmp_path):
        first = [(f"HE09{n}", "2026-11-03") for n in range(10, 30)]
        second = [("HE0930", "2026-11-04"), ("HE0931", "2026-11-04")]
        pages = [_response(_listing(first, 22)), _response(_listing(second, 22))]
        with (
            patch.object(extract, "RAW_DATA_PATH", tmp_path),
            patch.object(extract.requests, "get", side_effect=pages) as get,
            patch.object(extract.time, "sleep"),
        ):
            docs = extract.fetch_protocol_docs("2026-11-01", "2026-11-27")
        assert len(docs) == 22
        assert "&p=2" in get.call_args_list[1].args[0]
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "protocols_20261101_20261127.xml",
            "protocols_20261101_20261127_p2.xml",
        ]

    def test_an_empty_window_saves_nothing(self, tmp_path):
        with (
            patch.object(extract, "RAW_DATA_PATH", tmp_path),
            patch.object(extract.requests, "get", return_value=_response(_listing([], 0))),
        ):
            assert extract.fetch_protocol_docs("2026-07-01", "2026-07-07") == []
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_page_is_retried(self, tmp_path):
        side_effect = [
            requests.ConnectionError("Remote end closed connection without response"),
            _response(_listing([("HE0913", "2026-11-03")], 1)),
        ]
        with (
            patch.object(extract, "RAW_DATA_PATH", tmp_path),
            patch.object(extract.requests, "get", side_effect=side_effect),
            patch.object(extract.time, "sleep"),
        ):
            docs = extract.fetch_protocol_docs("2026-11-01", "2026-11-27")
        assert [d.findtext("dok_id") for d in docs] == ["HE0913"]


class TestListingStart:
    def test_lists_from_the_watermark_minus_the_margin(self):
        with patch.object(orchestrate, "fetch_latest_speech_date", return_value="2026-11-25"):
            assert orchestrate.listing_start(margin_days=14) == date(2026, 11, 11)

    def test_does_not_move_with_today(self):
        # The old window was today minus seven days, so a week whose run failed
        # fell out of it for good. The watermark waits for the data instead.
        with patch.object(orchestrate, "fetch_latest_speech_date", return_value="2026-10-22"):
            one_week = orchestrate.listing_start(margin_days=14, today=date(2026, 10, 29))
            three_weeks = orchestrate.listing_start(margin_days=14, today=date(2026, 11, 12))
        assert one_week == three_weeks == date(2026, 10, 8)

    def test_an_empty_table_lists_the_margin_back_from_today(self):
        with patch.object(orchestrate, "fetch_latest_speech_date", return_value=None):
            start = orchestrate.listing_start(margin_days=14, today=date(2026, 11, 27))
        assert start == date(2026, 11, 13)

    def test_a_failed_name_lookup_is_retried(self):
        side_effect = [httpx.ConnectError("getaddrinfo failed"), "2026-11-25"]
        with (
            patch.object(orchestrate, "fetch_latest_speech_date", side_effect=side_effect),
            patch.object(extract.time, "sleep"),
        ):
            assert orchestrate.listing_start(margin_days=14) == date(2026, 11, 11)


class TestRunEtl:
    DOCS = [("HE0913", "2026-11-03"), ("HE0914", "2026-11-04"), ("HE0915", "2026-11-10")]

    @staticmethod
    def _docs(entries: list[tuple[str, str]]) -> list[ET.Element]:
        return ET.fromstring(_listing(entries, len(entries))).findall(".//dokument")

    @staticmethod
    def _records(doc: ET.Element) -> list[dict[str, str]]:
        return [{"id": f"{doc.findtext('dok_id')}_anna-andersson_s"}]

    def _run(self, process):
        """Run the ETL over DOCS with every external call replaced."""
        with (
            patch.object(orchestrate, "listing_start", return_value=date(2026, 10, 20)),
            patch.object(orchestrate, "fetch_protocol_docs", return_value=self._docs(self.DOCS)),
            patch.object(orchestrate, "_process_doc", side_effect=process),
            patch.object(orchestrate, "insert_speeches", side_effect=len) as insert,
            patch.object(orchestrate, "extract_all_tweets", return_value=[]),
            patch.object(orchestrate.time, "sleep"),
        ):
            try:
                orchestrate.run_etl()
            finally:
                self.inserted = [c.args[0][0]["id"] for c in insert.call_args_list]

    def test_a_clean_run_upserts_each_protocol_in_order(self):
        self._run(self._records)
        assert self.inserted == [
            "HE0913_anna-andersson_s",
            "HE0914_anna-andersson_s",
            "HE0915_anna-andersson_s",
        ]

    def test_a_failed_protocol_does_not_stop_the_others_but_fails_the_run(self):
        def process(doc: ET.Element) -> list[dict[str, str]]:
            if doc.findtext("dok_id") == "HE0914":
                raise RuntimeError("PDF download failed after 6 attempts")
            return self._records(doc)

        with pytest.raises(RuntimeError, match="1 of 3 protocols failed: HE0914"):
            self._run(process)
        assert self.inserted == ["HE0913_anna-andersson_s", "HE0915_anna-andersson_s"]

    def test_lists_from_the_watermark(self):
        with (
            patch.object(orchestrate, "listing_start", return_value=date(2026, 11, 11)),
            patch.object(orchestrate, "fetch_protocol_docs", return_value=[]) as fetch,
            patch.object(orchestrate, "extract_all_tweets", return_value=[]),
        ):
            orchestrate.run_etl()
        fetch.assert_called_once_with("2026-11-11")
