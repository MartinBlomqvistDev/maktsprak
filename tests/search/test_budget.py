"""Tests for search/answer/budget.py: the ledgers, and the guard over instances that share one."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from urllib.parse import unquote

import httpx
import pytest

from search.answer import selector as selector_module
from search.answer.budget import (
    BudgetGuard,
    FileLedger,
    GcsLedger,
    GuardedSelector,
    LedgerError,
    MemoryLedger,
    ledger_from,
    most_cost,
)
from search.answer.models import Candidate, Selected
from search.answer.selector import ABSTENTION, ClaudeSelector, load_selector_prompt
from search.config import Prices
from search.eval.llm import Completion, Usage

OPUS = Prices(input=4.0, output=20.0)
NOVEMBER = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)


def clock():
    return NOVEMBER


class FakeBucket:
    """Cloud Storage's JSON API for one bucket, and the metadata server, as the ledger asks them."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, int]] = {}
        self.generation = 0
        self.fail_reads = False
        self.interfere = 0

    def put(self, name: str, data: bytes) -> None:
        self.generation += 1
        self.objects[name] = (data, self.generation)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "metadata.google.internal":
            assert request.headers["Metadata-Flavor"] == "Google"
            return httpx.Response(200, json={"access_token": "kit-token", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer kit-token"
        path = request.url.path
        if request.method == "GET" and path.startswith("/storage/v1/b/bucket/o/"):
            if self.fail_reads:
                return httpx.Response(503)
            name = unquote(path.rsplit("/", 1)[1])
            if name not in self.objects:
                return httpx.Response(404)
            data, generation = self.objects[name]
            return httpx.Response(200, content=data, headers={"x-goog-generation": str(generation)})
        if request.method == "POST" and path == "/upload/storage/v1/b/bucket/o":
            name = request.url.params["name"]
            if self.interfere:
                # Another instance writes between this one's read and its write.
                self.interfere -= 1
                held = self.objects.get(name)
                spent = json.loads(held[0])["spent_usd"] if held else 0
                payload = {"month": "2026-11", "spent_usd": spent + 1.0}
                self.put(name, json.dumps(payload).encode())
            current = self.objects.get(name, (b"", 0))[1]
            if int(request.url.params["ifGenerationMatch"]) != current:
                return httpx.Response(412)
            self.put(name, request.content)
            return httpx.Response(200, json={"generation": str(self.generation)})
        return httpx.Response(400)


def gcs_guard(bucket: FakeBucket, monthly: float = 2.0) -> BudgetGuard:
    client = httpx.Client(transport=httpx.MockTransport(bucket))
    return BudgetGuard(monthly, GcsLedger("bucket", "budget.json", client), clock=clock)


class TestGcsLedger:
    def test_two_instances_count_together(self):
        bucket = FakeBucket()
        first, second = gcs_guard(bucket), gcs_guard(bucket)
        for _ in range(50):
            first.record(0.03)
            second.record(0.03)
        assert first.spent() == pytest.approx(3.0) and second.spent() == pytest.approx(3.0)
        assert not first.allows() and not second.allows()

    def test_a_writer_that_comes_second_reads_again(self):
        bucket = FakeBucket()
        guard = gcs_guard(bucket)
        bucket.interfere = 2
        guard.record(0.25)
        assert guard.spent() == pytest.approx(2.25)

    def test_a_new_month_starts_from_nothing(self):
        bucket = FakeBucket()
        bucket.put("budget.json", b'{"month": "2026-10", "spent_usd": 5.0}')
        guard = gcs_guard(bucket)
        assert guard.spent() == 0.0 and guard.allows()
        guard.record(0.5)
        assert json.loads(bucket.objects["budget.json"][0]) == {
            "month": "2026-11",
            "spent_usd": 0.5,
        }

    @pytest.mark.parametrize(
        "content",
        [b"", b"{", b"[]", b'{"month": "2026-11"}', b'{"month": "2026-11", "spent_usd": -1}'],
    )
    def test_a_ledger_that_is_not_a_month_and_a_sum_cannot_be_read(self, content):
        bucket = FakeBucket()
        bucket.put("budget.json", content)
        with pytest.raises(LedgerError):
            gcs_guard(bucket).allows()

    def test_a_bucket_that_does_not_answer_cannot_be_read(self):
        bucket = FakeBucket()
        bucket.fail_reads = True
        with pytest.raises(LedgerError):
            gcs_guard(bucket).spent()

    def test_without_a_token_the_ledger_cannot_be_read(self):
        def no_metadata(request):
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(no_metadata))
        with pytest.raises(LedgerError, match="token"):
            GcsLedger("bucket", "budget.json", client).read()

    def test_the_url_names_the_bucket_and_the_object(self):
        ledger = GcsLedger.from_url("gs://maktsprak-search-ledger/budget.json")
        assert (ledger.bucket, ledger.name) == ("maktsprak-search-ledger", "budget.json")
        assert isinstance(ledger_from("gs://bucket/a/b.json"), GcsLedger)
        for url in ("gs://bucket", "gs:///budget.json"):
            with pytest.raises(ValueError):
                GcsLedger.from_url(url)


class TestFileLedger:
    def test_the_file_never_makes_its_directory(self, tmp_path):
        guard = BudgetGuard(2.0, tmp_path / "missing" / "budget.json", clock=clock)
        assert guard.spent() == 0.0
        with pytest.raises(LedgerError):
            guard.record(0.1)
        assert not (tmp_path / "missing").exists()

    def test_a_write_on_a_stale_version_is_refused(self, tmp_path):
        ledger = FileLedger(tmp_path / "budget.json")
        _, version = ledger.read()
        assert ledger.write(b"one", version)
        assert not ledger.write(b"two", version)
        assert ledger.read()[0] == b"one"

    def test_the_memory_ledger_refuses_a_stale_version(self):
        ledger = MemoryLedger()
        _, version = ledger.read()
        assert ledger.write(b"one", version) and not ledger.write(b"two", version)


def candidates() -> list[Candidate]:
    return [
        Candidate(
            rank=1,
            passage_id="H1-1:0-0",
            anf_id="H1-1",
            first=0,
            last=0,
            score=3.0,
            party="S",
            speaker="Anna Andersson (S)",
            date=date(2020, 1, 1),
            title=None,
            url="https://data.riksdagen.se/anforande/H1-1",
            paragraphs=("Kärnkraften ska avvecklas.",),
            text_sha256="0" * 64,
        )
    ]


class Baseline:
    def select(self, question, candidates):
        return Selected(selector="baseline", output=ABSTENTION)


def claude() -> ClaudeSelector:
    return ClaudeSelector(load_selector_prompt(), "claude-opus-5-5", "low", 8000, OPUS)


def completion(text=None, error=None, usage=Usage(input_tokens=7000, output_tokens=100)):
    return Completion(
        custom_id="answer",
        model="claude-opus-5-5",
        stop_reason=None if error else "end_turn",
        text=text,
        usage=usage,
        error=error,
    )


@pytest.fixture
def sent(monkeypatch):
    """Stand in for the request: return the queued completions in order."""
    queue: list[Completion] = []
    asked: list[dict] = []

    def complete_one(custom_id, params, client=None):
        asked.append(params)
        return queue.pop(0)

    monkeypatch.setattr(selector_module, "complete_one", complete_one)
    return queue, asked


ANSWER = json.dumps({"answerable": False, "selections": []})


class BrokenLedger:
    def __init__(self, reads=True, writes=True):
        self.inner = MemoryLedger()
        self.reads, self.writes = reads, writes

    def read(self):
        if not self.reads:
            raise LedgerError("cannot read")
        return self.inner.read()

    def write(self, data, version):
        if not self.writes:
            raise LedgerError("cannot write")
        return self.inner.write(data, version)


class TestGuardedSelector:
    def test_a_ledger_that_cannot_be_read_lets_the_baseline_answer(self, sent):
        _, asked = sent
        guard = BudgetGuard(2.0, BrokenLedger(reads=False), clock=clock)
        selected = GuardedSelector(claude(), Baseline(), guard).select("kärnkraften", candidates())
        assert (selected.selector, selected.fallback, asked) == ("baseline", "budget", [])

    def test_an_unwritten_cost_keeps_the_answer_and_refuses_the_model_after(self, sent):
        queue, asked = sent
        queue.append(completion(text=ANSWER))
        guard = BudgetGuard(2.0, BrokenLedger(writes=False))
        guarded = GuardedSelector(claude(), Baseline(), guard)
        first = guarded.select("kärnkraften", candidates())
        second = guarded.select("kärnkraften", candidates())
        assert (first.selector, first.fallback) == ("claude", None)
        assert (second.selector, second.fallback) == ("baseline", "budget")
        assert len(asked) == 1

    def test_a_request_that_broke_off_counts_the_most_it_could_cost(self, sent):
        queue, asked = sent
        queue.extend([completion(error="connection: timed out", usage=Usage())] * 2)
        guard = BudgetGuard(10.0, clock=clock)
        selector = claude()
        selected = GuardedSelector(selector, Baseline(), guard).select("kärnkraften", candidates())
        assert (selected.fallback, selected.attempts) == ("failed", 2)
        assert guard.spent() == pytest.approx(2 * most_cost(selector, asked[0]))

    def test_a_request_the_api_refused_costs_nothing(self, sent):
        queue, _ = sent
        queue.extend([completion(error="HTTP 529: overloaded", usage=Usage())] * 2)
        guard = BudgetGuard(2.0, clock=clock)
        GuardedSelector(claude(), Baseline(), guard).select("kärnkraften", candidates())
        assert guard.spent() == 0.0

    def test_the_budget_is_checked_again_before_the_second_request(self, sent):
        queue, asked = sent
        queue.extend([completion(error="HTTP 500: a"), completion(text=ANSWER)])
        guard = BudgetGuard(0.01, clock=clock)
        selected = GuardedSelector(claude(), Baseline(), guard).select("kärnkraften", candidates())
        assert (selected.selector, selected.fallback, len(asked)) == ("baseline", "budget", 1)

    def test_instances_that_share_a_ledger_stop_together(self, sent):
        queue, asked = sent
        ledger = MemoryLedger()
        first = GuardedSelector(claude(), Baseline(), BudgetGuard(0.05, ledger, clock=clock))
        second = GuardedSelector(claude(), Baseline(), BudgetGuard(0.05, ledger, clock=clock))
        queue.extend([completion(text=ANSWER)] * 2)
        first.select("kärnkraften", candidates())  # 0.03, below the budget
        second.select("kärnkraften", candidates())  # 0.06, past it by one request
        assert second.select("kärnkraften", candidates()).fallback == "budget"
        assert first.select("kärnkraften", candidates()).fallback == "budget"
        assert len(asked) == 2

    def test_the_most_a_request_can_cost_bounds_its_tokens(self):
        selector = claude()
        params = selector.request("kärnkraften", candidates())
        request_bytes = len(json.dumps(params, ensure_ascii=False).encode("utf-8"))
        assert most_cost(selector, params) == pytest.approx(
            (request_bytes * OPUS.input + 8000 * OPUS.output) / 1e6
        )
