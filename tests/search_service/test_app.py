"""Tests for search_service/app.py: the gate and the routes, over a package of the kit's corpus."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from search_service.app import (  # noqa: E402
    CALLER_HEADER,
    MAX_BODY_BYTES,
    SECRET_HEADER,
    RequestGate,
    create_app,
)
from search_service.service import (  # noqa: E402
    QUESTION_MAX_CHARS,
    SPEAKER_MAX_CHARS,
    RateLimiter,
)
from tests.search_service.package_kit import (  # noqa: E402
    FakeSelector,
    build_kit_package,
    kit_service,
)
from tests.search_service.spike_kit import write_spike_corpus  # noqa: E402

QUESTION = "skolan skolor jobb el sjukvården vårdköerna"
SECRET = "kit-secret"
KEY = {SECRET_HEADER: SECRET, CALLER_HEADER: "198.51.100.7"}
ROUTES = ("/search", "/answer", "/parties")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = write_spike_corpus(tmp_path_factory.mktemp("kit"))
    return build_kit_package(corpus, tmp_path_factory.mktemp("built"))


def client_for(built, answer_step=True, **options):
    directory, manifest, _, matrix, _ = built
    service = kit_service(directory, manifest, {QUESTION: matrix[4]}, **options)
    app = create_app(service, SECRET, {"status": "ok"}, answer_step=answer_step)
    return TestClient(app), service


def lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


class Recorder:
    """An application behind the gate that records the body it was handed."""

    def __init__(self):
        self.bodies = []

    async def __call__(self, scope, receive, send):
        message = await receive()
        self.bodies.append(message["body"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def body(*chunks, ends=True):
    """A body as the connection sends it, in chunks, and a disconnect when it does not end."""
    messages = [
        {"type": "http.request", "body": chunk, "more_body": not ends or index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]
    return messages if ends else [*messages, {"type": "http.disconnect"}]


def through_gate(headers, messages, kind="http", max_body=64):
    """One request through the gate.

    Returns:
        The messages the gate sent, how many messages of the body it asked
        for, and the bodies the application behind it was handed.
    """
    app = Recorder()
    gate = RequestGate(app, SECRET, max_body=max_body)
    scope = {
        "type": kind,
        "method": "POST",
        "path": "/search",
        "headers": [(name.encode("ascii"), value.encode("ascii")) for name, value in headers],
    }
    queue = list(messages)
    asked = []
    sent = []

    async def receive():
        asked.append(True)
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(gate(scope, receive, send))
    return sent, len(asked), app.bodies


def status(sent):
    return sent[0]["status"] if sent else None


class TestGate:
    @pytest.mark.parametrize("path", [*ROUTES, "/docs", "/elsewhere"])
    @pytest.mark.parametrize("headers", [{}, {SECRET_HEADER: "wrong"}, {SECRET_HEADER: ""}])
    def test_a_request_without_the_secret_is_refused_on_every_path_but_health(
        self, built, path, headers
    ):
        client, _ = client_for(built)
        response = client.post(path, json={"question": QUESTION}, headers=headers)
        assert (response.status_code, response.json()) == (401, {"error": "unauthorized"})

    def test_without_the_secret_a_request_outside_the_contract_tells_nothing(self, built):
        client, _ = client_for(built)
        for body in ({}, {"question": QUESTION, "mode": "answer"}, {"question": 7}):
            response = client.post("/search", json=body)
            assert response.status_code == 401

    def test_without_the_secret_the_body_is_never_read(self):
        sent, asked, handed = through_gate([], body(b"x" * 10_000))
        assert (status(sent), asked, handed) == (401, 0, [])

    def test_a_length_over_the_limit_is_refused_before_the_body_is_read(self):
        headers = [(SECRET_HEADER, SECRET), ("content-length", "65")]
        sent, asked, handed = through_gate(headers, body(b"x" * 65))
        assert (status(sent), asked, handed) == (413, 0, [])

    def test_a_length_that_is_not_a_number_is_refused(self):
        headers = [(SECRET_HEADER, SECRET), ("content-length", "12a")]
        sent, asked, _ = through_gate(headers, body(b"x"))
        assert (status(sent), asked) == (400, 0)

    def test_a_streamed_body_is_held_to_the_same_limit(self):
        sent, asked, handed = through_gate([(SECRET_HEADER, SECRET)], body(*[b"x" * 40] * 10))
        assert (status(sent), asked, handed) == (413, 2, [])

    def test_a_body_within_the_limit_is_handed_on_whole(self):
        sent, asked, handed = through_gate([(SECRET_HEADER, SECRET)], body(b"a" * 30, b"b" * 34))
        assert (status(sent), asked, handed) == (204, 2, [b"a" * 30 + b"b" * 34])

    def test_a_caller_that_goes_away_gets_no_answer(self):
        sent, _, handed = through_gate([(SECRET_HEADER, SECRET)], body(b"a", ends=False))
        assert (sent, handed) == ([], [])

    def test_a_websocket_is_closed(self):
        sent, asked, handed = through_gate([(SECRET_HEADER, SECRET)], [], kind="websocket")
        assert (sent, asked, handed) == ([{"type": "websocket.close", "code": 1008}], 0, [])

    def test_the_limit_holds_through_the_application(self, built):
        client, _ = client_for(built)
        padded = {"question": QUESTION, "filters": {}, "pad": "x" * MAX_BODY_BYTES}
        response = client.post("/search", json=padded, headers=KEY)
        assert (response.status_code, response.json()) == (413, {"error": "invalid_request"})

    @pytest.mark.parametrize(
        "body",
        [
            {"question": "x" * (QUESTION_MAX_CHARS + 1)},
            {"question": QUESTION, "filters": {"speaker": "1" * (SPEAKER_MAX_CHARS + 1)}},
        ],
    )
    def test_a_question_or_speaker_beyond_its_length_is_refused(self, built, body):
        client, _ = client_for(built)
        response = client.post("/search", json=body, headers=KEY)
        assert (response.status_code, response.json()) == (400, {"error": "invalid_request"})

    def test_a_gate_needs_a_secret(self):
        with pytest.raises(ValueError, match="secret"):
            RequestGate(Recorder(), "")


class TestAnswerStep:
    @pytest.mark.parametrize("path", ["/answer", "/parties"])
    def test_off_by_default_the_answer_step_is_refused(self, built, path):
        directory, manifest, *_ = built
        client = TestClient(create_app(kit_service(directory, manifest, {}), SECRET, {}))
        response = client.post(path, json={"question": QUESTION}, headers=KEY)
        assert (response.status_code, response.json()) == (403, {"error": "answer_step_off"})

    def test_off_the_search_still_answers(self, built):
        client, service = client_for(built, answer_step=False)
        response = client.post("/search", json={"question": QUESTION}, headers=KEY)
        assert response.status_code == 200 and lines(response)[0]["type"] == "search"
        assert client.post("/answer", json={"question": QUESTION}, headers=KEY).status_code == 403
        assert (service.counters["answers"], service.selector.asked) == (0, [])

    def test_off_a_refused_answer_counts_against_no_limit(self, built):
        client, service = client_for(built, answer_step=False, limits={"search": 1, "answer": 1})
        for _ in range(3):
            client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert service.counters["rate_limited"] == 0


class TestHealth:
    def test_without_the_key_health_says_only_that_it_is_up(self, built):
        directory, manifest, *_ = built
        health = {"status": "ok", "manifest_sha256": "f" * 64, "selector": "claude"}
        client = TestClient(create_app(kit_service(directory, manifest, {}), SECRET, health))
        for headers in ({}, {SECRET_HEADER: "wrong"}):
            response = client.get("/health", headers=headers)
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "manifest_sha256": "f" * 64}

    def test_with_the_key_health_says_how_it_is_set_up_and_counts(self, built):
        client, _ = client_for(built)
        client.post("/search", json={"question": QUESTION}, headers=KEY)
        response = client.get("/health", headers=KEY)
        assert response.json()["status"] == "ok"
        assert response.json()["counters"]["searches"] == 1

    def test_health_forgets_callers_whose_window_has_passed(self, built):
        directory, manifest, *_ = built
        service = kit_service(directory, manifest, {})
        clock = [0.0]
        service.rates = RateLimiter(60, {"search": 5, "answer": 5}, clock=lambda: clock[0])
        service.rates.check("198.51.100.7", "search")
        clock[0] = 61.0
        TestClient(create_app(service, SECRET, {})).get("/health")
        assert len(service.rates) == 0


class TestRoutes:
    def test_a_search_is_one_line(self, built):
        client, _ = client_for(built)
        response = client.post("/search", json={"question": QUESTION}, headers=KEY)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        (line,) = lines(response)
        assert (line["type"], line["retrieval"]) == ("search", "hybrid") and line["hits"]

    def test_an_answer_is_the_search_then_the_answer(self, built):
        client, _ = client_for(built)
        response = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert [line["type"] for line in lines(response)] == ["search", "answer"]

    def test_a_question_the_api_cannot_embed_is_searched_with_bm25(self, built):
        client, _ = client_for(built)
        response = client.post("/search", json={"question": "skolan"}, headers=KEY)
        (line,) = lines(response)
        assert (line["retrieval"], line["vector_wait_ms"]) == ("bm25", None)

    def test_a_broken_selector_ends_the_stream_with_an_error_line(self, built):
        client, _ = client_for(built, selector=FakeSelector(fail=True))
        response = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert lines(response)[-1] == {"type": "error", "error": "answer_failed"}

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"question": QUESTION, "mode": "answer"},
            {"question": "x" * 401},
            {"question": QUESTION, "filters": {"party": "FP"}},
        ],
    )
    def test_a_request_outside_the_contract_is_refused(self, built, body):
        client, _ = client_for(built)
        response = client.post("/answer", json=body, headers=KEY)
        assert (response.status_code, response.json()) == (400, {"error": "invalid_request"})

    def test_a_caller_beyond_the_limit_is_told_when_to_come_back(self, built):
        client, service = client_for(built, limits={"search": 5, "answer": 1})
        first = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        second = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert first.status_code == 200
        assert (second.status_code, second.json()) == (429, {"error": "rate_limited"})
        assert int(second.headers["retry-after"]) > 0
        assert service.counters["rate_limited"] == 1

    def test_the_view_of_the_parties_is_one_line_of_eight(self, built):
        client, _ = client_for(built)
        response = client.post("/parties", json={"question": QUESTION}, headers=KEY)
        (line,) = lines(response)
        assert line["type"] == "parties" and len(line["parties"]) == 8
        refused = client.post(
            "/parties", json={"question": QUESTION, "filters": {"party": "S"}}, headers=KEY
        )
        assert refused.status_code == 400

    def test_an_empty_secret_is_refused(self, built):
        directory, manifest, *_ = built
        with pytest.raises(ValueError, match="secret"):
            create_app(kit_service(directory, manifest, {}), "", {})
