"""Tests for search_service/app.py: the routes, over a package of the kit's corpus."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from search_service.app import CALLER_HEADER, SECRET_HEADER, create_app  # noqa: E402
from tests.search_service.package_kit import (  # noqa: E402
    FakeSelector,
    build_kit_package,
    kit_service,
)
from tests.search_service.spike_kit import write_spike_corpus  # noqa: E402

QUESTION = "skolan skolor jobb el sjukvården vårdköerna"
SECRET = "kit-secret"
KEY = {SECRET_HEADER: SECRET, CALLER_HEADER: "198.51.100.7"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = write_spike_corpus(tmp_path_factory.mktemp("kit"))
    return build_kit_package(corpus, tmp_path_factory.mktemp("built"))


def client_for(built, **options):
    directory, manifest, _, matrix, _ = built
    service = kit_service(directory, manifest, {QUESTION: matrix[4]}, **options)
    return TestClient(create_app(service, SECRET, {"status": "ok"})), service


def lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


class TestRoutes:
    def test_health_needs_no_key_and_counts(self, built):
        client, _ = client_for(built)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok" and "counters" in response.json()

    @pytest.mark.parametrize("headers", [{}, {SECRET_HEADER: "wrong"}])
    def test_a_question_without_the_secret_is_refused(self, built, headers):
        client, _ = client_for(built)
        response = client.post("/search", json={"question": QUESTION}, headers=headers)
        assert (response.status_code, response.json()) == (401, {"error": "unauthorized"})

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
