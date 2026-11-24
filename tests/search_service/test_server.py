"""Tests for search_service/spike/server.py. Skipped without FastAPI."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from search_service.spike.contracts import (  # noqa: E402
    Hit,
    PartyViewRequest,
    PartyViewResponse,
    SearchRequest,
    SearchResponse,
    Timings,
)
from search_service.spike.server import (  # noqa: E402
    Startup,
    create_app,
    create_app_from_environment,
    process_age,
    store_directory,
)
from tests.search_service.spike_kit import kit_settings  # noqa: E402

HIT = Hit(passage_id="GR0950-1:0-1", party="S", date=date(2004, 3, 2), score=0.5, text="Skolan.")


class FakeStore:
    passages = 12

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []
        self.vectors: list = []

    def search(self, request: SearchRequest, vector=None) -> SearchResponse:
        self.requests.append(request)
        self.vectors.append(vector)
        return SearchResponse(hits=(HIT,), allowed=12, timings=Timings(total_ms=1.5))

    def party_view(self, request: PartyViewRequest, vector=None) -> PartyViewResponse:
        self.vectors.append(vector)
        return PartyViewResponse(hits={"S": HIT, "M": None}, timings=Timings(total_ms=2.0))


class FakeEmbedder:
    dim = 4

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestServer:
    @pytest.fixture
    def client(self):
        startup = Startup(store="fake", kind="numpy", passages=12, process_age_s=None, open_s=0.1)
        store = FakeStore()
        return TestClient(create_app(store, None, startup)), store

    def test_health_says_how_the_service_started(self, client):
        http, _ = client
        assert http.get("/health").json()["store"] == "fake"

    def test_a_search_reaches_the_store_and_comes_back(self, client):
        http, store = client
        body = {"text": "skolan", "vector_seed": 3, "filter": {"party": "S"}}
        answer = SearchResponse.model_validate(http.post("/search", json=body).json())
        assert answer.hits == (HIT,)
        assert store.requests[0].filter.party == "S"

    def test_a_request_with_an_unknown_party_is_refused(self, client):
        http, _ = client
        response = http.post(
            "/search", json={"text": "x", "vector_seed": 1, "filter": {"party": "X"}}
        )
        assert response.status_code == 422

    def test_the_party_view_and_the_counters(self, client):
        http, _ = client
        views = http.post("/parties", json={"text": "skolan", "vector_seed": 3}).json()
        assert views["hits"]["M"] is None
        assert set(http.get("/usage").json()) >= {"memory_peak", "anon", "oom_kills"}

    def test_the_process_age_is_a_number_or_unknown(self):
        age = process_age()
        assert age is None or age >= 0

    def test_numpy_stores_share_a_directory(self, tmp_path):
        settings = kit_settings()
        assert store_directory(tmp_path, settings.store("numpy-8-int8")) == tmp_path / "numpy"
        lance = settings.store("lance-8-float16")
        assert store_directory(tmp_path, lance) == tmp_path / "lance-8-float16"


class TestServerWithAModel:
    @pytest.fixture
    def client(self):
        startup = Startup(
            store="fake",
            kind="numpy",
            passages=12,
            model="fake-onnx-fp32",
            process_age_s=None,
            model_import_s=0.2,
            model_load_s=0.25,
            model_first_s=0.05,
            model_s=0.5,
            open_s=0.1,
        )
        store, embedder = FakeStore(), FakeEmbedder()
        return TestClient(create_app(store, embedder, startup)), store, embedder

    def test_a_search_embeds_its_question_first(self, client):
        http, store, embedder = client
        body = http.post("/search", json={"text": "skolan", "vector_seed": 3}).json()
        answer = SearchResponse.model_validate(body)
        assert embedder.texts == ["skolan"]
        assert store.vectors[0].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert answer.timings.embed_ms is not None
        assert answer.timings.total_ms >= answer.timings.embed_ms

    def test_a_lexical_search_needs_no_vector(self, client):
        http, store, embedder = client
        http.post("/search", json={"text": "skolan", "vector_seed": 3, "mode": "lexical"})
        assert embedder.texts == []
        assert store.vectors == [None]

    def test_the_party_view_embeds_once(self, client):
        http, _, embedder = client
        http.post("/parties", json={"text": "skolan", "vector_seed": 3})
        assert embedder.texts == ["skolan"]

    def test_the_model_can_be_timed_alone(self, client):
        http, _, _ = client
        answer = http.post("/embed", json={"text": "kärnkraft"}).json()
        assert answer["dim"] == 4
        assert answer["embed_ms"] >= 0


def test_without_a_model_embed_is_not_found():
    startup = Startup(store="fake", kind="numpy", passages=12, process_age_s=None, open_s=0.1)
    http = TestClient(create_app(FakeStore(), None, startup))
    assert http.post("/embed", json={"text": "x"}).status_code == 404


def test_without_a_store_search_is_not_found():
    startup = Startup(
        store=None,
        kind=None,
        passages=None,
        model="fake-torch-fp32",
        process_age_s=None,
        open_s=None,
    )
    http = TestClient(create_app(None, FakeEmbedder(), startup))
    assert http.post("/search", json={"text": "x", "vector_seed": 1}).status_code == 404


def test_health_tells_the_parts_of_the_models_start():
    startup = Startup(
        store=None,
        kind=None,
        passages=None,
        model="fake-onnx-int8",
        process_age_s=None,
        model_import_s=0.2,
        model_load_s=0.3,
        model_first_s=0.01,
        model_s=0.51,
        open_s=None,
    )
    health = TestClient(create_app(None, FakeEmbedder(), startup)).get("/health").json()
    assert (health["model_import_s"], health["model_load_s"], health["model_first_s"]) == (
        0.2,
        0.3,
        0.01,
    )


@pytest.mark.parametrize(
    ("runtime", "precision", "message"),
    [("tensorflow", "fp32", "SPIKE_RUNTIME"), ("onnx", "bf16", "onnx runs a model")],
)
def test_a_runtime_or_precision_it_cannot_run_is_refused(monkeypatch, runtime, precision, message):
    monkeypatch.setenv("SPIKE_MODEL", "bekko-a25m")
    monkeypatch.setenv("SPIKE_RUNTIME", runtime)
    monkeypatch.setenv("SPIKE_PRECISION", precision)
    monkeypatch.delenv("SPIKE_STORE", raising=False)
    with pytest.raises(ValueError, match=message):
        create_app_from_environment()
