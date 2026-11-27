"""Tests for search_service/app.py's start: the environment, each refusal, and the selector.

The package is the kit's, built for the settings' own embedding model and the
selector's own prompt, so that the service starts from the real settings. The
embedding API is never called: the embedder is replaced by one that knows one
question.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from search.answer.selector import load_selector_prompt  # noqa: E402
from search.config import load_settings  # noqa: E402
from search.index.embed import OpenRouterEmbedder  # noqa: E402
from search_service import app as app_module  # noqa: E402
from search_service.app import (  # noqa: E402
    ANSWER_STEP_ENV,
    CALLER_HEADER,
    SECRET_HEADER,
    create_app_from_environment,
)
from search_service.package import (  # noqa: E402
    MANIFEST_NAME,
    Embedding,
    PackageManifest,
    StalePackageError,
    read_manifest,
)
from tests.search_service.package_kit import ANSWER, build_kit_package  # noqa: E402
from tests.search_service.spike_kit import write_spike_corpus  # noqa: E402

QUESTION = "skolan skolor jobb el sjukvården vårdköerna"
SECRET = "kit-secret"
KEY = {SECRET_HEADER: SECRET, CALLER_HEADER: "198.51.100.7"}


@dataclass(frozen=True)
class Built:
    directory: Path
    digest: str
    vector: np.ndarray


def api_candidate(settings):
    return next(c for c in settings.vectors.candidates if c.kind == "openrouter")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Built:
    candidate = api_candidate(load_settings())
    embedding = Embedding(
        name=candidate.name,
        model=candidate.model,
        revision=candidate.revision,
        provider=candidate.provider,
        quantization=candidate.quantization,
        dimension=candidate.dimension,
        query_prompt=candidate.query_prompt,
    )
    answer = ANSWER.model_copy(update={"prompt_sha256": load_selector_prompt().sha256})
    corpus = write_spike_corpus(tmp_path_factory.mktemp("kit"))
    directory, _, _, matrix, _ = build_kit_package(
        corpus, tmp_path_factory.mktemp("built"), embedding=embedding, answer=answer
    )
    return Built(directory, read_manifest(directory)[1], matrix[4])


@pytest.fixture(autouse=True)
def no_embedding_api(monkeypatch, built):
    async def embed_one(self, client, text, role):
        if text != QUESTION:
            raise RuntimeError("the API failed")
        return built.vector

    monkeypatch.setattr(OpenRouterEmbedder, "embed_one", embed_one)


def environment(monkeypatch, built, directory=None, **values):
    """The service's environment: a search alone, unless *values* say otherwise."""
    settings = {
        "INDEX_DIR": str(directory or built.directory),
        "INDEX_MANIFEST_SHA256": built.digest if directory is None else None,
        "INDEX_VERIFY": "hashes",
        "SEARCH_SHARED_SECRET": SECRET,
        "OPENROUTER_API_KEY": "kit-openrouter-key",
        ANSWER_STEP_ENV: None,
        "ANSWER_REPORT": str(built.directory / "no-report.json"),
        "ANTHROPIC_API_KEY": None,
        "BUDGET_LEDGER": None,
    } | values
    for name, value in settings.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def edited(built, tmp_path, change) -> Path:
    """A copy of the package whose manifest *change* has rewritten."""
    directory = tmp_path / "package"
    shutil.copytree(built.directory, directory)
    manifest: PackageManifest = read_manifest(directory)[0]
    (directory / MANIFEST_NAME).write_text(
        change(manifest).model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return directory


def settings_with(**update):
    """The real settings, with the embedding model's entry changed."""
    real = load_settings()
    changed = api_candidate(real).model_copy(update=update)
    candidates = tuple(
        changed if candidate.name == changed.name else candidate
        for candidate in real.vectors.candidates
    )
    return real.model_copy(
        update={"vectors": real.vectors.model_copy(update={"candidates": candidates})}
    )


def lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def launch_report(tmp_path, launch=True, runs="selector") -> str:
    """A test run's report whose launch rule holds or fails, with the selector that runs."""
    path = tmp_path / "answer_test.json"
    report = {"beats_baseline": runs == "selector", "runs": runs, "launch": launch}
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def switched_on(tmp_path, **values):
    """The environment's values for an answer step that is switched on and may launch."""
    return {ANSWER_STEP_ENV: "on", "ANSWER_REPORT": launch_report(tmp_path)} | values


def health_of(client):
    return client.get("/health", headers=KEY).json()


def malformed_ledger(tmp_path) -> str:
    ledger = tmp_path / "budget.json"
    ledger.write_text("{", encoding="utf-8")
    return str(ledger)


class TestSearchAlone:
    def test_the_search_starts_without_the_key_and_the_ledger(self, monkeypatch, built):
        environment(monkeypatch, built)
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
            searched = client.post("/search", json={"question": QUESTION}, headers=KEY)
            unkeyed = client.post("/search", json={"question": QUESTION})
            refused = [
                client.post(path, json={"question": QUESTION}, headers=KEY)
                for path in ("/answer", "/parties")
            ]
        assert (health["answer_step"], health["selector"], health["ledger"]) == (
            "off",
            "none",
            "none",
        )
        assert health["manifest_sha256"] == built.digest and health["verified"] == "hashes"
        assert searched.status_code == 200 and lines(searched)[0]["retrieval"] == "hybrid"
        assert unkeyed.status_code == 401
        assert [response.status_code for response in refused] == [403, 403]
        assert {response.json()["error"] for response in refused} == {"answer_step_off"}

    def test_off_neither_the_key_nor_the_ledger_is_read(self, monkeypatch, built, tmp_path):
        environment(
            monkeypatch,
            built,
            ANTHROPIC_API_KEY="kit-anthropic-key",
            BUDGET_LEDGER=malformed_ledger(tmp_path),
        )
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
        assert (health["answer_step"], health["selector"], health["ledger"]) == (
            "off",
            "none",
            "none",
        )

    @pytest.mark.parametrize("value", ["", "off", "true", "ON", "1"])
    def test_anything_but_on_leaves_the_answer_step_off(self, monkeypatch, built, value):
        environment(monkeypatch, built, **{ANSWER_STEP_ENV: value})
        with TestClient(create_app_from_environment()) as client:
            assert health_of(client)["answer_step"] == "off"
            answered = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert answered.status_code == 403


class TestLaunchReport:
    @pytest.mark.parametrize(
        ("content", "state"),
        [
            (None, "missing"),
            ("{", "unreadable"),
            ('{"runs": "nobody", "launch": true}', "unreadable"),
            ('{"runs": "selector", "launch": false}', "no_launch"),
            ('{"runs": "baseline", "launch": "yes"}', "no_launch"),
        ],
    )
    def test_switched_on_the_answer_step_stays_off_without_a_launch(
        self, monkeypatch, built, tmp_path, content, state
    ):
        report = tmp_path / "answer_test.json"
        if content is not None:
            report.write_text(content, encoding="utf-8")
        environment(
            monkeypatch,
            built,
            **{ANSWER_STEP_ENV: "on", "ANSWER_REPORT": str(report), "ANTHROPIC_API_KEY": "k"},
        )
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
            answered = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert (health["answer_step"], health["answer_report"], health["selector"]) == (
            "off",
            state,
            "none",
        )
        assert answered.status_code == 403

    def test_a_report_that_has_the_baseline_run_never_asks_the_model(
        self, monkeypatch, built, tmp_path
    ):
        values = switched_on(tmp_path, ANTHROPIC_API_KEY="k")
        values["ANSWER_REPORT"] = launch_report(tmp_path, runs="baseline")
        environment(monkeypatch, built, **values)
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
            answer = lines(client.post("/answer", json={"question": QUESTION}, headers=KEY))[-1]
        assert (health["answer_step"], health["answer_report"], health["selector"]) == (
            "on",
            "launch",
            "baseline",
        )
        assert (answer["selector"], answer["fallback"]) == ("baseline", None)


class TestAnswerStep:
    def test_without_the_key_the_baseline_answers(self, monkeypatch, built, tmp_path):
        environment(monkeypatch, built, **switched_on(tmp_path))
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
            answered = client.post("/answer", json={"question": QUESTION}, headers=KEY)
            parties = client.post("/parties", json={"question": QUESTION}, headers=KEY)
        assert (health["answer_step"], health["selector"], health["ledger"]) == (
            "on",
            "baseline",
            "none",
        )
        search, answer = lines(answered)
        assert (search["type"], answer["type"], answer["selector"]) == (
            "search",
            "answer",
            "baseline",
        )
        assert parties.status_code == 200 and lines(parties)[0]["type"] == "parties"

    def test_with_the_key_and_a_ledger_the_model_answers(self, monkeypatch, built, tmp_path):
        pytest.importorskip("anthropic")
        values = switched_on(
            tmp_path, ANTHROPIC_API_KEY="k", BUDGET_LEDGER=str(tmp_path / "budget.json")
        )
        environment(monkeypatch, built, **values)
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
        assert (health["selector"], health["ledger"]) == ("claude", "file")

    @pytest.mark.parametrize("ledger", [None, "missing/budget.json", "gs://bucket"])
    def test_without_a_ledger_the_month_counts_as_spent(self, monkeypatch, built, tmp_path, ledger):
        place = ledger if ledger is None or ledger.startswith("gs:") else str(tmp_path / ledger)
        environment(
            monkeypatch, built, **switched_on(tmp_path, ANTHROPIC_API_KEY="k", BUDGET_LEDGER=place)
        )
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
            answered = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert (health["selector"], health["ledger"]) == ("baseline", "missing")
        answer = lines(answered)[-1]
        assert (answer["selector"], answer["fallback"]) == ("baseline", "budget")

    @pytest.mark.parametrize(
        "content",
        ["", "{", "[]", '{"month": "2026-11"}', '{"month": "2026-11", "spent_usd": "a lot"}'],
    )
    def test_an_unreadable_ledger_refuses_the_model_and_the_service_starts(
        self, monkeypatch, built, tmp_path, content
    ):
        pytest.importorskip("anthropic")
        ledger = tmp_path / "budget.json"
        ledger.write_text(content, encoding="utf-8")
        values = switched_on(tmp_path, ANTHROPIC_API_KEY="k", BUDGET_LEDGER=str(ledger))
        environment(monkeypatch, built, **values)
        with TestClient(create_app_from_environment()) as client:
            health = health_of(client)
            answered = client.post("/answer", json={"question": QUESTION}, headers=KEY)
        assert (health["selector"], health["ledger"]) == ("claude", "unreadable")
        answer = lines(answered)[-1]
        assert (answer["selector"], answer["fallback"]) == ("baseline", "budget")
        assert ledger.read_text(encoding="utf-8") == content


class TestRefusals:
    def test_an_unknown_way_of_checking_is_refused(self, monkeypatch, built):
        environment(monkeypatch, built, INDEX_VERIFY="all")
        with pytest.raises(SystemExit, match="INDEX_VERIFY"):
            create_app_from_environment()

    def test_a_package_that_is_not_the_pinned_one_is_refused(self, monkeypatch, built):
        environment(monkeypatch, built, INDEX_MANIFEST_SHA256="0" * 64)
        with pytest.raises(StalePackageError, match="pinned"):
            create_app_from_environment()

    @pytest.mark.parametrize("missing", ["SEARCH_SHARED_SECRET", "OPENROUTER_API_KEY"])
    def test_a_missing_secret_or_key_is_refused(self, monkeypatch, built, missing):
        environment(monkeypatch, built, **{missing: None})
        with pytest.raises(SystemExit, match="must both be set"):
            create_app_from_environment()

    def test_another_question_model_in_the_settings_is_refused(self, monkeypatch, built):
        environment(monkeypatch, built)
        monkeypatch.setattr(app_module, "load_settings", lambda: settings_with(revision="other"))
        with pytest.raises(SystemExit, match="question's model"):
            create_app_from_environment()

    def test_a_question_model_without_a_price_is_refused(self, monkeypatch, built):
        environment(monkeypatch, built)
        no_price = settings_with(price_per_million_usd=None)
        monkeypatch.setattr(app_module, "load_settings", lambda: no_price)
        with pytest.raises(SystemExit, match="no price"):
            create_app_from_environment()

    def test_a_store_the_manifest_does_not_record_is_refused(self, monkeypatch, built, tmp_path):
        directory = edited(
            built,
            tmp_path,
            lambda m: m.model_copy(
                update={"retrieval": m.retrieval.model_copy(update={"k1": 1.3})}
            ),
        )
        environment(monkeypatch, built, directory)
        with pytest.raises(SystemExit, match="store"):
            create_app_from_environment()

    def test_a_speeches_table_the_manifest_does_not_record_is_refused(
        self, monkeypatch, built, tmp_path
    ):
        directory = edited(built, tmp_path, lambda m: m.model_copy(update={"speeches": 99}))
        environment(monkeypatch, built, directory)
        with pytest.raises(SystemExit, match="speeches"):
            create_app_from_environment()

    @pytest.mark.parametrize("step", [None, "on"])
    def test_another_prompt_is_refused_with_the_answer_step_on_or_off(
        self, monkeypatch, built, tmp_path, step
    ):
        directory = edited(
            built,
            tmp_path,
            lambda m: m.model_copy(
                update={"answer": m.answer.model_copy(update={"prompt_sha256": "a" * 64})}
            ),
        )
        environment(monkeypatch, built, directory, **{ANSWER_STEP_ENV: step})
        with pytest.raises(SystemExit, match="prompt"):
            create_app_from_environment()
