"""Tests for search_service/spike/build.py, on the kit's corpus."""

from __future__ import annotations

import json

from search_service.spike import build
from tests.search_service.spike_kit import kit_settings, write_spike_corpus


def test_settings_without_query_models_leave_the_prepared_ones(tmp_path, monkeypatch):
    corpus = write_spike_corpus(tmp_path / "spike")
    settings = kit_settings()
    settings = settings.model_copy(
        update={
            "stores": (),
            "query_models": settings.query_models.model_copy(update={"models": ()}),
        }
    )
    monkeypatch.setattr(build, "load_spike_settings", lambda: settings)
    out = tmp_path / "stores"
    (out / "models").mkdir(parents=True)
    (out / "models" / "preparation.json").write_text("[1]", encoding="utf-8")
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(["en fråga"]), encoding="utf-8")
    code = build.main(
        [
            "--search-dir",
            str(corpus.search_dir),
            "--tuning",
            str(corpus.tuning),
            "--out",
            str(out),
            "--models",
            "--check-texts",
            str(texts),
        ]
    )
    assert code == 0
    assert (out / "models" / "preparation.json").read_text(encoding="utf-8") == "[1]"
