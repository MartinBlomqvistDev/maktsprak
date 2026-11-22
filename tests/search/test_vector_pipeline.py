"""Steps 06, 07 and 13 end to end on the synthetic corpus.

A deterministic embedder stands in for the models, so no model is loaded and no
request leaves the machine. The gates that check git and the pre-registration
are stood aside, since the synthetic question sets are registered nowhere; the
gates themselves are tested on their own below.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from types import ModuleType

import pytest

from search.config import (
    REPO_ROOT,
    CandidateSettings,
    PassageSettings,
    TrackedPaths,
    VectorSettings,
    load_settings,
)
from search.corpus.passages import paragraph_word_counts
from search.eval import hybrid
from search.eval.questions import EvalQuestion, Provenance, QuestionFilter, write_question_set
from search.index.analyzer import Analyzer
from search.index.embed import EmbedderSpec, HashEmbedder
from search.index.lexical import (
    IndexManifest,
    build_index,
    file_sha256,
    lexical_dir,
    passage_ids_sha256,
    passage_texts,
    tokenize_corpus,
)
from tests.search.corpus_kit import SPEECHES, write_corpus

PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 22), generation=1
)


def load_script(name: str) -> ModuleType:
    """Import a numbered pipeline script as a module."""
    path = REPO_ROOT / "search" / "pipeline" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pipeline_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SpecHashEmbedder(HashEmbedder):
    """The deterministic embedder, wearing a candidate's spec."""

    def __init__(self, candidate: CandidateSettings) -> None:
        super().__init__(dimension=candidate.dimension, name=candidate.name)
        self.spec = EmbedderSpec.from_candidate(candidate)


def fake_build(candidate: CandidateSettings, **_: object) -> SpecHashEmbedder:
    return SpecHashEmbedder(candidate)


def generated(set_name: str, number: int, split: str, text: str, passage: str, **extra):
    return EvalQuestion(
        question_id=f"{set_name}-{number:04d}",
        question_set=set_name,
        split=split,
        text=text,
        gold=(passage.split(":")[0],),
        passage_id=passage,
        provenance=PROVENANCE,
        **extra,
    )


QUESTIONS = {
    "natural": [
        generated("natural", 1, "dev", "Fler lärare och mindre klasser i skolan", "GR0950-1:0-1"),
        generated("natural", 2, "dev", "Kortare köer i sjukvården", "HB0960-1:0-1"),
        generated("natural", 3, "test", "Ska kärnkraften byggas ut?", "H209100-1:0-1"),
        generated("natural", 4, "test", "Skatten på arbete", "H109050-2:0-1"),
    ],
    "keyword": [
        generated("keyword", 1, "dev", "lärare klasser skolan", "GR0950-1:0-1"),
        generated("keyword", 2, "test", "kärnkraft utbyggnad", "H209100-1:0-1"),
    ],
    "interpellation": [
        EvalQuestion(
            question_id=f"interpellation-000{number}",
            question_set="interpellation",
            split=split,
            text=text,
            gold=gold,
            session=session,
        )
        for number, split, text, gold, session in (
            (1, "dev", "vårdköerna", ("HB0960-1", "HB0960-2"), "2022/23"),
            (2, "test", "friskolorna", ("GR0950-2",), "2003/04"),
        )
    ],
    "filtered": [
        generated(
            "filtered", 1, "dev", "kärnkraften", "H209100-2:0-1", filter=QuestionFilter(party="MP")
        ),
        generated(
            "filtered",
            2,
            "test",
            "skolan",
            "GR0950-1:0-1",
            filter=QuestionFilter(period="2003-2009"),
        ),
    ],
    "stance": [
        generated(
            "stance",
            1,
            "dev",
            "Vilka argument har förts fram mot kärnkraftsavvecklingen?",
            "H109050-1:0-1",
            kind="against",
        ),
        generated(
            "stance",
            2,
            "test",
            "Vilka argument har förts fram för friskolor?",
            "GR0950-2:0-1",
            kind="for",
        ),
    ],
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A data root with the synthetic corpus, its BM25 index, question sets and settings."""
    root = tmp_path / "root"
    search_dir = root / "data" / "search"
    search_dir.mkdir(parents=True)
    speeches_path, passages_path = write_corpus(search_dir)
    reports = tmp_path / "reports"
    questions_dir = reports / "questions"
    for name, questions in QUESTIONS.items():
        write_question_set(questions_dir / f"{name}.jsonl", questions)

    words = sum(sum(paragraph_word_counts(speech)) for speech in SPEECHES)
    paragraphs = sum(len(speech.paragraphs) for speech in SPEECHES)
    (reports / "anforanden_build.json").write_text(
        json.dumps(
            {
                "speeches": len(SPEECHES),
                "paragraphs": paragraphs,
                "words": words,
                "inputs_sha256": "x",
            }
        ),
        encoding="utf-8",
    )
    (reports / "bm25_tuning_dev.json").write_text(
        json.dumps({"chosen": {"analyzer": "stem-nostop", "k1": 1.2, "b": 0.9}}), encoding="utf-8"
    )
    (reports / "embedding_candidates.json").write_text(
        json.dumps({"api_candidates": {"remote": {"corpus_tokens": 1000}}}), encoding="utf-8"
    )
    analyzer = Analyzer(stem=True, stopwords=False)
    corpus = tokenize_corpus(passage_texts(speeches_path, passages_path))
    index = build_index(corpus, analyzer)
    index.save(
        lexical_dir(search_dir, analyzer),
        IndexManifest(
            analyzer=analyzer,
            passages=index.passages,
            passage_ids_sha256=passage_ids_sha256(corpus.passage_ids),
            passages_sha256=file_sha256(passages_path),
            speeches_sha256=file_sha256(speeches_path),
            terms=len(index.terms),
            postings=len(index.doc_ids),
            tokens=int(index.doc_lengths.sum()),
            mean_passage_terms=index.mean_length,
        ),
    )

    local = CandidateSettings(
        name="small",
        kind="sentence_transformers",
        model="org/small",
        revision="a",
        dimension=64,
        query_prompt="",
        document_prompt="",
        max_tokens=64,
        licensed=True,
    )
    remote = CandidateSettings(
        name="remote",
        kind="openrouter",
        model="org/remote",
        revision="b",
        dimension=32,
        query_prompt="",
        document_prompt="",
        provider="p",
        quantization="bf16",
        price_per_million_usd=0.01,
        licensed=True,
    )
    vectors = VectorSettings(
        candidates=(local, remote),
        tie_margin=0.01,
        api_budget_usd=10.0,
        run_cap_usd=1.0,
        subset_seed=1,
        subset_size=6,
        shard_rows=3,
        window=50,
        rrf_k=(10.0, 60.0),
        rrf_k_preferred=60.0,
        api_batch_inputs=8,
        api_concurrency=1,
    )
    settings = load_settings(data_root=root).model_copy(
        update={
            "tracked": TrackedPaths(reports=reports, questions=questions_dir),
            "passages": PassageSettings(target_words=200, trial_targets=(120, 320)),
            "vectors": vectors,
        }
    )
    embed_step, index_step, evaluate = (
        load_script("06_embed_passages"),
        load_script("07_build_vector_index"),
        load_script("13_evaluate_vectors"),
    )
    monkeypatch.setattr(embed_step, "build_embedder", fake_build)
    monkeypatch.setattr(evaluate, "build_embedder", fake_build)
    monkeypatch.setattr(evaluate, "require_committed", lambda path: None)
    monkeypatch.setattr(evaluate, "require_registered", lambda texts, what: None)
    monkeypatch.setattr(
        evaluate,
        "require_sets_registered",
        lambda directory: {name: "h" * 64 for name in QUESTIONS},
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return settings, embed_step, index_step, evaluate


def embed_args(candidate: str, inputs: str) -> argparse.Namespace:
    return argparse.Namespace(
        candidate=candidate,
        inputs=inputs,
        device=None,
        batch_size=8,
        precision="fp32",
        max_shards=None,
        max_cost_usd=None,
        env_file=None,
    )


def test_steps_06_07_and_13_run_from_the_subset_to_the_test_split(world):
    settings, embed_step, index_step, evaluate = world
    reports = settings.tracked.reports

    embed_step.export(settings, "subset", 200)
    subset = json.loads((reports / "vector_subset.json").read_text(encoding="utf-8"))
    assert subset["gold_speeches"] == 5 and subset["passages"] >= 6
    for candidate in ("small", "remote"):
        embed_step.embed(settings, embed_args(candidate, "subset-200"))
        index_step.build(settings, "subset-200", candidate)
    assert (reports / "vector_runs" / "subset-200--small.json").is_file()

    evaluate.run_compare(settings)
    comparison = json.loads((reports / "vector_comparison_dev.json").read_text(encoding="utf-8"))
    assert comparison["missing"] == []
    assert [score["name"] for score in comparison["scores"]] == ["small", "remote"]
    assert comparison["choice"]["chosen"] == "small"
    assert set(comparison["bm25_on_subset"]) == set(QUESTIONS)

    for words in (200, 120, 320):
        embed_step.export(settings, "full", words)
        embed_step.embed(settings, embed_args("small", f"full-{words}"))
        index_step.build(settings, f"full-{words}", "small")
    assert (reports / "passages_120.json").is_file()

    evaluate.run_tune(settings)
    tuning = json.loads((reports / "hybrid_tuning_dev.json").read_text(encoding="utf-8"))
    assert len(tuning["grid"]) == 6 and tuning["candidate"] == "small"
    assert [size["target_words"] for size in tuning["sizes"]] == [200, 120, 320]

    evaluate.run_test(settings, None)
    result = json.loads((reports / "retrieval_test.json").read_text(encoding="utf-8"))
    assert [c["system"] for c in result["comparisons"]] == ["hybrid", "vectors", "vectors"]
    assert [c["question_set"] for c in result["comparisons"]] == ["natural", "natural", "keyword"]
    assert result["decision"]["outcome"] in ("A", "B")
    assert set(result["sets"]) == {"bm25", "vectors", "hybrid"}
    with pytest.raises(SystemExit, match="run once"):
        evaluate.run_test(settings, None)


def test_the_comparison_waits_for_every_candidate(world):
    settings, embed_step, index_step, evaluate = world
    embed_step.export(settings, "subset", 200)
    embed_step.embed(settings, embed_args("small", "subset-200"))
    index_step.build(settings, "subset-200", "small")
    evaluate.run_compare(settings)
    report = json.loads(
        (settings.tracked.reports / "vector_comparison_dev.json").read_text(encoding="utf-8")
    )
    assert report["missing"] == ["remote"] and report["choice"] is None
    with pytest.raises(SystemExit, match="every candidate"):
        evaluate.run_tune(settings)


def test_the_comparison_refuses_a_local_run_outside_float32(world):
    settings, embed_step, index_step, evaluate = world
    embed_step.export(settings, "subset", 200)
    embed_step.embed(settings, embed_args("small", "subset-200"))
    index_step.build(settings, "subset-200", "small")
    path = evaluate.run_path(settings.data.search, "subset-200", "small") / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**manifest, "precision": "bf16"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="computed in bf16"):
        evaluate.run_compare(settings)


def test_the_test_run_needs_confirm(world, monkeypatch):
    _, _, _, evaluate = world
    monkeypatch.setattr("sys.argv", ["13_evaluate_vectors.py", "test"])
    with pytest.raises(SystemExit, match="--confirm"):
        evaluate.main()


def test_the_bundle_readme_names_every_local_candidate():
    embed_step = load_script("06_embed_passages")
    text = embed_step.bundle_readme(
        "subset-200", "stepA", "stepA.zip", ["saga-embed", "arctic-l-v2"]
    )
    assert "--candidate saga-embed --device mps" in text
    assert "--candidate arctic-l-v2 --device cuda" in text
    assert "data/search/vectors/runs/subset-200/<candidate>/" in text


class TestGates:
    def test_a_file_that_is_not_committed_stops_the_step(self, tmp_path):
        path = REPO_ROOT / "data" / "reports" / "search" / "no-such-report.json"
        with pytest.raises(SystemExit, match="does not exist"):
            hybrid.require_committed(path)

    def test_text_missing_from_the_preregistration_stops_the_step(self, monkeypatch):
        monkeypatch.setattr(hybrid, "committed_bytes", lambda relative: b"hash-one hash-two")
        hybrid.require_registered(["hash-one"], "a set")
        with pytest.raises(SystemExit, match="hash-three"):
            hybrid.require_registered(["hash-one", "hash-three"], "a set")

    def test_a_changed_file_stops_the_step(self, monkeypatch, tmp_path):
        target = REPO_ROOT / "data" / "reports" / "search" / "bm25_tuning_dev.json"
        monkeypatch.setattr(hybrid, "committed_bytes", lambda relative: b"something else")
        with pytest.raises(SystemExit, match="must be committed"):
            hybrid.require_committed(target)

    def test_line_endings_do_not_change_a_texts_hash(self, tmp_path):
        unix, windows = tmp_path / "unix.json", tmp_path / "windows.json"
        unix.write_bytes(b'{"a": 1}\n')
        windows.write_bytes(b'{"a": 1}\r\n')
        assert hybrid.text_sha256(unix) == hybrid.text_sha256(windows)
