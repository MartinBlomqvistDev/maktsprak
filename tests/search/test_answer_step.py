"""Tests for search/pipeline/14_run_answer_step.py, on the synthetic corpus and without the API.

The message batches are replaced by a stand-in, so the whole step runs here:
candidates, the baseline, round 1 with three models and a request that is sent
once more, the judge, the report with the choice of model, and round 2.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from search.answer.candidates import load_speeches, rank_passages
from search.answer.selector import load_selector_prompt
from search.config import Settings, load_settings
from search.corpus.table import load_passage_table
from search.eval.generation import append_log
from search.eval.llm import Completion, Usage
from search.eval.questions import EvalQuestion, Provenance
from search.index.analyzer import Analyzer
from search.index.lexical import (
    Bm25Params,
    IndexManifest,
    build_index,
    file_sha256,
    lexical_dir,
    passage_ids_sha256,
    passage_texts,
    tokenize_corpus,
)
from tests.search.corpus_kit import write_corpus
from tests.search.test_answer import CORPUS, ENERGY


def load_step():
    """Import the script, whose name is not a module name, registered before it runs."""
    path = Path(__file__).resolve().parents[2] / "search" / "pipeline" / "14_run_answer_step.py"
    spec = importlib.util.spec_from_file_location("answer_step", path)
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


step = load_step()

OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"
STEMMED = Analyzer(stem=True, stopwords=False)
PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 22), generation=1
)
QUESTIONS = [
    EvalQuestion(
        question_id="natural-0001",
        question_set="natural",
        split="dev",
        text="Ska kärnkraften byggas ut?",
        gold=("H209100-1",),
        passage_id=ENERGY,
        provenance=PROVENANCE,
    ),
    EvalQuestion(
        question_id="natural-0002",
        question_set="natural",
        split="dev",
        text="Vad vill partierna göra för lärarna i skolan?",
        gold=("GR0950-1",),
        passage_id="GR0950-1:0-1",
        provenance=PROVENANCE,
    ),
    EvalQuestion(
        question_id="unanswerable-0001",
        question_set="unanswerable",
        split="dev",
        text="Vem vann fotbollsmatchen?",
        gold=(),
        kind="off_topic",
        provenance=PROVENANCE,
    ),
    EvalQuestion(
        question_id="unanswerable-0002",
        question_set="unanswerable",
        split="dev",
        text="Vad sades om kärnkraften år 2031?",
        gold=(),
        kind="after_index",
        provenance=PROVENANCE,
    ),
]


@pytest.fixture
def settings(tmp_path: Path, monkeypatch) -> Settings:
    """Settings whose bulk data and reports live in *tmp_path*, over the synthetic corpus."""
    loaded = load_settings(data_root=tmp_path)
    reports = tmp_path / "reports"
    settings = loaded.model_copy(
        update={"tracked": loaded.tracked.model_copy(update={"reports": reports})}
    )
    search = settings.data.search
    search.mkdir(parents=True)
    speeches_path, passages_path = write_corpus(search, CORPUS)
    table = load_passage_table(passages_path)
    index = build_index(tokenize_corpus(passage_texts(speeches_path, passages_path)), STEMMED)
    index.save(
        lexical_dir(search, STEMMED),
        IndexManifest(
            analyzer=STEMMED,
            passages=index.passages,
            passage_ids_sha256=passage_ids_sha256(table.passage_ids),
            passages_sha256=file_sha256(passages_path),
            speeches_sha256=file_sha256(speeches_path),
            terms=len(index.terms),
            postings=len(index.doc_ids),
            tokens=int(index.doc_lengths.sum()),
            mean_passage_terms=round(index.mean_length, 3),
        ),
    )
    reports.mkdir()
    (reports / step.TUNING_REPORT).write_text(
        json.dumps({"chosen": {"analyzer": "stem-nostop", "k1": 1.2, "b": 0.9}}), encoding="utf-8"
    )
    agree(settings, accepted=True)
    monkeypatch.setattr(step, "dev_questions", lambda _: QUESTIONS)
    return settings


def agree(settings: Settings, accepted: bool | None) -> None:
    """Record the judge's agreement with the hand grades, or remove it when *accepted* is None."""
    path = settings.tracked.reports / step.AGREEMENT_REPORT
    if accepted is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(json.dumps({"agreement": {"accepted": accepted}}), encoding="utf-8")


def use_new_prompt(monkeypatch, directory: Path) -> None:
    """Point the step at a copy of the selector's prompt with a line added, as a new round would."""
    directory.mkdir()
    source = step.PROMPT_DIR / f"{step.PROMPT_NAME}.md"
    (directory / source.name).write_text(
        source.read_text(encoding="utf-8") + "\nEn rad till.\n", encoding="utf-8"
    )
    monkeypatch.setattr(step, "PROMPT_DIR", directory)
    monkeypatch.setattr(step, "load_selector_prompt", lambda: load_selector_prompt(directory))


MANIFEST = SimpleNamespace(
    sets={name: SimpleNamespace(sha256=f"{name}-hash") for name in step.ANSWER_SETS}
)


class FakeBatches:
    """Stands in for ``run_batch``: answers each request by a rule, and records the batches."""

    def __init__(self, answer):
        self.answer = answer
        self.batches: list[list[str]] = []
        self.models: list[set[str]] = []

    def __call__(self, requests, on_submitted=None, batch_id=None, poll_seconds=30):
        batch = f"msgbatch_{len(self.batches) + 1}"
        if on_submitted is not None:
            on_submitted(batch)
        self.batches.append(list(requests))
        self.models.append({params["model"] for params in requests.values()})
        return {
            custom_id: self.answer(custom_id, params, len(self.batches)).model_copy(
                update={"custom_id": custom_id, "batch_id": batch}
            )
            for custom_id, params in requests.items()
        }


def reply(text: str, stop: str = "end_turn") -> Completion:
    return Completion(
        custom_id="",
        model="claude-opus-5-5",
        stop_reason=stop,
        text=text,
        usage=Usage(input_tokens=7000, output_tokens=500),
    )


def selector_reply(custom_id: str, params: dict, batch: int) -> Completion:
    """Quote the first paragraph of the first candidate; refuse natural-0002 the first time."""
    if custom_id.startswith("unanswerable"):
        return reply(json.dumps({"answerable": False, "selections": []}))
    if custom_id == "natural-0002" and batch == 1:
        return reply("", stop="refusal")
    content = params["messages"][0]["content"]
    first_id = content.split('<utdrag id="', 1)[1].split('"', 1)[0]
    return reply(
        json.dumps(
            {
                "answerable": True,
                "selections": [{"passage_id": first_id, "first_paragraph": 1, "last_paragraph": 1}],
            }
        )
    )


def judge_reply(custom_id: str, params: dict, batch: int) -> Completion:
    return reply(json.dumps({"reason": "Svarar.", "grade": 2}))


class TestRegistration:
    def test_the_step_waits_for_the_addendum(self, monkeypatch):
        text = (Path(__file__).resolve().parents[2] / step.PREREGISTRATION).read_text("utf-8")
        without = text.replace(step.ADDENDUM, "")
        monkeypatch.setattr(step, "committed_preregistration", lambda: without)
        with pytest.raises(SystemExit, match="addendum on the answer step"):
            step.check_registered(load_settings())

    def test_the_step_waits_for_the_addendum_on_three_models(self, monkeypatch):
        text = (Path(__file__).resolve().parents[2] / step.PREREGISTRATION).read_text("utf-8")
        without = text.replace(step.MODELS_ADDENDUM, "")
        monkeypatch.setattr(step, "committed_preregistration", lambda: without)
        with pytest.raises(SystemExit, match="addendum on three models"):
            step.check_registered(load_settings())

    def test_the_committed_sets_and_addendum_pass(self, monkeypatch):
        text = (Path(__file__).resolve().parents[2] / step.PREREGISTRATION).read_text("utf-8")
        monkeypatch.setattr(step, "committed_preregistration", lambda: text)
        manifest = step.check_registered(load_settings())
        assert set(step.ANSWER_SETS) <= set(manifest.sets)

    def test_the_models_of_round_1_are_priced_and_go_cheapest_first(self):
        settings = load_settings()
        assert step.by_price(settings, step.ROUND_ONE_MODELS) == [HAIKU, SONNET, OPUS]
        assert step.WITHOUT_THE_JUDGE == OPUS


class TestCost:
    def test_tokens_grow_with_the_words_beyond_the_judges_own(self):
        assert step.estimate_tokens(100, 100) == step.TOKENS_OVERHEAD
        assert step.estimate_tokens(200, 100) == round(step.TOKENS_OVERHEAD + 100 * 2.63)

    def test_a_batch_beyond_the_budget_is_stopped(self, settings, monkeypatch):
        monkeypatch.setattr(step, "spent", lambda _: {"round-1-opus-5-5": 37.0})
        step.check_budget(settings, 0.5)
        with pytest.raises(SystemExit, match="beyond the budget"):
            step.check_budget(settings, 1.5)

    def test_a_lower_cap_from_the_command_line_stops_a_batch_sooner(self, settings, monkeypatch):
        monkeypatch.setattr(step, "spent", lambda _: {"round-1-opus-5-5": 20.0})
        step.check_budget(settings, 4.0, cap_usd=25.0)
        with pytest.raises(SystemExit, match="beyond the budget of 25.00 USD"):
            step.check_budget(settings, 6.0, cap_usd=25.0)
        with pytest.raises(SystemExit, match="beyond the budget of 38.00 USD"):
            step.check_budget(settings, 19.0, cap_usd=100.0)

    def test_each_round_is_priced_at_its_own_models_price(self, settings):
        for model in (OPUS, HAIKU):
            directory = step.round_dir(settings, step.run_name(1, model))
            directory.mkdir(parents=True)
            record = step.RoundRecord(
                number=1,
                model=model,
                effort="low",
                max_tokens=8000,
                prompt_sha256="p" * 64,
                created=step.now(),
            )
            step.write_report(record, directory / step.ROUND_NAME)
            usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
            completion = Completion(custom_id="q", usage=usage)
            append_log(directory / "completions_first.jsonl", [completion])
        assert step.spent(settings) == {"round-1-haiku-5-5": 0.075, "round-1-opus-5-5": 3.0}


class TestTheStep:
    def test_candidates_baseline_three_models_judge_report_and_round_2(
        self, settings, monkeypatch, tmp_path
    ):
        step.make_candidates(settings, MANIFEST)
        run, records = step.read_candidates(settings)
        assert run.analyzer == "stem-nostop" and run.questions == 4
        by_id = {record.question_id: record for record in records}
        assert by_id["unanswerable-0001"].candidates == ()
        assert by_id["natural-0001"].candidates[0].passage_id == ENERGY

        step.run_baseline(settings)
        baseline = step.read_answers(settings, "baseline")
        assert baseline["unanswerable-0001"].abstained

        selector_batches = FakeBatches(selector_reply)
        monkeypatch.setattr(step, "run_batch", selector_batches)
        step.run_round(settings, 1, HAIKU)
        assert len(selector_batches.batches) == 2
        assert selector_batches.batches[1] == ["natural-0002"]
        haiku = step.run_name(1, HAIKU)
        record = json.loads((step.round_dir(settings, haiku) / step.ROUND_NAME).read_text("utf-8"))
        assert record["batches"] == {"first": "msgbatch_1", "again": "msgbatch_2"}
        assert (record["model"], record["effort"], record["max_tokens"]) == (HAIKU, "low", 8000)
        assert (step.round_dir(settings, haiku) / "prompt.md").is_file()
        answers = step.read_answers(settings, haiku)
        assert answers["natural-0002"].selected.attempts == 2
        assert answers["natural-0002"].selected.fallback is None
        assert [quote.passage_id for quote in answers["natural-0001"].quotes] == [ENERGY]
        assert answers["unanswerable-0002"].abstained
        with pytest.raises(SystemExit, match="already answered"):
            step.run_round(settings, 1, HAIKU)
        with pytest.raises(SystemExit, match="not chosen yet"):
            step.run_round(settings, 2, HAIKU)

        step.run_round(settings, 1, SONNET)
        step.run_round(settings, 1, OPUS)
        assert selector_batches.models == [{HAIKU}, {HAIKU}, {SONNET}, {OPUS}]
        with pytest.raises(SystemExit, match="not chosen yet"):
            step.run_round(settings, 2, HAIKU)

        judge_batches = FakeBatches(judge_reply)
        monkeypatch.setattr(step, "run_batch", judge_batches)
        step.run_judge(settings)
        candidate_texts = {
            "\n\n".join(candidate.paragraphs)
            for record in records
            if record.answerable
            for candidate in record.candidates
        }
        pairs = step.grading_pairs(settings, records)
        assert candidate_texts <= {pair.text for pair in pairs}
        assert len(judge_batches.batches) == 1

        step.make_report(settings)
        report = json.loads((settings.tracked.reports / step.REPORT).read_text("utf-8"))
        names = [step.run_name(1, model) for model in (HAIKU, SONNET, OPUS)]
        assert [item["name"] for item in report["selectors"]] == ["baseline", *names]
        first = report["selectors"][1]
        assert first["model"] == HAIKU and first["pending_grades"] == 0
        assert first["quote_precision"] == 1.0 and first["score"] == 1.0
        assert first["abstention"]["abstains_unanswerable"] == 1.0
        assert first["cost_per_question_usd"] > 0
        assert [item["name"] for item in report["comparisons"]] == names
        assert set(report["cost_usd"]) == {*names, "judge"}
        assert report["chosen_model"] == HAIKU
        assert report["leading_round"] == 1
        lines = (settings.tracked.reports / step.QUESTIONS_REPORT).read_text("utf-8").splitlines()
        assert len(lines) == 4 * len(QUESTIONS)
        assert {json.loads(line)["run"] for line in lines} == {"baseline", *names}

        monkeypatch.setattr(step, "run_batch", FakeBatches(selector_reply))
        with pytest.raises(SystemExit, match="only the chosen model"):
            step.run_round(settings, 2, OPUS)
        with pytest.raises(SystemExit, match="already ran this prompt"):
            step.run_round(settings, 2, HAIKU)
        with pytest.raises(SystemExit, match="has no answers yet"):
            step.run_round(settings, 3, HAIKU)
        use_new_prompt(monkeypatch, tmp_path / "prompt-2")
        step.run_round(settings, 2, HAIKU)
        second = json.loads(
            (step.round_dir(settings, step.run_name(2, HAIKU)) / step.ROUND_NAME).read_text("utf-8")
        )
        assert second["prompt_sha256"] != record["prompt_sha256"]
        assert step.answers_path(settings, step.run_name(2, HAIKU)).is_file()

    def test_a_round_picks_up_a_batch_already_answered(self, settings, monkeypatch):
        step.make_candidates(settings, MANIFEST)
        step.run_baseline(settings)
        batches = FakeBatches(selector_reply)
        monkeypatch.setattr(step, "run_batch", batches)
        step.run_round(settings, 1, OPUS)
        step.answers_path(settings, step.run_name(1, OPUS)).unlink()
        step.run_round(settings, 1, OPUS)
        assert len(batches.batches) == 2

    def test_nothing_paid_runs_before_the_judges_agreement_is_known(self, settings, monkeypatch):
        step.make_candidates(settings, MANIFEST)
        step.run_baseline(settings)
        agree(settings, accepted=None)
        monkeypatch.setattr(step, "run_batch", FakeBatches(selector_reply))
        with pytest.raises(SystemExit, match="not known yet"):
            step.run_round(settings, 1, OPUS)
        with pytest.raises(SystemExit, match="not known yet"):
            step.run_judge(settings)

    def test_when_the_judge_falls_short_opus_runs_alone(self, settings, monkeypatch):
        step.make_candidates(settings, MANIFEST)
        step.run_baseline(settings)
        agree(settings, accepted=False)
        monkeypatch.setattr(step, "run_batch", FakeBatches(selector_reply))
        with pytest.raises(SystemExit, match="runs claude-opus-5-5 alone"):
            step.run_round(settings, 1, HAIKU)
        step.run_round(settings, 1, OPUS)
        step.make_report(settings)
        report = json.loads((settings.tracked.reports / step.REPORT).read_text("utf-8"))
        assert report["judge_accepted"] is False
        assert report["chosen_model"] == OPUS and report["leading_round"] == 1

    def test_round_1_runs_every_model_with_the_same_prompt(self, settings, monkeypatch, tmp_path):
        step.make_candidates(settings, MANIFEST)
        step.run_baseline(settings)
        monkeypatch.setattr(step, "run_batch", FakeBatches(selector_reply))
        step.run_round(settings, 1, HAIKU)
        use_new_prompt(monkeypatch, tmp_path / "changed")
        with pytest.raises(SystemExit, match="same prompt"):
            step.run_round(settings, 1, SONNET)

    def test_every_round_runs_at_the_registered_effort_and_limit(self, settings, monkeypatch):
        step.make_candidates(settings, MANIFEST)
        step.run_baseline(settings)
        monkeypatch.setattr(step, "run_batch", FakeBatches(selector_reply))
        for change in ({"effort": "medium"}, {"max_tokens": 16000}):
            changed = settings.model_copy(
                update={"answer": settings.answer.model_copy(update=change)}
            )
            with pytest.raises(SystemExit, match="Every round runs at low effort"):
                step.run_round(changed, 1, OPUS)

    def test_the_report_gives_no_precision_before_the_judge(self, settings):
        step.make_candidates(settings, MANIFEST)
        step.run_baseline(settings)
        step.make_report(settings)
        report = json.loads((settings.tracked.reports / step.REPORT).read_text("utf-8"))
        [baseline] = report["selectors"]
        assert baseline["quote_precision"] is None and baseline["pending_grades"] > 0


def test_the_candidates_are_the_ones_bm25_ranks(settings):
    step.make_candidates(settings, MANIFEST)
    _, records = step.read_candidates(settings)
    index, _, table = step.load_retrieval(settings)
    speeches = load_speeches(settings.data.search / "anforanden.parquet", ["H209100-1"])
    hits = rank_passages(index, QUESTIONS[0].text, Bm25Params(k1=1.2, b=0.9))
    expected = [table.passage_ids[hit.passage] for hit in hits]
    assert [candidate.passage_id for candidate in records[0].candidates] == expected
    assert speeches["H209100-1"].paragraphs == tuple(CORPUS[0].paragraphs)
