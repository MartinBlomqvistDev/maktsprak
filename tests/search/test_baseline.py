"""Tests for search/eval/baseline.py."""

from __future__ import annotations

from datetime import date

import pytest

from search.corpus.table import load_passage_table
from search.eval.baseline import (
    GridPoint,
    choose,
    run_set,
    speech_years,
    summarize_set,
    tune,
)
from search.eval.questions import EvalQuestion, Provenance, QuestionFilter
from search.index.analyzer import ANALYZERS
from search.index.lexical import DEFAULT_PARAMS, build_index, passage_texts, tokenize_corpus
from tests.search.corpus_kit import write_corpus

PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 22), generation=1
)


def point(analyzer="stem-stop", k1=1.2, b=0.75, recall=0.5, mrr=0.3) -> GridPoint:
    return GridPoint(analyzer=analyzer, k1=k1, b=b, recall_10=recall, mrr_10=mrr)


class TestChoose:
    def test_the_highest_recall_wins(self):
        assert choose([point(recall=0.5), point(k1=0.6, recall=0.6)]).k1 == 0.6

    def test_a_tie_goes_to_the_higher_mrr(self):
        assert choose([point(mrr=0.3), point(k1=0.6, mrr=0.4)]).k1 == 0.6

    def test_then_to_the_nearest_defaults(self):
        chosen = choose([point(k1=0.6), point(k1=1.5, b=0.75), point(k1=1.2, b=0.9)])
        assert (chosen.k1, chosen.b) == (1.2, 0.9)

    def test_then_to_the_analyzer_listed_first(self):
        assert choose([point(analyzer="nostem-stop"), point(analyzer="stem-nostop")]).analyzer == (
            "stem-nostop"
        )


def question(number: int, text: str, passage_id: str, **extra) -> EvalQuestion:
    name = "filtered" if "filter" in extra else "natural"
    return EvalQuestion(
        question_id=f"{name}-{number:04d}",
        question_set=name,
        split="dev",
        text=text,
        gold=(passage_id.split(":")[0],),
        passage_id=passage_id,
        provenance=PROVENANCE,
        **extra,
    )


@pytest.fixture
def setting(tmp_path):
    speeches_path, passages_path = write_corpus(tmp_path)
    table = load_passage_table(passages_path)
    corpus = tokenize_corpus(passage_texts(speeches_path, passages_path))
    indexes = {analyzer.name: build_index(corpus, analyzer) for analyzer in ANALYZERS}
    return table, indexes


class TestRuns:
    def test_the_tuning_tries_every_configuration(self, setting):
        table, indexes = setting
        questions = [question(1, "Ska kärnkraften byggas ut?", "H209100-1:0-1")]
        grid = tune(indexes, table, questions, [0.9, 1.2], [0.5, 0.75])
        assert len(grid) == 4 * 2 * 2
        assert all(0 <= p.recall_10 <= 1 for p in grid)

    def test_a_question_finds_its_speech(self, setting):
        table, indexes = setting
        questions = [question(1, "Fler lärare och mindre klasser i skolan", "GR0950-1:0-1")]
        results = run_set(
            indexes["stem-stop"], table, questions, DEFAULT_PARAMS, speech_years(table)
        )
        assert results[0].ranking[0] == "GR0950-1"
        assert results[0].metrics["mrr@10"] == 1.0
        assert (results[0].period, results[0].party) == ("2003-2009", "S")

    def test_a_filter_keeps_other_parties_out(self, setting):
        table, indexes = setting
        filtered = question(1, "kärnkraften", "H209100-2:0-1", filter=QuestionFilter(party="MP"))
        results = run_set(
            indexes["stem-stop"], table, [filtered], DEFAULT_PARAMS, speech_years(table)
        )
        assert results[0].ranking == ("H209100-2",)

    def test_a_set_is_summarized_overall_by_period_and_by_party(self, setting):
        table, indexes = setting
        questions = [
            question(1, "Fler lärare och mindre klasser i skolan", "GR0950-1:0-1"),
            question(2, "Kortare köer i sjukvården", "HB0960-1:0-1"),
        ]
        results = run_set(
            indexes["stem-stop"], table, questions, DEFAULT_PARAMS, speech_years(table)
        )
        summary = summarize_set(results, seed=1, resamples=200)
        assert summary.overall.questions == 2
        assert set(summary.by_period) == {"2003-2009", "2018-2023"}
        assert set(summary.by_party) == {"S", "SD"}
        low, high = summary.overall.intervals["recall@10"]
        assert low <= summary.overall.means["recall@10"] <= high
