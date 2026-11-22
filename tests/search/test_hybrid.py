"""Tests for search/eval/hybrid.py: the rules of step A, the tuning and the test run."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from search.corpus.table import load_passage_table
from search.eval.baseline import FilterMasks, QuestionResult, speech_years
from search.eval.hybrid import (
    CandidateScore,
    HybridPoint,
    QueryVectors,
    choose_candidate,
    choose_hybrid,
    combined_mask,
    compare,
    decide,
    deviating_parties,
    fused_hits,
    lexical_hits,
    score_rankings,
    speech_parties,
    tune_grid,
    vector_hits,
)
from search.eval.questions import EvalQuestion, Provenance, QuestionFilter
from search.index.analyzer import Analyzer
from search.index.embed import HashEmbedder
from search.index.lexical import DEFAULT_PARAMS, build_index, passage_texts, tokenize_corpus
from search.index.vector import VectorIndex
from tests.search.corpus_kit import write_corpus

PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 22), generation=1
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
    """The synthetic corpus with its BM25 index, its vectors and three questions."""
    speeches_path, passages_path = write_corpus(tmp_path)
    table = load_passage_table(passages_path)
    texts = [text for _, text in passage_texts(speeches_path, passages_path)]
    lexical = build_index(
        tokenize_corpus(passage_texts(speeches_path, passages_path)),
        Analyzer(stem=True, stopwords=False),
    )
    embedder = HashEmbedder(dimension=256)
    index = VectorIndex(embedder.embed(texts, "document"), np.arange(len(table)))
    questions = [
        question(1, "Fler lärare och mindre klasser i skolan", "GR0950-1:0-1"),
        question(2, "Kortare köer i sjukvården", "HB0960-1:0-1"),
        question(3, "kärnkraften", "H209100-2:0-1", filter=QuestionFilter(party="MP")),
    ]
    queries = QueryVectors.of(
        [q.question_id for q in questions], embedder.embed([q.text for q in questions], "query")
    )
    return table, lexical, index, questions, queries


class TestRankings:
    def test_each_leg_finds_the_questions_speech(self, setting):
        table, lexical, index, questions, queries = setting
        masks = FilterMasks(table)
        years, parties = speech_years(table), speech_parties(table)
        for hits in (
            vector_hits(index, masks, questions, queries, 10),
            lexical_hits(lexical, DEFAULT_PARAMS, masks, questions, 10),
        ):
            results = score_rankings(table, questions, hits, years, parties)
            assert [r.ranking[0] for r in results] == ["GR0950-1", "HB0960-1", "H209100-2"]

    def test_a_filter_holds_in_both_legs_and_in_the_fusion(self, setting):
        table, lexical, index, questions, queries = setting
        masks = FilterMasks(table)
        filtered = [questions[2]]
        vector = vector_hits(index, masks, filtered, queries, 10)
        words = lexical_hits(lexical, DEFAULT_PARAMS, masks, filtered, 10)
        fused = fused_hits(words, vector, 60, 10)
        for ranking in (vector[0], words[0], fused[0]):
            assert {str(table.parties[p]) for p in ranking} == {"MP"}

    def test_a_subset_narrows_both_legs(self, setting):
        table, lexical, index, questions, queries = setting
        masks = FilterMasks(table)
        within = np.zeros(len(table), dtype=bool)
        within[[0, 2]] = True
        vector = vector_hits(index, masks, questions[:1], queries, 10, within)
        words = lexical_hits(lexical, DEFAULT_PARAMS, masks, questions[:1], 10, within)
        assert set(vector[0]) <= {0, 2} and set(words[0]) <= {0, 2}

    def test_a_question_is_counted_under_its_speechs_party_and_period(self, setting):
        table, lexical, index, questions, queries = setting
        results = score_rankings(
            table,
            questions[:1],
            vector_hits(index, FilterMasks(table), questions[:1], queries, 10),
            speech_years(table),
            speech_parties(table),
        )
        assert (results[0].party, results[0].period) == ("S", "2003-2009")
        assert results[0].metrics["recall@10"] == 1.0

    def test_combined_masks(self, setting):
        table, _, _, questions, _ = setting
        masks = FilterMasks(table)
        within = np.ones(len(table), dtype=bool)
        within[0] = False
        assert combined_mask(masks, questions[0], None) is None
        assert combined_mask(masks, questions[0], within) is within
        both = combined_mask(masks, questions[2], within)
        assert both.sum() == (table.parties == "MP").sum()

    def test_every_question_needs_one_vector(self):
        with pytest.raises(ValueError):
            QueryVectors.of(["a", "a"], np.zeros((2, 4), dtype=np.float32))
        with pytest.raises(KeyError):
            QueryVectors.of(["a"], np.zeros((1, 4))).matrix([question(9, "x", "Y-1:0-0")])


def score(name, recall, licensed=True, cost=None) -> CandidateScore:
    return CandidateScore(
        name=name, recall_10=recall, mrr_10=0.1, licensed=licensed, corpus_cost_usd=cost
    )


class TestChooseCandidate:
    def test_the_highest_recall_wins(self):
        choice = choose_candidate([score("small", 0.30), score("large", 0.40)], 0.01, 10.0)
        assert (choice.chosen, choice.best) == ("large", "large")

    def test_within_the_margin_the_lightest_wins(self):
        scores = [score("small", 0.392), score("swedish", 0.395), score("large", 0.40)]
        choice = choose_candidate(scores, 0.01, 10.0)
        assert choice.chosen == "small" and choice.tied == ("small", "swedish", "large")

    def test_a_difference_of_exactly_the_margin_is_a_tie(self):
        choice = choose_candidate([score("small", 0.39), score("large", 0.40)], 0.01, 10.0)
        assert choice.chosen == "small"

    def test_an_unlicensed_model_cannot_be_chosen(self):
        scores = [score("small", 0.30), score("swedish", 0.50, licensed=False)]
        choice = choose_candidate(scores, 0.01, 10.0)
        assert choice.chosen == "small" and "swedish" in choice.excluded

    def test_an_api_model_over_budget_cannot_be_chosen(self):
        scores = [score("small", 0.30), score("api", 0.50, cost=12.0)]
        choice = choose_candidate(scores, 0.01, 10.0)
        assert choice.chosen == "small" and "12.00 USD" in choice.excluded["api"]

    def test_an_api_model_within_budget_can_win(self):
        choice = choose_candidate([score("small", 0.30), score("api", 0.50, cost=2.12)], 0.01, 10.0)
        assert choice.chosen == "api"

    def test_no_eligible_candidate_raises(self):
        with pytest.raises(ValueError):
            choose_candidate([score("swedish", 0.5, licensed=False)], 0.01, 10.0)


def point(words, k, recall, mrr=0.1) -> HybridPoint:
    return HybridPoint(target_words=words, k=k, recall_10=recall, mrr_10=mrr)


class TestChooseHybrid:
    def test_the_highest_recall_wins(self):
        assert (
            choose_hybrid([point(200, 60, 0.4), point(120, 10, 0.5)], 200, 60).target_words == 120
        )

    def test_a_tie_goes_to_mrr_then_to_200_words_then_to_k_nearest_60(self):
        assert (
            choose_hybrid([point(200, 60, 0.4), point(320, 60, 0.4, 0.2)], 200, 60).target_words
            == 320
        )
        assert (
            choose_hybrid([point(320, 60, 0.4), point(200, 10, 0.4)], 200, 60).target_words == 200
        )
        chosen = choose_hybrid(
            [point(200, 30, 0.4), point(200, 100, 0.4), point(200, 60, 0.4)], 200, 60
        )
        assert chosen.k == 60
        assert choose_hybrid([point(200, 100, 0.4), point(200, 20, 0.4)], 200, 60).k == 20

    def test_the_grid_has_one_point_per_constant(self, setting):
        table, lexical, index, questions, queries = setting
        masks = FilterMasks(table)
        words = lexical_hits(lexical, DEFAULT_PARAMS, masks, questions, 10)
        vector = vector_hits(index, masks, questions, queries, 10)
        grid = tune_grid(table, questions, words, vector, 200, (10, 60), 10)
        assert [(p.target_words, p.k) for p in grid] == [(200, 10), (200, 60)]
        assert all(p.recall_10 == 1.0 for p in grid)


def results(values) -> list[QuestionResult]:
    return [
        QuestionResult(
            question_id=f"natural-{number:04d}",
            period="2010-2017",
            party="S",
            ranking=(),
            metrics={"recall@5": v, "recall@10": v, "recall@50": v, "mrr@10": v},
        )
        for number, v in enumerate(values)
    ]


class TestTestRun:
    def test_a_clear_win_gives_outcome_a(self):
        comparison = compare(
            "natural", "hybrid", results([1.0] * 40), "bm25", results([0.0] * 40), 1, 500
        )
        assert comparison.difference == 1.0
        assert decide(comparison).outcome == "A"

    def test_a_small_gain_gives_outcome_b(self):
        hybrid = results([1.0] * 2 + [0.0] * 98)
        bm25 = results([1.0] * 1 + [0.0] * 99)
        decision = decide(compare("natural", "hybrid", hybrid, "bm25", bm25, 1, 500))
        assert decision.outcome == "B" and decision.difference == pytest.approx(0.01)

    def test_a_gain_whose_interval_reaches_zero_gives_outcome_b(self):
        hybrid = results([1.0, 0.0] * 20)
        bm25 = results([0.0, 1.0] * 18 + [0.0, 0.0] * 2)
        comparison = compare("natural", "hybrid", hybrid, "bm25", bm25, 2, 2000)
        assert comparison.difference >= 0.03 and comparison.interval[0] <= 0
        assert decide(comparison).outcome == "B"

    def test_the_two_systems_must_answer_the_same_questions(self):
        with pytest.raises(ValueError):
            compare("natural", "hybrid", results([1.0]), "bm25", results([1.0, 0.0]), 1, 10)

    def test_parties_more_than_005_from_the_mean_are_named(self):
        named = deviating_parties({"S": 0.41, "M": 0.29, "V": 0.36}, 0.35)
        assert named == {"M": -0.06, "S": 0.06}

    def test_a_party_exactly_005_from_the_mean_is_not_named(self):
        assert deviating_parties({"S": 0.40, "M": 0.30}, 0.35) == {}
