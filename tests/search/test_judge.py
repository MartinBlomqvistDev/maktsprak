"""Tests for search/eval/judge.py."""

from __future__ import annotations

from collections import Counter

import pytest

from search.corpus.table import load_passage_table
from search.eval.generation import PROMPTS_DIR, load_prompt
from search.eval.judge import (
    GRADES,
    PilotQuestion,
    agreement,
    calibration_form,
    calibration_pairs,
    cohen_kappa,
    linear_weights,
    read_grade,
    weighted_kappa,
)
from search.eval.llm import Completion
from search.index.analyzer import Analyzer
from search.index.lexical import build_index, passage_texts, tokenize_corpus
from tests.search.corpus_kit import write_corpus


class TestKappa:
    def test_full_agreement_is_one(self):
        assert cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2], GRADES) == pytest.approx(1.0)

    def test_a_worked_example(self):
        first = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
        second = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
        assert cohen_kappa(first, second, (0, 1)) == pytest.approx(0.4)

    def test_one_grade_throughout_on_both_sides_is_one(self):
        assert cohen_kappa([2, 2, 2], [2, 2, 2], GRADES) == 1.0

    def test_linear_weights_give_part_credit_for_a_near_miss(self):
        first, second = [0, 1, 2, 2], [0, 2, 2, 1]
        unweighted = cohen_kappa(first, second, GRADES)
        weighted = weighted_kappa(first, second, GRADES, linear_weights(GRADES))
        assert weighted > unweighted
        assert linear_weights(GRADES)[0] == [1.0, 0.5, 0.0]

    def test_lists_of_different_length_are_refused(self):
        with pytest.raises(ValueError):
            cohen_kappa([0, 1], [0], GRADES)


class TestAgreement:
    def test_the_judge_is_accepted_at_the_threshold_and_ungraded_pairs_are_named(self):
        martin = {f"p{n}": n % 3 for n in range(30)}
        judge = dict(martin)
        judge["p0"] = None
        result = agreement(martin, judge)
        assert result.accepted and result.kappa == pytest.approx(1.0)
        assert result.pairs == 29 and result.ungraded == ("p0",)

    def test_disagreement_is_counted_in_the_table(self):
        martin = {"a": 2, "b": 0, "c": 1, "d": 2}
        judge = {"a": 2, "b": 1, "c": 1, "d": 0}
        result = agreement(martin, judge)
        assert result.table == {"0-1": 1, "1-1": 1, "2-0": 1, "2-2": 1}
        assert not result.accepted
        assert result.exact == 0.5

    def test_different_pairs_are_refused(self):
        with pytest.raises(ValueError):
            agreement({"a": 1}, {"b": 1})


class TestGrades:
    def test_the_judges_prompt_holds_the_question_the_passage_and_the_three_grades(self):
        prompt = load_prompt("judge", PROMPTS_DIR)
        assert "{question}" in prompt.user and "{passage}" in prompt.user
        assert all(f"{grade}:" in prompt.system for grade in GRADES)
        assert "—" not in prompt.system + prompt.user

    def test_a_grade_is_read_from_the_answer(self):
        good = Completion(custom_id="x", stop_reason="end_turn", text='{"reason": "r", "grade": 2}')
        assert read_grade(good) == (2, "r")

    @pytest.mark.parametrize(
        "completion",
        [
            Completion(custom_id="x", error="HTTP 529"),
            Completion(custom_id="x", stop_reason="refusal"),
            Completion(custom_id="x", stop_reason="end_turn", text='{"grade": 3}'),
            Completion(custom_id="x", stop_reason="end_turn", text="{"),
        ],
    )
    def test_anything_else_has_no_grade(self, completion):
        assert read_grade(completion)[0] is None


class TestPairs:
    @pytest.fixture
    def setting(self, tmp_path):
        speeches_path, passages_path = write_corpus(tmp_path)
        table = load_passage_table(passages_path)
        texts = dict(passage_texts(speeches_path, passages_path))
        index = build_index(tokenize_corpus(texts.items()), Analyzer(stem=True, stopwords=True))
        questions = [
            PilotQuestion(
                question_id="pilot-natural-0001",
                text="Ska kärnkraften byggas ut?",
                passage_id="H209100-1:0-1",
            ),
            PilotQuestion(
                question_id="pilot-natural-0002",
                text="Behöver skolan fler lärare?",
                passage_id="GR0950-1:0-1",
            ),
        ]
        return questions, index, table, texts

    def test_each_question_gets_its_own_passage_and_bm25s_next(self, setting):
        questions, index, table, _ = setting
        pairs = calibration_pairs(questions, index, table, seed=1, count=2, per_question=3)
        assert len(pairs) <= 6
        by_question = Counter(pair.question_id for pair in pairs)
        assert set(by_question) == {"pilot-natural-0001", "pilot-natural-0002"}
        own = [pair for pair in pairs if pair.source == "written_from"]
        assert {pair.passage_id for pair in own} == {"H209100-1:0-1", "GR0950-1:0-1"}
        for pair in pairs:
            if pair.source != "written_from":
                question = next(q for q in questions if q.question_id == pair.question_id)
                assert pair.passage_id != question.passage_id
        assert [pair.pair_id for pair in pairs] == [
            f"pair-{n:03d}" for n in range(1, len(pairs) + 1)
        ]

    def test_the_form_names_every_pair(self, setting):
        questions, index, table, texts = setting
        pairs = calibration_pairs(questions, index, table, seed=1, count=2, per_question=2)
        texts = {key: value.replace("\n", "\n\n") for key, value in texts.items()}
        page = calibration_form(pairs, texts, "kalibrering-test")
        assert all(pair.pair_id in page for pair in pairs)
        assert "2: besvarar frågan" in page and "__ITEMS__" not in page
