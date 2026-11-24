"""Tests for search/eval/hand_grades.py: the draw, the pairs' order, the form and the grades."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pytest

from search.eval.hand_grades import (
    HandDraw,
    draw_questions,
    hand_form,
    name_pairs,
    order_pairs,
    read_hand_grades,
)
from search.eval.judge import GRADING_INTRO
from search.eval.quote_judge import GradingPair

FORM = "bedomning-test"


def pair(question_id: str, text: str) -> GradingPair:
    return GradingPair(question_id=question_id, question=f"Fråga {question_id}?", text=text)


class TestDraw:
    def test_the_draw_depends_on_the_seed_alone(self):
        ids = [f"natural-{number:04d}" for number in range(300)]
        drawn = draw_questions(ids, seed=20261124, size=100)
        assert drawn == draw_questions(reversed(ids), seed=20261124, size=100)
        assert drawn == sorted(drawn) and len(set(drawn)) == 100 and set(drawn) <= set(ids)
        assert drawn != draw_questions(ids, seed=1, size=100)

    def test_too_few_questions_raise(self):
        with pytest.raises(ValueError, match="2 answerable questions"):
            draw_questions(["a", "b", "a"], seed=1, size=3)


class TestPairs:
    def test_each_pair_comes_once_in_an_order_the_seed_draws(self):
        pairs = [pair("q1", "Ett."), pair("q2", "Två."), pair("q1", "Ett."), pair("q3", "Tre.")]
        ordered = order_pairs(pairs, seed=7)
        assert len(ordered) == 3
        assert ordered == order_pairs(list(reversed(pairs)), seed=7)
        names = name_pairs(ordered)
        assert [name.pair_id for name in names] == ["pair-001", "pair-002", "pair-003"]
        assert [name.key for name in names] == [item.key for item in ordered]


class TestForm:
    def test_the_form_shows_each_question_and_quotation_under_the_calibrations_words(self):
        ordered = order_pairs([pair("q1", "Ett stycke.\n\nEtt till."), pair("q2", "Två.")], seed=3)
        names = name_pairs(ordered)
        page = hand_form(ordered, names, FORM)
        assert all(name.pair_id in page for name in names)
        assert "<p>Ett stycke.</p><p>Ett till.</p>" in page and "Fråga q2?" in page
        assert "2: besvarar frågan" in page and html.escape(GRADING_INTRO) in page
        assert "__ITEMS__" not in page and json.dumps(FORM) in page

    def test_names_that_are_not_the_pairs_raise(self):
        ordered = [pair("q1", "Ett."), pair("q2", "Två.")]
        with pytest.raises(ValueError, match="not those of its pairs"):
            hand_form(ordered, name_pairs(list(reversed(ordered))), FORM)


class TestGrades:
    @pytest.fixture
    def draw(self) -> HandDraw:
        ordered = order_pairs([pair("q1", "Ett."), pair("q2", "Två.")], seed=3)
        return HandDraw(
            seed=3,
            run_sha256="r",
            answers_sha256={},
            questions=("q1", "q2"),
            pairs=name_pairs(ordered),
        )

    @staticmethod
    def write(tmp_path: Path, answers: list[dict], form: str = FORM) -> Path:
        path = tmp_path / "answers.json"
        path.write_text(json.dumps({"form": form, "answers": answers}), encoding="utf-8")
        return path

    def test_the_grades_are_read_by_each_pairs_key(self, tmp_path, draw):
        answers = [
            {"question_id": named.pair_id, "grade": grade, "comment": ""}
            for named, grade in zip(draw.pairs, (2, 0), strict=True)
        ]
        grades = read_hand_grades(self.write(tmp_path, answers), draw, FORM)
        assert grades == {draw.pairs[0].key: 2, draw.pairs[1].key: 0}

    @pytest.mark.parametrize("grade", [None, 3, -1, True, "2"])
    def test_a_pair_without_a_grade_of_0_1_or_2_raises(self, tmp_path, draw, grade):
        answers = [
            {"question_id": draw.pairs[0].pair_id, "grade": 2},
            {"question_id": draw.pairs[1].pair_id, "grade": grade},
        ]
        with pytest.raises(ValueError, match="Without a grade"):
            read_hand_grades(self.write(tmp_path, answers), draw, FORM)

    def test_answers_from_another_form_or_naming_another_pair_raise(self, tmp_path, draw):
        answers = [{"question_id": named.pair_id, "grade": 2} for named in draw.pairs]
        with pytest.raises(ValueError, match="come from the form"):
            read_hand_grades(self.write(tmp_path, answers, "kalibrering-av-domaren"), draw, FORM)
        answers.append({"question_id": "pair-999", "grade": 1})
        with pytest.raises(ValueError, match="Unknown"):
            read_hand_grades(self.write(tmp_path, answers), draw, FORM)
