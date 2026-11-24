"""Tests for search/eval/review.py."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import pytest

from search.eval.questions import EvalQuestion, Provenance, QuestionFilter
from search.eval.review import ReviewItem, draw_review, review_form, tally

PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 22), generation=1
)


def questions(name: str, count: int) -> list[EvalQuestion]:
    made = []
    for number in range(1, count + 1):
        extra = {"filter": QuestionFilter(party="S")} if name == "filtered" else {}
        if name == "stance":
            extra = {"kind": "for"}
        made.append(
            EvalQuestion(
                question_id=f"{name}-{number:04d}",
                question_set=name,
                split="dev",
                text=f"Fråga {number}?",
                gold=(f"H{number}-1",),
                passage_id=f"H{number}-1:0-0",
                provenance=PROVENANCE,
                **extra,
            )
        )
    return made


@pytest.fixture
def items() -> list[ReviewItem]:
    sets = {name: questions(name, 30) for name in ("natural", "keyword", "filtered", "stance")}
    passages = {f"H{number}-1:0-0": f"Stycke ett.\n\nStycke {number}." for number in range(31)}
    return draw_review(sets, passages, seed=5)


def answers(items, approve: int) -> str:
    entries = [
        {
            "question_id": item.question_id,
            "reasonable": True,
            "answered": number < approve,
            "comment": "",
        }
        for number, item in enumerate(items)
    ]
    return json.dumps({"form": "x", "answers": entries})


class TestDraw:
    def test_fifty_are_drawn_in_proportion_to_the_sets(self, items):
        assert len(items) == 50
        assert Counter(item.question_set for item in items) == {
            "natural": 20,
            "keyword": 20,
            "filtered": 7,
            "stance": 3,
        }
        assert [item.item for item in items] == list(range(1, 51))
        assert {item.filter for item in items if item.question_set == "filtered"} == {"parti S"}

    def test_the_seed_decides_the_draw(self, items):
        sets = {name: questions(name, 30) for name in ("natural", "keyword", "filtered", "stance")}
        passages = {f"H{number}-1:0-0": "x" for number in range(31)}
        again = draw_review(sets, passages, seed=5)
        assert [item.question_id for item in again] == [item.question_id for item in items]


class TestTally:
    def test_43_of_50_pass_and_42_do_not(self, items):
        passing = tally(items, answers(items, 43), 1, date(2026, 11, 22))
        assert (passing.approved, passing.passed) == (43, True)
        failing = tally(items, answers(items, 42), 1, date(2026, 11, 22))
        assert (failing.approved, failing.passed) == (42, False)
        assert sum(total for _, total in failing.by_set.values()) == 50

    def test_answers_for_other_questions_are_refused(self, items):
        broken = json.loads(answers(items, 50))
        broken["answers"][0]["question_id"] = "natural-9999"
        with pytest.raises(ValueError, match="do not match"):
            tally(items, json.dumps(broken), 1, date(2026, 11, 22))


class TestForm:
    def test_the_form_names_every_question_and_leaves_no_marker(self, items):
        page = review_form(items, "granskning-test")
        assert all(item.question_id in page for item in items)
        assert "__" not in page.replace("__proto__", "")
        assert "Stycke ett." in page

    def test_the_answers_leave_as_text_without_a_dialog(self, items):
        page = review_form(items, "granskning-test")
        assert 'id="dump"' in page and 'id="nosave"' in page
        assert "prompt(" not in page and "confirm(" not in page
