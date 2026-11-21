"""Tests for search/eval/questions.py."""

from __future__ import annotations

import hashlib
from datetime import date

import pytest
from pydantic import ValidationError

from search.eval.questions import (
    EvalQuestion,
    Provenance,
    QuestionFilter,
    read_question_set,
    set_file_name,
    split_of,
    write_question_set,
)

PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 21), generation=1
)


def natural(number: int = 1, split: str = "dev", **changes) -> EvalQuestion:
    """A valid natural question, with any field changed."""
    fields = {
        "question_id": f"natural-{number:04d}",
        "question_set": "natural",
        "split": split,
        "text": "Varför behövs fler lärare?",
        "gold": ("H209100-1",),
        "passage_id": "H209100-1:0-1",
        "provenance": PROVENANCE,
    }
    fields.update(changes)
    return EvalQuestion(**fields)


class TestRules:
    def test_a_natural_question_is_valid(self):
        assert natural().gold == ("H209100-1",)

    def test_the_gold_speech_must_be_the_passages_own(self):
        with pytest.raises(ValidationError, match="passage's own"):
            natural(gold=("H209100-2",))

    def test_the_key_must_start_with_the_set(self):
        with pytest.raises(ValidationError, match="start with"):
            natural(question_id="keyword-0001")

    def test_a_generated_question_needs_its_provenance(self):
        with pytest.raises(ValidationError, match="provenance"):
            natural(provenance=None)

    def test_text_on_two_lines_is_refused(self):
        with pytest.raises(ValidationError, match="one line"):
            natural(text="En\nfråga?")

    def test_a_filtered_question_needs_exactly_one_filter(self):
        question = natural(
            question_id="filtered-0001", question_set="filtered", filter=QuestionFilter(party="S")
        )
        assert question.filter.party == "S"
        with pytest.raises(ValidationError, match="carries a filter"):
            natural(question_id="filtered-0002", question_set="filtered")
        with pytest.raises(ValidationError):
            QuestionFilter(party="S", period="2010-2017")
        with pytest.raises(ValidationError):
            QuestionFilter(party="X")

    def test_an_unanswerable_question_has_no_gold_and_a_kind(self):
        question = EvalQuestion(
            question_id="unanswerable-0001",
            question_set="unanswerable",
            split="test",
            text="Hur kokar man ägg?",
            gold=(),
            kind="off_topic",
            provenance=PROVENANCE,
        )
        assert question.gold == ()
        with pytest.raises(ValidationError, match="kind"):
            EvalQuestion(**{**question.model_dump(), "kind": "opinion"})
        with pytest.raises(ValidationError, match="no gold"):
            EvalQuestion(**{**question.model_dump(), "gold": ("H209100-1",)})

    def test_an_interpellation_has_its_session_and_no_provenance(self):
        question = EvalQuestion(
            question_id="interpellation-0001",
            question_set="interpellation",
            split="dev",
            text="Arbetsvillkor inom gigekonomin",
            gold=("HC09139-36", "HC09139-37"),
            session="2024/25",
        )
        assert question.session == "2024/25"
        with pytest.raises(ValidationError, match="provenance"):
            EvalQuestion(**{**question.model_dump(), "provenance": PROVENANCE})

    def test_a_stance_question_asks_for_or_against(self):
        fields = {"question_id": "stance-0001", "question_set": "stance"}
        assert natural(**fields, kind="against").kind == "against"
        with pytest.raises(ValidationError, match="for or against"):
            natural(**fields, kind=None)

    def test_a_gold_speech_listed_twice_is_refused(self):
        with pytest.raises(ValidationError, match="twice"):
            EvalQuestion(
                question_id="interpellation-0001",
                question_set="interpellation",
                split="dev",
                text="Tema",
                gold=("a-1", "a-1"),
                session="2024/25",
            )


class TestFiles:
    def test_a_set_reads_back_sorted_by_key_and_hashed_by_its_bytes(self, tmp_path):
        path = tmp_path / set_file_name("natural")
        questions = [natural(2), natural(1, split="test")]
        digest = write_question_set(path, questions)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
        back = read_question_set(path)
        assert [question.question_id for question in back] == ["natural-0001", "natural-0002"]
        assert back[1] == questions[0]

    def test_writing_the_same_set_twice_gives_the_same_hash(self, tmp_path):
        questions = [natural(1), natural(2)]
        first = write_question_set(tmp_path / "a.jsonl", questions)
        second = write_question_set(tmp_path / "b.jsonl", list(reversed(questions)))
        assert first == second

    def test_one_file_holds_one_set(self, tmp_path):
        keyword = natural(question_id="keyword-0001", question_set="keyword")
        with pytest.raises(ValueError, match="one set"):
            write_question_set(tmp_path / "x.jsonl", [natural(), keyword])

    def test_a_repeated_key_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="twice"):
            write_question_set(tmp_path / "x.jsonl", [natural(), natural()])

    def test_the_splits_can_be_taken_apart(self):
        questions = [natural(1), natural(2, split="test")]
        assert [q.question_id for q in split_of(questions, "test")] == ["natural-0002"]
