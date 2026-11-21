"""Tests for search/eval/assemble.py."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from search.corpus.table import load_passage_table
from search.eval.assemble import (
    interpellation_questions,
    passage_questions,
    unanswerable_questions,
    write_sets,
)
from search.eval.generation import Attempt
from search.eval.interpellations import Debate
from search.eval.questions import QUESTION_SETS, read_question_set
from search.eval.sampling import Slot
from search.eval.unanswerable import UnanswerableRequest
from tests.search.corpus_kit import write_corpus

WHEN = datetime(2026, 11, 22, 10, tzinfo=timezone.utc)


def filled(slot_id: str, passage_id: str, answer: dict) -> Attempt:
    return Attempt(
        custom_id=f"{slot_id}-1",
        slot_id=slot_id,
        passage_id=passage_id,
        retry=False,
        prompt="natural",
        prompt_sha256="1" * 64,
        outcome="filled",
        response=json.dumps(answer, ensure_ascii=False),
        model="claude-opus-5-5",
        created=WHEN,
    )


@pytest.fixture
def table(tmp_path):
    _, passages_path = write_corpus(tmp_path)
    return load_passage_table(passages_path)


class TestPassageQuestions:
    def test_each_filled_slot_becomes_its_questions(self, table):
        slots = [
            Slot(slot_id="natural-0001", question_set="natural", split="dev", frame_position=0),
            Slot(slot_id="natural-0002", question_set="natural", split="test", frame_position=1),
            Slot(
                slot_id="filtered-0001",
                question_set="filtered",
                split="test",
                frame_position=2,
                filter_kind="party",
            ),
            Slot(
                slot_id="filtered-0002",
                question_set="filtered",
                split="dev",
                frame_position=3,
                filter_kind="period",
            ),
            Slot(slot_id="stance-0001", question_set="stance", split="dev", frame_position=4),
        ]
        question = {"usable": True, "reason": "", "question": "Varför?", "keywords": ["a", "b"]}
        stance = {"usable": True, "reason": "", "direction": "mot", "question": "Vilka?"}
        attempts = [
            filled("natural-0001", "H209100-1:0-1", question),
            filled("filtered-0001", "GR0950-1:0-1", question),
            filled("filtered-0002", "GR0950-2:0-1", question),
            filled("stance-0001", "HB0960-1:0-1", stance),
        ]
        sets = passage_questions(slots, attempts, table, generation=1)
        assert [q.question_id for q in sets["natural"]] == ["natural-0001"]
        assert sets["keyword"][0].question_id == "keyword-0001"
        assert sets["keyword"][0].text == "a b"
        assert sets["keyword"][0].split == "dev"
        assert sets["filtered"][0].filter.party == "S"
        assert sets["filtered"][1].filter.period == "2003-2009"
        assert sets["stance"][0].kind == "against"
        assert sets["natural"][0].provenance.created.isoformat() == "2026-11-22"
        assert sets["natural"][0].gold == ("H209100-1",)


class TestOtherSets:
    def test_unanswerable_questions_are_kept_and_split_per_kind(self):
        requests = [
            UnanswerableRequest(
                custom_id="unanswerable-off-topic",
                kind="off_topic",
                asked=6,
                prompt_sha256="2" * 64,
                response=json.dumps({"questions": [f"Fråga {n}?" for n in range(6)]}),
                model="claude-opus-5-5",
                created=WHEN,
            )
        ]
        keep = {"off_topic": 5, "unknown_speaker": 2, "sd_before_2010": 2, "after_index": 5}
        questions, refused = unanswerable_questions(requests, keep, [], generation=1, seed=3)
        assert [q.question_id for q in questions] == [f"unanswerable-{n:04d}" for n in range(1, 6)]
        assert {q.kind for q in questions} == {"off_topic"}
        assert sum(q.split == "dev" for q in questions) == 2
        assert refused["off_topic"] == {}

    def test_each_debate_is_one_question_with_its_speeches_as_gold(self):
        debates = [
            Debate(heading="h1", session="2007/08", topic="Tema ett", speeches=("A-1", "A-2")),
            Debate(heading="h2", session="2007/08", topic="Tema två", speeches=("B-1",)),
        ]
        questions = interpellation_questions(debates, seed=1)
        assert [q.text for q in questions] == ["Tema ett", "Tema två"]
        assert questions[0].gold == ("A-1", "A-2")
        assert questions[0].session == "2007/08"


class TestWrite:
    def test_every_set_is_written_with_its_hash_and_counts(self, tmp_path):
        sets = {name: [] for name in QUESTION_SETS}
        debates = [Debate(heading="h", session="2007/08", topic="Tema", speeches=("A-1",))]
        sets["interpellation"] = interpellation_questions(debates, seed=1)
        records = write_sets(tmp_path, sets)
        record = records["interpellation"]
        path = tmp_path / record.file
        assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record.questions == 1 and record.dev + record.test == 1
        assert read_question_set(path) == sets["interpellation"]
        assert records["natural"].questions == 0

    def test_a_missing_set_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            write_sets(tmp_path, {"natural": []})
