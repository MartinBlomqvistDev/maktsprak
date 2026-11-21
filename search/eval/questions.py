"""The evaluation's questions: their contract, and how a question set is stored and hashed.

A question set is a JSON Lines file, one question per line, sorted by id. The
hash of each file is recorded in an addendum to the pre-registration before
any system is scored on it, so a file that changes afterwards no longer matches
the record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from search.parties import VALID_PARTIES

QuestionSet = Literal["natural", "keyword", "interpellation", "filtered", "unanswerable", "stance"]
Split = Literal["dev", "test"]

#: The sets whose questions a language model wrote from a passage.
PASSAGE_SETS: tuple[QuestionSet, ...] = ("natural", "keyword", "filtered", "stance")

#: The kinds of unanswerable question, as the pre-registration lists them.
UnanswerableKind = Literal["off_topic", "unknown_speaker", "sd_before_2010", "after_index"]

#: Whether a stance question asks for the arguments for or against something.
StanceDirection = Literal["for", "against"]

QUESTION_SETS: tuple[QuestionSet, ...] = (
    "natural",
    "keyword",
    "interpellation",
    "filtered",
    "unanswerable",
    "stance",
)


class Provenance(BaseModel):
    """Who wrote a question, with what, and when.

    Attributes:
        model: The id of the language model that wrote it.
        prompt_sha256: The hash of the prompt file it was written from.
        created: The day it was written.
        generation: Which of the at most two generations of the question sets it belongs to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    prompt_sha256: str
    created: date
    generation: int = Field(ge=1, le=2)


class QuestionFilter(BaseModel):
    """The one filter a filtered question is searched under: a party or a period.

    Attributes:
        party: One of the eight parties, or ``None``.
        period: A period label such as ``2010-2017``, or ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    party: str | None = None
    period: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> QuestionFilter:
        if (self.party is None) == (self.period is None):
            raise ValueError("A filter names either a party or a period, not both or neither.")
        if self.party is not None and self.party not in VALID_PARTIES:
            raise ValueError(f"{self.party!r} is not one of the eight parties.")
        return self


class EvalQuestion(BaseModel):
    """One question of the evaluation, with the speeches that answer it.

    Attributes:
        question_id: The key, for example ``natural-0017``.
        question_set: Which set the question belongs to.
        split: ``dev`` for the development split, ``test`` for the test split.
        text: The question as a searcher would type it.
        gold: The speeches that answer it. Empty for an unanswerable question.
        passage_id: The passage the question was written from, for the sets a
            model wrote from a passage.
        filter: The filter of a filtered question.
        kind: The kind of an unanswerable question, or the direction of a stance question.
        session: The session of an interpellation question.
        provenance: The model, prompt and date behind a generated question.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    question_set: QuestionSet
    split: Split
    text: str = Field(min_length=1)
    gold: tuple[str, ...]
    passage_id: str | None = None
    filter: QuestionFilter | None = None
    kind: str | None = None
    session: str | None = None
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _check_set_rules(self) -> EvalQuestion:
        name = self.question_id
        if not name.startswith(f"{self.question_set}-"):
            raise ValueError(f"{name}: the key must start with its set's name.")
        if self.text != self.text.strip() or "\n" in self.text:
            raise ValueError(f"{name}: the text must be one line without outer whitespace.")
        if len(set(self.gold)) != len(self.gold):
            raise ValueError(f"{name}: a gold speech is listed twice.")
        generated = self.question_set != "interpellation"
        if generated != (self.provenance is not None):
            raise ValueError(f"{name}: a model-written question needs its provenance, no other.")
        if (self.question_set == "unanswerable") == bool(self.gold):
            raise ValueError(f"{name}: only an unanswerable question has no gold speech.")
        from_passage = self.question_set in PASSAGE_SETS
        if from_passage != (self.passage_id is not None):
            raise ValueError(f"{name}: only a question written from a passage names one.")
        speech = self.passage_id.split(":")[0] if self.passage_id is not None else None
        if from_passage and self.gold != (speech,):
            raise ValueError(f"{name}: the gold speech must be the passage's own speech.")
        if (self.question_set == "filtered") != (self.filter is not None):
            raise ValueError(f"{name}: only a filtered question carries a filter.")
        if (self.question_set == "interpellation") != (self.session is not None):
            raise ValueError(f"{name}: only an interpellation question names its session.")
        if self.question_set == "unanswerable" and self.kind not in (
            "off_topic",
            "unknown_speaker",
            "sd_before_2010",
            "after_index",
        ):
            raise ValueError(f"{name}: {self.kind!r} is not a kind of unanswerable question.")
        if self.question_set == "stance" and self.kind not in ("for", "against"):
            raise ValueError(f"{name}: a stance question asks for or against, not {self.kind!r}.")
        if self.question_set not in ("unanswerable", "stance") and self.kind is not None:
            raise ValueError(f"{name}: only unanswerable and stance questions have a kind.")
        return self


def set_file_name(question_set: QuestionSet) -> str:
    """The file a question set is stored in, for example ``natural.jsonl``."""
    return f"{question_set}.jsonl"


def write_question_set(path: Path, questions: Sequence[EvalQuestion]) -> str:
    """Write one question set as JSON Lines, sorted by id, and return the file's hash.

    Args:
        path: The file to write.
        questions: The questions; all of one set, each id once.

    Returns:
        The SHA-256 of the written file, in hex.

    Raises:
        ValueError: If the questions belong to more than one set or an id repeats.
    """
    sets = {question.question_set for question in questions}
    if len(sets) > 1:
        raise ValueError(f"A question set file holds one set, not {sorted(sets)}.")
    ids = [question.question_id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("A question id occurs twice.")
    lines = [
        question.model_dump_json(exclude_none=True)
        for question in sorted(questions, key=lambda question: question.question_id)
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_question_set(path: Path) -> list[EvalQuestion]:
    """Read a question set written by :func:`write_question_set`, checking every question.

    Args:
        path: The file.

    Returns:
        The questions, in the file's order.
    """
    text = path.read_text(encoding="utf-8")
    return [EvalQuestion.model_validate_json(line) for line in text.splitlines() if line]


def split_of(questions: Iterable[EvalQuestion], split: Split) -> list[EvalQuestion]:
    """Keep the questions of one split."""
    return [question for question in questions if question.split == split]
