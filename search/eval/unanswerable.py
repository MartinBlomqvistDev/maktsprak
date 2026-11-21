"""The unanswerable questions: what the model is asked for, and what the code accepts.

Each kind is written by one request that asks for more questions than the set
keeps. The code checks every question it can check and keeps the first ones
that pass, in the order the model wrote them:

- ``off_topic``: the form of the question only.
- ``unknown_speaker``: the name is written in the question and is not part of
  any speaker's name in the corpus.
- ``sd_before_2010``: the question names Sverigedemokraterna and a year from
  2003 to 2009, and no other year.
- ``after_index``: the question names a year from 2028 to 2035, and no other year.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from search.config import UnanswerableKind
from search.eval.generation import MAX_QUESTION_WORDS, response_json, years_in
from search.eval.llm import Usage

#: The kinds, in the order the set lists them.
KINDS: tuple[UnanswerableKind, ...] = (
    "off_topic",
    "unknown_speaker",
    "sd_before_2010",
    "after_index",
)

#: The years a question about Sverigedemokraterna before 2010 may name.
SD_YEARS = range(2003, 2010)

#: The years a question about events after the index may name.
AFTER_YEARS = range(2028, 2036)

_QUESTIONS_OF = {
    "type": "object",
    "properties": {"questions": {"type": "array"}},
    "required": ["questions"],
    "additionalProperties": False,
}


def _schema(items: Mapping[str, Any]) -> dict[str, Any]:
    schema = {**_QUESTIONS_OF, "properties": {"questions": {"type": "array", "items": dict(items)}}}
    return schema


#: The JSON schema of each kind's answer.
SCHEMAS: Mapping[UnanswerableKind, Mapping[str, Any]] = {
    "off_topic": _schema({"type": "string"}),
    "unknown_speaker": _schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "question": {"type": "string"}},
            "required": ["name", "question"],
            "additionalProperties": False,
        }
    ),
    "sd_before_2010": _schema(
        {
            "type": "object",
            "properties": {"year": {"type": "integer"}, "question": {"type": "string"}},
            "required": ["year", "question"],
            "additionalProperties": False,
        }
    ),
    "after_index": _schema(
        {
            "type": "object",
            "properties": {"year": {"type": "integer"}, "question": {"type": "string"}},
            "required": ["year", "question"],
            "additionalProperties": False,
        }
    ),
}


class UnanswerableRequest(BaseModel):
    """One request for unanswerable questions of one kind; a line of its log.

    Attributes:
        custom_id: The request's id.
        kind: The kind of question asked for.
        asked: How many questions were asked for.
        prompt_sha256: The prompt file's hash.
        response: The model's answer, as returned.
        model: The model that answered.
        stop_reason: Why the model stopped.
        error: What went wrong when no answer came back.
        usage: The tokens used.
        cost_usd: What the request cost.
        batch_id: The message batch it ran in, if any.
        created: When the answer was recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str
    kind: UnanswerableKind
    asked: int
    prompt_sha256: str
    response: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    usage: Usage = Usage()
    cost_usd: float = 0.0
    batch_id: str | None = None
    created: datetime


def _form_problem(question: str) -> str | None:
    """Say what is wrong with the form of a question, or ``None``."""
    if not question or question != question.strip() or "\n" in question:
        return "not one line"
    if not question.endswith("?"):
        return "no question mark"
    if len(question.split()) > MAX_QUESTION_WORDS:
        return "too long"
    return None


def check(
    kind: UnanswerableKind, item: Any, speaker_names: Sequence[str]
) -> tuple[str | None, str | None]:
    """Check one question the model wrote.

    Args:
        kind: The kind it was asked for.
        item: The question as the answer holds it: a string for ``off_topic``,
            an object with ``question`` and ``name`` or ``year`` for the others.
        speaker_names: Every speaker in the corpus as published, lower-case.

    Returns:
        The question's text, or ``None`` if it could not be read, and what is
        wrong with it, or ``None`` if nothing is.
    """
    if kind == "off_topic":
        text = item if isinstance(item, str) else None
    else:
        text = item.get("question") if isinstance(item, dict) else None
    if not isinstance(text, str):
        return None, "unreadable"
    problem = _form_problem(text)
    if problem:
        return text, problem
    years = years_in(text)
    if kind == "unknown_speaker":
        name = item.get("name", "")
        if not isinstance(name, str) or len(name.split()) < 2 or name not in text:
            return text, "the name is missing from the question or is not a full name"
        if any(name.lower() in speaker for speaker in speaker_names):
            return text, "the name is part of a speaker's name in the corpus"
    elif kind == "sd_before_2010":
        if "sverigedemokrat" not in text.lower():
            return text, "Sverigedemokraterna is not named"
        if not years or any(year not in SD_YEARS for year in years):
            return text, "the years are not from 2003 to 2009"
    elif kind == "after_index" and (not years or any(year not in AFTER_YEARS for year in years)):
        return text, "the years are not from 2028 to 2035"
    return text, None


def accepted(
    kind: UnanswerableKind,
    requests: Iterable[UnanswerableRequest],
    keep: int,
    speaker_names: Sequence[str],
) -> tuple[list[tuple[str, UnanswerableRequest]], dict[str, int]]:
    """Keep the first questions of a kind that pass the checks, each text once.

    Args:
        kind: The kind.
        requests: The kind's requests, in the order they were made.
        keep: How many to keep.
        speaker_names: Every speaker in the corpus as published, lower-case.

    Returns:
        The kept questions with the request each came from, and how many were
        turned away for each reason.
    """
    kept: list[tuple[str, UnanswerableRequest]] = []
    seen: set[str] = set()
    turned_away: dict[str, int] = {}
    for request in requests:
        if request.kind != kind:
            continue
        answer = response_json(request.response)
        items = answer.get("questions", []) if isinstance(answer, dict) else []
        for item in items:
            if len(kept) >= keep:
                break
            text, problem = check(kind, item, speaker_names)
            if text is None or problem is not None:
                reason = problem or "unreadable"
                turned_away[reason] = turned_away.get(reason, 0) + 1
                continue
            if text.casefold() in seen:
                reason = "repeats an earlier question"
                turned_away[reason] = turned_away.get(reason, 0) + 1
                continue
            seen.add(text.casefold())
            kept.append((text, request))
    return kept, dict(sorted(turned_away.items()))
