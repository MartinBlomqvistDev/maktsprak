"""Grading the answer step's quotations with the calibrated judge.

The judge grades a quotation as it graded the calibration pairs: with the same
model, effort and prompt, ``search/eval/prompts/judge.md``, whose hash is
checked against the one the pre-registration records. A pair of a question and
a text is graded once, and its grade serves every selector and round that gives
that text for that question. On the development split the judge also grades
every candidate of the answerable questions, so that an abstention can be told
apart from a miss of the retrieval; those grades are description.

A pair the judge gave no grade, because the request failed, was refused or
was cut off, is sent once more. After two such attempts it stays ungraded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from search.config import Prices
from search.eval.generation import Prompt
from search.eval.judge import judge_request, read_grade
from search.eval.llm import Completion, Usage, cost_usd

#: The hash of the judge's prompt as it graded the calibration pairs.
JUDGE_PROMPT_SHA256 = "c7361c709ba78f606ad8d70ebe460b2b238dfa4a5c600ee11141889f80803168"

#: How many requests a pair gets before it stays ungraded.
MAX_ATTEMPTS = 2


def text_sha256(text: str) -> str:
    """The hash of a text, as a graded pair records it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GradingPair(BaseModel):
    """A question and a text for the judge to grade.

    Attributes:
        question_id: The question's key.
        question: The question.
        text: A quotation or a candidate, paragraphs separated by a blank line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    question: str
    text: str

    @property
    def key(self) -> str:
        """The pair's key: the question and the hash of the text."""
        return f"{self.question_id}:{text_sha256(self.text)}"

    @property
    def custom_id(self) -> str:
        """The pair's id in a message batch, which allows letters, digits, ``-`` and ``_``."""
        return f"{self.question_id}-{text_sha256(self.text)[:24]}"


class QuoteGrade(BaseModel):
    """The judge's grade of one pair; a line of the grade log.

    Attributes:
        question_id: The question.
        text_sha256: The hash of the text graded.
        grade: The grade, or ``None`` when no usable answer came back.
        reason: The judge's reason, or what went wrong.
        model: The model that answered.
        prompt_sha256: The hash of the judge's prompt.
        stop_reason: Why the model stopped.
        error: What went wrong, when something did.
        usage: The tokens used.
        cost_usd: What the request cost.
        batch_id: The message batch the request ran in.
        created: When the answer was recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    text_sha256: str
    grade: int | None
    reason: str = ""
    model: str | None = None
    prompt_sha256: str
    stop_reason: str | None = None
    error: str | None = None
    usage: Usage = Usage()
    cost_usd: float = 0.0
    batch_id: str | None = None
    created: datetime

    @property
    def key(self) -> str:
        """The key of the pair graded."""
        return f"{self.question_id}:{self.text_sha256}"


def check_judge_prompt(prompt: Prompt) -> None:
    """Stop unless the judge's prompt is the one it had for the calibration pairs.

    Raises:
        ValueError: If the prompt's hash differs.
    """
    if prompt.sha256 != JUDGE_PROMPT_SHA256:
        raise ValueError(
            f"The judge's prompt has the hash {prompt.sha256}, "
            f"not the calibrated {JUDGE_PROMPT_SHA256}."
        )


def unique_pairs(pairs: Iterable[GradingPair]) -> list[GradingPair]:
    """The pairs with each key once, in the order first seen."""
    seen: dict[str, GradingPair] = {}
    for pair in pairs:
        seen.setdefault(pair.key, pair)
    return list(seen.values())


def grades_by_key(log: Iterable[QuoteGrade]) -> dict[str, int | None]:
    """Each pair's grade: the first usable one in the log, else ``None``."""
    grades: dict[str, int | None] = {}
    for line in log:
        if grades.get(line.key) is None:
            grades[line.key] = line.grade
    return grades


def pending(pairs: Iterable[GradingPair], log: Sequence[QuoteGrade]) -> list[GradingPair]:
    """The pairs still to send: without a usable grade, and with attempts left.

    Args:
        pairs: The pairs that need a grade.
        log: Every grade recorded so far.

    Returns:
        The pairs to send, each once.
    """
    grades = grades_by_key(log)
    attempts: dict[str, int] = {}
    for line in log:
        attempts[line.key] = attempts.get(line.key, 0) + 1
    return [
        pair
        for pair in unique_pairs(pairs)
        if grades.get(pair.key) is None and attempts.get(pair.key, 0) < MAX_ATTEMPTS
    ]


def grading_requests(
    pairs: Sequence[GradingPair], prompt: Prompt, model: str, effort: str, max_tokens: int
) -> dict[str, dict[str, Any]]:
    """The judge's requests for some pairs, by their batch id.

    Raises:
        ValueError: If two pairs share a batch id.
    """
    requests = {
        pair.custom_id: judge_request(prompt, pair.question, pair.text, model, effort, max_tokens)
        for pair in pairs
    }
    if len(requests) != len(pairs):
        raise ValueError("Two pairs share a batch id.")
    return requests


def read_grades(
    pairs: Sequence[GradingPair],
    completions: Mapping[str, Completion],
    prompt: Prompt,
    prices: Prices,
    batch_discount: float,
    created: datetime,
) -> list[QuoteGrade]:
    """Turn the judge's completions into grades, one per pair sent.

    Args:
        pairs: The pairs sent.
        completions: The completions, by batch id.
        prompt: The judge's prompt.
        prices: The judge's list prices.
        batch_discount: The share taken off the price in a message batch.
        created: When the grades are recorded.

    Returns:
        One grade per pair; a pair without a completion gets ``None`` and the reason.
    """
    grades = []
    for pair in pairs:
        completion = completions.get(pair.custom_id) or Completion(
            custom_id=pair.custom_id, error="no result"
        )
        grade, reason = read_grade(completion)
        grades.append(
            QuoteGrade(
                question_id=pair.question_id,
                text_sha256=text_sha256(pair.text),
                grade=grade,
                reason=reason,
                model=completion.model,
                prompt_sha256=prompt.sha256,
                stop_reason=completion.stop_reason,
                error=completion.error,
                usage=completion.usage,
                cost_usd=cost_usd(completion.usage, prices, batch_discount),
                batch_id=completion.batch_id,
                created=created,
            )
        )
    return grades
