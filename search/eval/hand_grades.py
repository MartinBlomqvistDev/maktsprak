"""Grading the quotations of the test run by hand, as the addenda of 2026-11-24 set out.

The judge's kappa against my grades of the calibration pairs was 0.321, below
0.6, so the quotations of the test run are graded by hand. The addendum on the
answer step's test run fixes how, and the addendum on the hybrid's candidates
applies it to the run from those candidates:

- 100 answerable questions of the test split are drawn at random with the seed
  20261124 (:func:`draw_questions`).
- Every quotation that either selector gives to a drawn question is graded,
  each pair of question and text once (:func:`order_pairs`), on the scale and
  with the instructions of the calibration pairs (:func:`hand_form`).
- The pairs come in an order drawn with the same seed, and each shows only the
  question and the quotation, so the grader cannot tell which selector chose it.

Quote precision is the share of a selector's quotations on the drawn questions
graded 2, and the paired bootstrap for it resamples the drawn questions.
Balanced accuracy needs no grade and is computed on every question of the test
split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from search.eval.judge import GRADE_FIELDS_JS, GRADES, GRADING_INTRO, grading_article
from search.eval.quote_judge import GradingPair, text_sha256
from search.eval.review import render_page

#: The seed that draws the questions and the order of their pairs.
HAND_SEED = 20261124

#: How many answerable questions of the test split are drawn.
HAND_QUESTIONS = 100


class HandPair(BaseModel):
    """One pair on the form.

    Attributes:
        pair_id: Its key in the form's answers, numbered in the form's order.
        question_id: The question.
        text_sha256: The hash of the quotation's text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    question_id: str
    text_sha256: str

    @property
    def key(self) -> str:
        """The key of the :class:`GradingPair` with the same question and text."""
        return f"{self.question_id}:{self.text_sha256}"


class HandDraw(BaseModel):
    """What is graded by hand: the questions drawn, and the pairs in the form's order.

    Attributes:
        seed: The seed of the draw and of the order.
        run_sha256: The hash of the record of the run whose answers are graded.
        answers_sha256: The hash of each selector's answers file.
        questions: The questions drawn, in the order of their ids.
        pairs: The pairs, in the form's order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    run_sha256: str
    answers_sha256: dict[str, str]
    questions: tuple[str, ...]
    pairs: tuple[HandPair, ...]


class HandGradesRecord(BaseModel):
    """The hand grades as recorded once, before any metric is computed from them.

    The record is committed before the test report reads the grades, so git
    shows that the grades were fixed first. The answers themselves are kept
    under their hash.

    Attributes:
        form: The form the answers come from.
        draw_sha256: The hash of the record of the draw the grades answer.
        answers_sha256: The hash of the answers the form handed over.
        pairs: How many pairs they grade.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    form: str
    draw_sha256: str
    answers_sha256: str
    pairs: int


def draw_questions(answerable: Iterable[str], seed: int, size: int) -> list[str]:
    """Draw the answerable questions whose quotations are graded by hand.

    The questions are sorted by id before the draw, so the draw depends on the
    seed alone.

    Args:
        answerable: The ids of the answerable questions.
        seed: Decides the draw.
        size: How many to draw.

    Returns:
        The questions drawn, sorted by id.

    Raises:
        ValueError: If there are fewer questions than *size*.
    """
    pool = sorted(set(answerable))
    if len(pool) < size:
        raise ValueError(f"{len(pool)} answerable questions, and {size} are to be drawn.")
    return sorted(random.Random(seed).sample(pool, size))


def order_pairs(pairs: Iterable[GradingPair], seed: int) -> list[GradingPair]:
    """Each distinct pair once, in an order drawn with *seed*.

    Two selectors that give a question the same text give one pair, whose grade
    serves both. The pairs are sorted by key before they are shuffled, so the
    order depends on the seed alone.

    Args:
        pairs: The pairs, in any order and possibly repeated.
        seed: Decides the order.

    Returns:
        The distinct pairs, in the form's order.
    """
    unique: dict[str, GradingPair] = {}
    for pair in pairs:
        unique.setdefault(pair.key, pair)
    ordered = [unique[key] for key in sorted(unique)]
    random.Random(seed).shuffle(ordered)
    return ordered


def name_pairs(ordered: Sequence[GradingPair]) -> tuple[HandPair, ...]:
    """The pairs as the form names them: ``pair-001`` onwards, in the form's order."""
    width = max(3, len(str(len(ordered))))
    return tuple(
        HandPair(
            pair_id=f"pair-{number:0{width}d}",
            question_id=pair.question_id,
            text_sha256=text_sha256(pair.text),
        )
        for number, pair in enumerate(ordered, start=1)
    )


def hand_form(ordered: Sequence[GradingPair], names: Sequence[HandPair], storage_key: str) -> str:
    """Render the form: each pair with the three grades, under the calibration's instructions.

    Args:
        ordered: The pairs, in the form's order.
        names: Their names on the form, in the same order.
        storage_key: Names the form's saved answers in the browser and its file.

    Returns:
        One self-contained HTML page.

    Raises:
        ValueError: If the names are not those of the pairs.
    """
    if [name.key for name in names] != [pair.key for pair in ordered]:
        raise ValueError("The names on the form are not those of its pairs.")
    articles = [
        grading_article(number, len(ordered), name.pair_id, pair.question, pair.text)
        for number, (pair, name) in enumerate(zip(ordered, names, strict=True), start=1)
    ]
    return render_page(
        heading="Bedömning av testdelens citat",
        title_count=len(ordered),
        intro=GRADING_INTRO,
        articles=articles,
        item_ids=[name.pair_id for name in names],
        storage_key=storage_key,
        fields_js=GRADE_FIELDS_JS,
    )


def read_hand_grades(path: Path, draw: HandDraw, storage_key: str) -> dict[str, int]:
    """The grades from the form's answers, by the key of each pair.

    Args:
        path: The answers the form handed over, as JSON.
        draw: What the form showed.
        storage_key: The form's name, which the answers carry.

    Returns:
        Each pair's grade, by the pair's key.

    Raises:
        ValueError: If the answers come from another form, a pair has no
            grade, a grade is not 0, 1 or 2, or the answers name a pair the form
            did not show.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("form") != storage_key:
        raise ValueError(
            f"The answers come from the form {raw.get('form')!r}, not {storage_key!r}."
        )
    given = {entry["question_id"]: entry.get("grade") for entry in raw["answers"]}
    names = {pair.pair_id: pair for pair in draw.pairs}
    unknown = sorted(set(given) - set(names))
    missing = sorted(
        pair_id
        for pair_id in names
        if isinstance(given.get(pair_id), bool) or given.get(pair_id) not in GRADES
    )
    if unknown or missing:
        raise ValueError(
            f"The answers do not match the form. Without a grade: {missing}. Unknown: {unknown}."
        )
    return {names[pair_id].key: int(given[pair_id]) for pair_id in names}
