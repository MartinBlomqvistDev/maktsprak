"""The judge: a language model that grades how well a passage answers a question.

The scale has three grades, the same for the judge and for Martin:

- 2: the passage answers the question. It gives a position, an argument, a
  proposal or a fact that the question asks for.
- 1: the passage is about the question's topic but does not answer it.
- 0: the passage is about something else.

The judge's grades are used only if they agree with Martin's on 100
question-passage pairs, with Cohen's kappa at least 0.6
(``docs/search/PREREGISTRATION.md``). The kappa that decides is the unweighted
one over the three grades; the linear-weighted kappa and the kappa of "answers"
(grade 2) against the rest are reported beside it and decide nothing.

The 100 pairs are built from the pilot's questions, which belong to no question
set, so the calibration touches neither split. Each of 25 drawn pilot questions
is paired with the passage it was written from and with the first three other
passages that BM25 ranks for it, at Lucene's default parameters.
"""

from __future__ import annotations

import html
import random
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from search.corpus.table import PassageTable
from search.eval.generation import Prompt, response_json
from search.eval.llm import Completion, Usage, message_params
from search.eval.review import render_page
from search.index.lexical import DEFAULT_PARAMS, LexicalIndex

#: The grades, lowest first.
GRADES: tuple[int, ...] = (0, 1, 2)

#: Cohen's kappa the judge must reach for its grades to be used.
KAPPA_THRESHOLD = 0.6

#: How many pilot questions the calibration draws, and how many passages each gets.
CALIBRATION_QUESTIONS = 25
PASSAGES_PER_QUESTION = 4

#: The JSON schema of the judge's answer.
JUDGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "grade": {"type": "integer", "enum": list(GRADES)},
    },
    "required": ["reason", "grade"],
    "additionalProperties": False,
}

#: What each grade means, as the form shows it to Martin. The judge's prompt
#: holds the same three sentences.
GRADE_LABELS: Mapping[int, str] = {
    2: "2: besvarar frågan",
    1: "1: rätt ämne, besvarar inte frågan",
    0: "0: annat ämne",
}


class PilotQuestion(BaseModel):
    """A question the pilot wrote, with the passage it was written from.

    Attributes:
        question_id: The pilot's key for it.
        text: The question.
        passage_id: The passage it was written from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    text: str
    passage_id: str


class JudgePair(BaseModel):
    """One question-passage pair of the calibration.

    Attributes:
        pair_id: The key, in the order the form shows the pairs.
        question_id: The pilot question.
        question: Its text.
        passage_id: The passage.
        source: ``written_from`` for the passage the question was written from,
            ``bm25_<rank>`` for a passage BM25 ranked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    question_id: str
    question: str
    passage_id: str
    source: str


def calibration_pairs(
    questions: Sequence[PilotQuestion],
    index: LexicalIndex,
    table: PassageTable,
    seed: int,
    count: int = CALIBRATION_QUESTIONS,
    per_question: int = PASSAGES_PER_QUESTION,
) -> list[JudgePair]:
    """Draw pilot questions and pair each with its own passage and BM25's next best.

    Args:
        questions: The pilot's questions.
        index: The BM25 index the other passages are ranked in.
        table: The passages, to name the ranked ones.
        seed: Decides the questions drawn and the order of the pairs.
        count: How many questions to draw.
        per_question: How many passages each question gets, its own included.

    Returns:
        The pairs, numbered in a random order.
    """
    rng = random.Random(seed)
    drawn = rng.sample(list(questions), count)
    pairs: list[tuple[PilotQuestion, str, str]] = []
    for question in drawn:
        pairs.append((question, question.passage_id, "written_from"))
        others = [
            (hit.passage, rank)
            for rank, hit in enumerate(index.search(question.text, DEFAULT_PARAMS), start=1)
            if table.passage_ids[hit.passage] != question.passage_id
        ]
        for position, rank in others[: per_question - 1]:
            pairs.append((question, table.passage_ids[position], f"bm25_{rank}"))
    rng.shuffle(pairs)
    return [
        JudgePair(
            pair_id=f"pair-{number:03d}",
            question_id=question.question_id,
            question=question.text,
            passage_id=passage_id,
            source=source,
        )
        for number, (question, passage_id, source) in enumerate(pairs, start=1)
    ]


def judge_request(
    prompt: Prompt, question: str, passage: str, model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    """Build the judge's request for one pair."""
    return message_params(
        model=model,
        max_tokens=max_tokens,
        effort=effort,
        system=prompt.system,
        user=prompt.user.format(question=question, passage=passage),
        schema=JUDGE_SCHEMA,
    )


class JudgeVerdict(BaseModel):
    """The judge's grade of one pair; a line of the judge's log.

    Attributes:
        pair_id: The pair.
        grade: The grade, or ``None`` when no usable answer came back.
        reason: The judge's reason.
        model: The model that answered.
        prompt_sha256: The hash of the judge's prompt file.
        stop_reason: Why the model stopped.
        error: What went wrong, when something did.
        usage: The tokens used.
        cost_usd: What the request cost.
        created: When the answer was recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    grade: int | None
    reason: str = ""
    model: str | None = None
    prompt_sha256: str
    stop_reason: str | None = None
    error: str | None = None
    usage: Usage = Usage()
    cost_usd: float = 0.0
    created: datetime


def read_grade(completion: Completion) -> tuple[int | None, str]:
    """Take the grade and the reason from the judge's answer.

    Returns:
        The grade, or ``None`` when the answer is missing, cut off, refused or
        not one of the grades, and the reason or what went wrong.
    """
    if completion.error is not None:
        return None, completion.error
    if completion.stop_reason in ("refusal", "max_tokens"):
        return None, f"stopped: {completion.stop_reason}"
    answer = response_json(completion.text)
    if not isinstance(answer, dict) or answer.get("grade") not in GRADES:
        return None, "no grade in the answer"
    return int(answer["grade"]), str(answer.get("reason", ""))


def cohen_kappa(first: Sequence[int], second: Sequence[int], categories: Sequence[int]) -> float:
    """Cohen's kappa of two raters over the same items, unweighted.

    Args:
        first: One rater's grades.
        second: The other's, item by item.
        categories: Every grade either may give.

    Returns:
        ``(p_o - p_e) / (1 - p_e)``, where ``p_o`` is the share of items on
        which they agree and ``p_e`` the agreement expected from each rater's
        own distribution. 1.0 when both give one and the same grade throughout.

    Raises:
        ValueError: If the two lists differ in length or are empty.
    """
    return weighted_kappa(
        first, second, categories, [[float(a == b) for b in categories] for a in categories]
    )


def linear_weights(categories: Sequence[int]) -> list[list[float]]:
    """Linear agreement weights: 1 on the diagonal, falling by the distance between grades."""
    span = len(categories) - 1
    return [[1 - abs(i - j) / span for j in range(len(categories))] for i in range(len(categories))]


def weighted_kappa(
    first: Sequence[int],
    second: Sequence[int],
    categories: Sequence[int],
    weights: Sequence[Sequence[float]],
) -> float:
    """Cohen's kappa with agreement weights between grades.

    Args:
        first: One rater's grades.
        second: The other's, item by item.
        categories: Every grade either may give, in order.
        weights: ``weights[i][j]`` is the credit for grades ``categories[i]``
            and ``categories[j]``; 1 on the diagonal.

    Returns:
        ``(p_o - p_e) / (1 - p_e)`` with weighted observed and expected agreement.

    Raises:
        ValueError: If the two lists differ in length or are empty.
    """
    if len(first) != len(second) or not first:
        raise ValueError("Kappa needs two equally long, non-empty lists of grades.")
    where = {grade: number for number, grade in enumerate(categories)}
    size = len(categories)
    counts = [[0] * size for _ in range(size)]
    for a, b in zip(first, second, strict=True):
        counts[where[a]][where[b]] += 1
    total = len(first)
    rows = [sum(row) / total for row in counts]
    columns = [sum(counts[i][j] for i in range(size)) / total for j in range(size)]
    observed = sum(weights[i][j] * counts[i][j] / total for i in range(size) for j in range(size))
    expected = sum(weights[i][j] * rows[i] * columns[j] for i in range(size) for j in range(size))
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


class Agreement(BaseModel):
    """How the judge's grades agree with Martin's.

    Attributes:
        pairs: How many pairs both graded.
        kappa: The unweighted kappa over the three grades, which decides.
        threshold: The kappa required.
        accepted: Whether the judge's grades may be used.
        linear_kappa: The linear-weighted kappa, reported only.
        answers_kappa: The kappa of grade 2 against grades 0 and 1, reported only.
        exact: The share of pairs given the same grade.
        table: How often each pair of grades occurred, Martin's first.
        ungraded: Pairs the judge gave no grade, left out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: int
    kappa: float
    threshold: float = KAPPA_THRESHOLD
    accepted: bool
    linear_kappa: float
    answers_kappa: float
    exact: float
    table: dict[str, int]
    ungraded: tuple[str, ...]


def agreement(martin: Mapping[str, int], judge: Mapping[str, int | None]) -> Agreement:
    """Compare the judge's grades with Martin's, pair by pair.

    Args:
        martin: Martin's grade of every pair.
        judge: The judge's grade of every pair, ``None`` where it gave none.

    Returns:
        The agreement. A pair the judge did not grade is left out and named.

    Raises:
        ValueError: If the two do not cover the same pairs.
    """
    if set(martin) != set(judge):
        raise ValueError("Martin's grades and the judge's cover different pairs.")
    ungraded = tuple(sorted(pair for pair, grade in judge.items() if grade is None))
    first: list[int] = []
    second: list[int] = []
    for pair in sorted(martin):
        grade = judge[pair]
        if grade is not None:
            first.append(martin[pair])
            second.append(grade)
    kappa = cohen_kappa(first, second, GRADES)
    table: dict[str, int] = {}
    for a, b in zip(first, second, strict=True):
        key = f"{a}-{b}"
        table[key] = table.get(key, 0) + 1
    return Agreement(
        pairs=len(first),
        kappa=round(kappa, 4),
        accepted=kappa >= KAPPA_THRESHOLD,
        linear_kappa=round(weighted_kappa(first, second, GRADES, linear_weights(GRADES)), 4),
        answers_kappa=round(
            cohen_kappa([int(a == 2) for a in first], [int(b == 2) for b in second], (0, 1)), 4
        ),
        exact=round(sum(a == b for a, b in zip(first, second, strict=True)) / len(first), 4),
        table=dict(sorted(table.items())),
        ungraded=ungraded,
    )


def calibration_form(
    pairs: Sequence[JudgePair], passages: Mapping[str, str], storage_key: str
) -> str:
    """Render the calibration form: each pair with the three grades to choose from.

    Args:
        pairs: The pairs, in the order to show them.
        passages: The text of each passage, by passage id.
        storage_key: Names the form's saved answers in the browser and its file.

    Returns:
        One self-contained HTML page.
    """
    articles = []
    for number, pair in enumerate(pairs, start=1):
        paragraphs = "".join(
            f"<p>{html.escape(paragraph)}</p>"
            for paragraph in passages[pair.passage_id].split("\n\n")
        )
        boxes = "".join(
            f'<label><input type="radio" name="{pair.pair_id}:grade" value="{grade}"> '
            f"{html.escape(GRADE_LABELS[grade])}</label><br>"
            for grade in sorted(GRADES, reverse=True)
        )
        articles.append(
            "<article>"
            f'<div class="meta">Par {number} av {len(pairs)}</div>'
            f'<div class="question">{html.escape(pair.question)}</div>'
            f"<blockquote>{paragraphs}</blockquote>"
            f"<fieldset><legend>Hur väl besvarar utdraget frågan?</legend>{boxes}</fieldset>"
            f'<textarea name="{pair.pair_id}:comment" rows="1" '
            'placeholder="Kommentar (frivillig)"></textarea>'
            "</article>"
        )
    intro = (
        "Ge varje par ett betyg. 2: utdraget besvarar frågan; det ger en ståndpunkt, ett "
        "argument, ett förslag eller en uppgift som frågan efterfrågar. 1: utdraget handlar "
        "om frågans ämne men besvarar inte frågan. 0: utdraget handlar om något annat. Bedöm "
        "bara det som står i utdraget, oavsett om du håller med talaren. Samma fråga kan "
        "komma tillbaka med andra utdrag. Svaren sparas i webbläsaren medan du arbetar. När "
        "du är klar, kopiera eller ladda ned svaren och lämna dem i sessionen."
    )
    return render_page(
        title_count=len(pairs),
        intro=intro,
        articles=articles,
        item_ids=[pair.pair_id for pair in pairs],
        storage_key=storage_key,
        fields_js='[{name: "grade", parse: (value) => Number(value)}]',
        heading="Kalibrering av domaren",
    )
