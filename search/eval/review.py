"""Martin's review of 50 generated questions: the draw, the form and the tally.

The 50 are drawn from the four sets a model wrote from a passage, in
proportion to their size: 20 natural, 20 keyword, 7 filtered and 3 stance
questions. Each question gets two answers: is it a reasonable question, and
does the passage answer it? A question is approved when both answers are yes.
Below 85% approved, the question sets are generated again, at most twice in
all (``docs/search/PREREGISTRATION.md``).

The form is one self-contained HTML file that runs in a browser without a
network. It keeps the answers in the browser while they are given, and hands
them over as JSON, which is checked here and recorded with the tally.
"""

from __future__ import annotations

import html
import json
import random
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from search.eval.questions import EvalQuestion

#: How many questions of each set the review draws.
REVIEW_SIZES: Mapping[str, int] = {"natural": 20, "keyword": 20, "filtered": 7, "stance": 3}

#: The share of approved questions below which the sets are generated again.
APPROVAL_THRESHOLD = 0.85


class ReviewItem(BaseModel):
    """One question as the form shows it.

    Attributes:
        item: The item's number on the form, from 1.
        question_id: The question's key.
        question_set: Its set.
        text: The question.
        filter: The filter of a filtered question, in words.
        passage: The passage it was written from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item: int
    question_id: str
    question_set: str
    text: str
    filter: str | None = None
    passage: str


def draw_review(
    sets: Mapping[str, Sequence[EvalQuestion]],
    passages: Mapping[str, str],
    seed: int,
    sizes: Mapping[str, int] = REVIEW_SIZES,
) -> list[ReviewItem]:
    """Draw the questions for the review, in a random order.

    Args:
        sets: The questions, by set.
        passages: The text of each passage, by passage id.
        seed: Decides the draw and the order.
        sizes: How many to draw from each set.

    Returns:
        The items, numbered in the order the form shows them.
    """
    rng = random.Random(seed)
    drawn: list[EvalQuestion] = []
    for name, size in sizes.items():
        drawn.extend(rng.sample(list(sets[name]), size))
    rng.shuffle(drawn)
    items = []
    for number, question in enumerate(drawn, start=1):
        described = None
        if question.filter is not None:
            described = (
                f"parti {question.filter.party}"
                if question.filter.party
                else f"period {question.filter.period}"
            )
        items.append(
            ReviewItem(
                item=number,
                question_id=question.question_id,
                question_set=question.question_set,
                text=question.text,
                filter=described,
                passage=passages[question.passage_id or ""],
            )
        )
    return items


class Verdict(BaseModel):
    """Martin's answers about one question.

    Attributes:
        question_id: The question's key.
        reasonable: Whether it is a reasonable question.
        answered: Whether the passage answers it.
        comment: Anything he added.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    reasonable: bool
    answered: bool
    comment: str = ""

    @property
    def approved(self) -> bool:
        """A question is approved when both answers are yes."""
        return self.reasonable and self.answered


class ReviewRecord(BaseModel):
    """The review, as recorded.

    Attributes:
        generation: The generation reviewed.
        reviewed: The day the answers were recorded.
        verdicts: One verdict per question, in form order.
        approved: How many questions were approved.
        share: The share approved.
        threshold: The share required.
        passed: Whether the share reached the threshold.
        by_set: Approved and reviewed questions, by set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    generation: int
    reviewed: date
    verdicts: tuple[Verdict, ...]
    approved: int
    share: float
    threshold: float = APPROVAL_THRESHOLD
    passed: bool
    by_set: dict[str, tuple[int, int]] = Field(default_factory=dict)


def tally(
    items: Sequence[ReviewItem], answers_json: str, generation: int, reviewed: date
) -> ReviewRecord:
    """Check the form's answers against the items and count the approved questions.

    Args:
        items: The items the form showed.
        answers_json: What the form handed over: a JSON object with an
            ``answers`` list of ``{question_id, reasonable, answered, comment}``.
        generation: The generation reviewed.
        reviewed: The day the answers were given.

    Returns:
        The record.

    Raises:
        ValueError: If a question lacks an answer, or an answer names a question
            the form did not show.
    """
    raw = json.loads(answers_json)
    verdicts = {entry["question_id"]: Verdict.model_validate(entry) for entry in raw["answers"]}
    shown = [item.question_id for item in items]
    if set(verdicts) != set(shown):
        missing = sorted(set(shown) - set(verdicts))
        extra = sorted(set(verdicts) - set(shown))
        raise ValueError(f"Answers do not match the form. Missing: {missing}. Unknown: {extra}.")
    ordered = tuple(verdicts[question_id] for question_id in shown)
    approved = sum(verdict.approved for verdict in ordered)
    by_set: dict[str, tuple[int, int]] = {}
    for item, verdict in zip(items, ordered, strict=True):
        done, total = by_set.get(item.question_set, (0, 0))
        by_set[item.question_set] = (done + verdict.approved, total + 1)
    share = approved / len(ordered)
    return ReviewRecord(
        generation=generation,
        reviewed=reviewed,
        verdicts=ordered,
        approved=approved,
        share=round(share, 4),
        passed=share >= APPROVAL_THRESHOLD,
        by_set=dict(sorted(by_set.items())),
    )


_SET_NAMES = {
    "natural": "naturlig fråga",
    "keyword": "sökord",
    "filtered": "fråga med filter",
    "stance": "hållningsfråga",
}

#: The page both forms are built on, with markers that :func:`render_page` fills in.
PAGE_FILE: Path = Path(__file__).resolve().with_name("form.html")


def _paragraphs(text: str) -> str:
    """Render a passage's paragraphs as HTML paragraphs."""
    return "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in text.split("\n\n"))


def _choices(item_id: str, name: str, legend: str, options: Sequence[tuple[str, str]]) -> str:
    """Render one row of radio buttons."""
    boxes = "".join(
        f'<label><input type="radio" name="{html.escape(item_id)}:{name}" '
        f'value="{html.escape(value)}"> {html.escape(label)}</label>'
        for value, label in options
    )
    return f"<fieldset><legend>{html.escape(legend)}</legend>{boxes}</fieldset>"


def render_page(
    *,
    heading: str,
    title_count: int,
    intro: str,
    articles: Sequence[str],
    item_ids: Sequence[str],
    storage_key: str,
    fields_js: str,
) -> str:
    """Fill in the page shared by the review and the judge's calibration.

    Args:
        heading: The page's title.
        title_count: How many items the page holds, shown after the title.
        intro: The instructions above the items, as plain text.
        articles: Each item as HTML.
        item_ids: The key of each item, in order, as the answers name them.
        storage_key: Names the saved answers in the browser and the downloaded file.
        fields_js: The answer fields as a JavaScript array of ``{name, parse}``.

    Returns:
        The page.
    """
    filled = {
        "__HEADING__": html.escape(heading),
        "__COUNT__": str(title_count),
        "__INTRO__": html.escape(intro),
        "__ITEMS__": json.dumps(list(item_ids), ensure_ascii=False),
        "__KEY__": json.dumps(storage_key),
        "__FIELDS__": fields_js,
        "__ARTICLES__": "\n".join(articles),
    }
    page = PAGE_FILE.read_text(encoding="utf-8")
    for marker, value in filled.items():
        page = page.replace(marker, value)
    return page


def review_form(items: Sequence[ReviewItem], storage_key: str) -> str:
    """Render the review form as one self-contained HTML page.

    Args:
        items: The items, in the order to show them.
        storage_key: Names the form's saved answers in the browser and its file.

    Returns:
        The page.
    """
    articles = []
    for item in items:
        kind = _SET_NAMES.get(item.question_set, item.question_set)
        extra = f", sökning med filter: {item.filter}" if item.filter else ""
        articles.append(
            "<article>"
            f'<div class="meta">Fråga {item.item} av {len(items)}, '
            f"{html.escape(kind)}{html.escape(extra)}</div>"
            f'<div class="question">{html.escape(item.text)}</div>'
            f"<blockquote>{_paragraphs(item.passage)}</blockquote>"
            + _choices(
                item.question_id,
                "reasonable",
                "Är frågan rimlig?",
                [("true", "Ja"), ("false", "Nej")],
            )
            + _choices(
                item.question_id,
                "answered",
                "Besvarar utdraget frågan?",
                [("true", "Ja"), ("false", "Nej")],
            )
            + f'<textarea name="{html.escape(item.question_id)}:comment" rows="1" '
            'placeholder="Kommentar (frivillig)"></textarea>'
            "</article>"
        )
    intro = (
        "Svara på två frågor om varje fråga: är den rimlig, och besvarar utdraget den? "
        "För sökord gäller: är det rimliga sökord, och är utdraget en bra träff för dem? "
        "Svaren sparas i webbläsaren medan du arbetar. När du är klar, kopiera eller "
        "ladda ned svaren och lämna dem i sessionen."
    )
    fields = (
        '[{name: "reasonable", parse: (value) => value === "true"},'
        ' {name: "answered", parse: (value) => value === "true"}]'
    )
    return render_page(
        heading="Granskning av frågor",
        title_count=len(items),
        intro=intro,
        articles=articles,
        item_ids=[item.question_id for item in items],
        storage_key=storage_key,
        fields_js=fields,
    )
