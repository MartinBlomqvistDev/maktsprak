"""The answer step's contracts: what a selector sees, what it may answer, and the answer.

A selector sees a question and its candidates, the passages the search ranked
highest. It answers in one form only: whether the candidates answer the
question, and at most four selections, each a candidate's passage id with the
first and last of its paragraphs, numbered from 1 within the passage. The
assembler turns the selections it accepts into quotations, and it is the only
code that writes the text of a quotation.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from search.eval.llm import Usage
from search.models import passage_id

#: The most quotations an answer holds.
MAX_QUOTES = 4

#: The two selectors behind the same interface.
SelectorName = Literal["baseline", "claude"]

#: Why the baseline selector answered in place of the language model.
Fallback = Literal["budget", "failed"]

#: Why the assembler rejected a selection, or the whole output.
Rejection = Literal[
    "malformed",
    "marked_unanswerable",
    "beyond_cap",
    "unknown_passage",
    "reversed_range",
    "outside_passage",
    "stale_text",
    "overlap",
]


class Speech(BaseModel):
    """The fields of one speech that a candidate and a quotation carry.

    Attributes:
        anf_id: The speech's key.
        party: One of the eight parties.
        speaker: The speaker as published, title and party included.
        date: The date of the protocol.
        title: The heading of the debate, when the source gives one.
        url: The address of the page that holds the speech.
        paragraphs: The speaker's paragraphs, in order.
        text_sha256: The hash of the paragraphs as they were read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anf_id: str
    party: str
    speaker: str
    date: date
    title: str | None
    url: str
    paragraphs: tuple[str, ...]
    text_sha256: str


class Candidate(BaseModel):
    """One passage a selector may quote from, as the search ranked it for a question.

    Attributes:
        rank: Its place in the ranking, from 1.
        passage_id: The passage's key.
        anf_id: The speech it belongs to.
        first: The zero-based index of its first paragraph in the speech.
        last: The zero-based index of its last paragraph, which is included.
        score: Its BM25 score.
        party: The speech's party.
        speaker: The speaker as published.
        date: The date of the protocol.
        title: The heading of the debate.
        url: The address of the speech's page.
        paragraphs: The passage's paragraphs, in order.
        text_sha256: The hash of the whole speech's paragraphs when the
            candidate was read, so a speech that has changed since is caught.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    passage_id: str
    anf_id: str
    first: int = Field(ge=0)
    last: int = Field(ge=0)
    score: float
    party: str
    speaker: str
    date: date
    title: str | None
    url: str
    paragraphs: tuple[str, ...]
    text_sha256: str

    @model_validator(mode="after")
    def _check_span(self) -> Candidate:
        if self.passage_id != passage_id(self.anf_id, self.first, self.last):
            raise ValueError(f"{self.passage_id}: the key does not match its speech and span.")
        if len(self.paragraphs) != self.last - self.first + 1:
            raise ValueError(f"{self.passage_id}: the paragraphs do not fill the span.")
        return self


class Selection(BaseModel):
    """One selection as a selector gives it.

    Attributes:
        passage_id: A candidate's passage id.
        first_paragraph: The first paragraph, numbered from 1 within the passage.
        last_paragraph: The last paragraph, which is included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    passage_id: str
    first_paragraph: int
    last_paragraph: int


class SelectorOutput(BaseModel):
    """The one form a selector's answer may take; nothing else is read.

    Validated in strict mode: a number written as text, a float or a boolean
    where an integer belongs, or any other field, fails the whole output.

    Attributes:
        answerable: Whether the candidates answer the question.
        selections: The selections, best first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    answerable: bool
    selections: tuple[Selection, ...]


class Rejected(BaseModel):
    """A selection the assembler did not accept, or an output it rejected whole.

    Attributes:
        position: The selection's place in the output, from 0, or ``None``
            when the whole output was rejected.
        reason: Why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int | None
    reason: Rejection


class Quote(BaseModel):
    """A quotation: whole paragraphs of one speech, verbatim from the corpus.

    Attributes:
        passage_id: The candidate the selection named.
        rank: That candidate's place in the ranking.
        first_paragraph: The first paragraph quoted, numbered from 1 within the passage.
        last_paragraph: The last paragraph quoted.
        anf_id: The speech.
        party: Its party.
        speaker: The speaker as published.
        date: The date of the protocol.
        title: The heading of the debate.
        url: The address of the speech's page.
        paragraphs: The paragraphs quoted, as the corpus holds them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str
    rank: int
    first_paragraph: int
    last_paragraph: int
    anf_id: str
    party: str
    speaker: str
    date: date
    title: str | None
    url: str
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        """The quotation's text: its paragraphs, separated by a blank line."""
        return "\n\n".join(self.paragraphs)


class Selected(BaseModel):
    """What a selector returned for one question, before the assembler reads it.

    Attributes:
        selector: The selector that answered.
        output: Its answer as JSON text, or ``None`` when none came back.
        fallback: Why the baseline answered in place of the language model, if it did.
        model: The model that answered, for the language model.
        stop_reason: Why the model stopped.
        error: What went wrong with a request, when something did.
        attempts: How many requests were sent.
        usage: The tokens all requests used.
        cost_usd: What all requests cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    selector: SelectorName
    output: str | None
    fallback: Fallback | None = None
    model: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    attempts: int = 0
    usage: Usage = Usage()
    cost_usd: float = 0.0


class Answer(BaseModel):
    """The answer to one question: its quotations and what was rejected on the way.

    Attributes:
        selected: What the selector returned.
        answerable: Whether the selector said the candidates answer the
            question, or ``None`` when its output was rejected whole.
        quotes: The quotations, in the selector's order.
        rejected: The selections, or the output, the assembler did not accept.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected: Selected
    answerable: bool | None
    quotes: tuple[Quote, ...]
    rejected: tuple[Rejected, ...]

    @property
    def abstained(self) -> bool:
        """Whether the answer holds no quotation."""
        return not self.quotes
