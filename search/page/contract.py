"""The shape of what ``/api/fraga`` answers, and how the answer step's records map onto it.

The route answers in JSON Lines, one object per line. A search answers with a
``search`` line. An answer adds an ``answer`` line after it, so the list of
results can show while the selector is still choosing. The view with eight
parties answers with one ``parties`` line, which holds one answer per party in
the site's party order. The site's TypeScript types in ``web/src/lib/fraga.ts``
mirror these models, and the page's example answers are written in this shape.

Every text in a line is a speaker's paragraphs as the corpus holds them. A
quotation's paragraphs are the ones the assembler accepted; nothing here
writes or changes a word of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from search.answer.models import MAX_QUOTES, Answer, Candidate
from search.answer.models import Quote as AssembledQuote
from search.parties import PARTY_ORDER

#: How a question's candidates were ranked: the hybrid, or BM25 alone when the
#: question's vector did not come in time.
Retrieval = Literal["hybrid", "bm25"]

#: The selector that answered.
SelectorName = Literal["claude", "baseline"]

#: The most results a search line holds.
MAX_HITS = 10


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpeechFacts(_Frozen):
    """What a result shows about its speech beyond what a candidate carries.

    Attributes:
        rm: The session, such as ``2024/25``.
        dok_nummer: The protocol's number within the session.
        is_reply: Whether the speech is a reply in a debate.
    """

    rm: str
    dok_nummer: int
    is_reply: bool


class Hit(_Frozen):
    """One passage in the list of results.

    Attributes:
        rank: Its place in the ranking, from 1.
        passage_id: The passage's key.
        anf_id: The speech's key.
        rm: The session.
        dok_nummer: The protocol's number within the session.
        party: The speech's party.
        speaker: The speaker as the Riksdag publishes the name, title included.
        date: The date of the protocol.
        title: The heading of the debate, or ``None`` when the source gives none.
        is_reply: Whether the speech is a reply.
        url: The address of the speech's page.
        paragraphs: The passage's paragraphs, as the corpus holds them.
    """

    rank: int = Field(ge=1)
    passage_id: str
    anf_id: str
    rm: str
    dok_nummer: int
    party: str
    speaker: str
    date: date
    title: str | None
    is_reply: bool
    url: str
    paragraphs: tuple[str, ...]


class Quote(Hit):
    """A quotation: whole paragraphs of one passage, as the assembler accepted them.

    Attributes:
        first_paragraph: The first paragraph quoted, numbered from 1 within the passage.
        last_paragraph: The last paragraph quoted.
    """

    first_paragraph: int = Field(ge=1)
    last_paragraph: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_range(self) -> Quote:
        if self.last_paragraph - self.first_paragraph + 1 != len(self.paragraphs):
            raise ValueError(f"{self.passage_id}: the paragraphs do not fill the quoted range.")
        return self


class IndexFacts(_Frozen):
    """What the page says about the index that answered.

    Attributes:
        last_date: The date of the latest speech in the index.
        speeches: How many speeches it holds.
        passages: How many passages it holds.
    """

    last_date: date
    speeches: int
    passages: int


class SearchLine(_Frozen):
    """The list of results for a question.

    Attributes:
        type: Always ``search``.
        retrieval: ``hybrid``, or ``bm25`` when the vector did not come in time.
        vector_wait_ms: How long the question waited for its vector, or ``None``
            when none came.
        index: The index that answered.
        hits: At most ten passages, best first, under the question's filters.
    """

    type: Literal["search"] = "search"
    retrieval: Retrieval
    vector_wait_ms: int | None
    index: IndexFacts
    hits: tuple[Hit, ...] = Field(max_length=MAX_HITS)


class AnswerLine(_Frozen):
    """The answer step's answer: the quotations, or none when it abstained.

    Attributes:
        type: Always ``answer``.
        selector: ``claude`` or ``baseline``.
        model: The language model that selected, when one did.
        fallback: ``budget`` when the month's budget was spent and the baseline
            answered, ``failed`` when two requests failed and it did.
        answerable: Whether the selector said the candidates answer the
            question, or ``None`` when its output was rejected whole.
        quotes: At most four quotations, in the selector's order.
    """

    type: Literal["answer"] = "answer"
    selector: SelectorName
    model: str | None
    fallback: Literal["budget", "failed"] | None
    answerable: bool | None
    quotes: tuple[Quote, ...] = Field(max_length=MAX_QUOTES)


class PartyAnswer(_Frozen):
    """One party's answer in the view with eight parties.

    Attributes:
        party: The party whose speeches the question was searched among.
        selector: ``claude`` or ``baseline``.
        model: The language model that selected, when one did.
        fallback: As in :class:`AnswerLine`.
        answerable: As in :class:`AnswerLine`.
        quotes: The party's quotations, best first; none when nothing answered.
    """

    party: str
    selector: SelectorName
    model: str | None
    fallback: Literal["budget", "failed"] | None
    answerable: bool | None
    quotes: tuple[Quote, ...] = Field(max_length=MAX_QUOTES)

    @model_validator(mode="after")
    def _own_party(self) -> PartyAnswer:
        if any(quote.party != self.party for quote in self.quotes):
            raise ValueError(f"A quotation under {self.party} belongs to another party.")
        return self


class PartiesLine(_Frozen):
    """The view with eight parties: the question searched under each party in turn.

    Attributes:
        type: Always ``parties``.
        retrieval: As in :class:`SearchLine`; one vector serves every party.
        vector_wait_ms: As in :class:`SearchLine`.
        index: The index that answered.
        parties: One answer per party, in the site's party order.
    """

    type: Literal["parties"] = "parties"
    retrieval: Retrieval
    vector_wait_ms: int | None
    index: IndexFacts
    parties: tuple[PartyAnswer, ...]

    @model_validator(mode="after")
    def _every_party_in_order(self) -> PartiesLine:
        if tuple(answer.party for answer in self.parties) != PARTY_ORDER:
            raise ValueError("The view holds one answer per party, in the site's party order.")
        return self


class ErrorLine(_Frozen):
    """The line the service sends when the answer step fails after the list of results.

    The list of results has gone out by then and stands; the page says the
    answer broke off and asks the visitor to ask again.

    Attributes:
        type: Always ``error``.
        error: Always ``answer_failed``: the only failure the service reports as a line.
    """

    type: Literal["error"] = "error"
    error: Literal["answer_failed"]


def hit_of(candidate: Candidate, facts: Mapping[str, SpeechFacts]) -> Hit:
    """A candidate as the list of results shows it.

    Args:
        candidate: The candidate.
        facts: The session, protocol number and reply flag of each speech, by key.

    Returns:
        The result, with the candidate's paragraphs unchanged.
    """
    speech = facts[candidate.anf_id]
    return Hit(
        rank=candidate.rank,
        passage_id=candidate.passage_id,
        anf_id=candidate.anf_id,
        rm=speech.rm,
        dok_nummer=speech.dok_nummer,
        party=candidate.party,
        speaker=candidate.speaker,
        date=candidate.date,
        title=candidate.title,
        is_reply=speech.is_reply,
        url=candidate.url,
        paragraphs=candidate.paragraphs,
    )


def quote_of(quote: AssembledQuote, facts: Mapping[str, SpeechFacts]) -> Quote:
    """An assembled quotation as the page shows it.

    Args:
        quote: The quotation as the assembler wrote it.
        facts: The session, protocol number and reply flag of each speech, by key.

    Returns:
        The quotation, with the assembler's paragraphs unchanged.
    """
    speech = facts[quote.anf_id]
    return Quote(
        rank=quote.rank,
        passage_id=quote.passage_id,
        anf_id=quote.anf_id,
        rm=speech.rm,
        dok_nummer=speech.dok_nummer,
        party=quote.party,
        speaker=quote.speaker,
        date=quote.date,
        title=quote.title,
        is_reply=speech.is_reply,
        url=quote.url,
        paragraphs=quote.paragraphs,
        first_paragraph=quote.first_paragraph,
        last_paragraph=quote.last_paragraph,
    )


def answer_fields(answer: Answer, facts: Mapping[str, SpeechFacts]) -> dict[str, object]:
    """The fields an answer line and a party's answer share.

    Args:
        answer: The answer, as the answer step assembled it.
        facts: The session, protocol number and reply flag of each speech, by key.

    Returns:
        The selector, model, fallback, verdict and quotations.
    """
    selected = answer.selected
    return {
        "selector": selected.selector,
        "model": selected.model if selected.selector == "claude" else None,
        "fallback": selected.fallback,
        "answerable": answer.answerable,
        "quotes": tuple(quote_of(quote, facts) for quote in answer.quotes),
    }


def answer_line(answer: Answer, facts: Mapping[str, SpeechFacts]) -> AnswerLine:
    """The answer step's answer as an ``answer`` line."""
    return AnswerLine(**answer_fields(answer, facts))


def party_answer(party: str, answer: Answer, facts: Mapping[str, SpeechFacts]) -> PartyAnswer:
    """One party's answer in the view with eight parties."""
    return PartyAnswer(party=party, **answer_fields(answer, facts))


def search_line(
    candidates: Sequence[Candidate],
    facts: Mapping[str, SpeechFacts],
    retrieval: Retrieval,
    vector_wait_ms: int | None,
    index: IndexFacts,
) -> SearchLine:
    """A question's candidates as a ``search`` line."""
    return SearchLine(
        retrieval=retrieval,
        vector_wait_ms=vector_wait_ms,
        index=index,
        hits=tuple(hit_of(candidate, facts) for candidate in candidates),
    )
