"""The requests and answers of the spike's search service."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from search.parties import VALID_PARTIES
from search_service.spike.settings import AnnParams

Mode = Literal["hybrid", "lexical", "vector"]


class _Frozen(BaseModel):
    """Base for contracts: immutable, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchFilter(_Frozen):
    """What a search is narrowed to. A field left out allows everything.

    Attributes:
        party: One of the eight parties.
        start: The first day allowed.
        end: The last day allowed.
        person_id: The Riksdag's id of one speaker, digits only.
    """

    party: str | None = None
    start: date | None = None
    end: date | None = None
    person_id: str | None = Field(default=None, pattern=r"^[0-9]+$")

    @model_validator(mode="after")
    def _check(self) -> SearchFilter:
        if self.party is not None and self.party not in VALID_PARTIES:
            raise ValueError(f"{self.party!r} is not one of the eight parties.")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("The first day allowed comes after the last.")
        return self

    @property
    def empty(self) -> bool:
        """Whether the filter allows every passage."""
        return all(value is None for value in self.model_dump().values())


class SearchRequest(_Frozen):
    """One search.

    Attributes:
        text: The question, as written.
        vector_seed: The seed of the question's synthetic vector.
        filter: What the search is narrowed to; it applies to both legs.
        mode: Both legs fused, or one leg alone.
        ann: Search parameters of an approximate vector index.
    """

    text: str = Field(min_length=1, max_length=2000)
    vector_seed: int = Field(ge=0)
    filter: SearchFilter = SearchFilter()
    mode: Mode = "hybrid"
    ann: AnnParams = AnnParams()


class Hit(_Frozen):
    """One passage a search returned.

    Attributes:
        passage_id: The passage.
        party: Its speech's party.
        date: Its speech's date.
        score: Its score in the ranking returned.
        text: Its text.
    """

    passage_id: str
    party: str
    date: date
    score: float
    text: str


class Timings(_Frozen):
    """Where the time of one search went, in milliseconds; a part not timed apart is ``None``.

    Attributes:
        total_ms: The whole search, from the request to the hits with their text,
            the embedding of the question included.
        embed_ms: Embedding the question with the query model, when the service has one.
        filter_ms: Turning the filter into the passages it allows.
        lexical_ms: The BM25 leg.
        vector_ms: The vector leg.
        fusion_ms: Merging the two rankings.
        fetch_ms: Reading the hits' text and fields.
    """

    total_ms: float
    embed_ms: float | None = None
    filter_ms: float | None = None
    lexical_ms: float | None = None
    vector_ms: float | None = None
    fusion_ms: float | None = None
    fetch_ms: float | None = None


class EmbedRequest(_Frozen):
    """One question for the query model alone.

    Attributes:
        text: The question.
    """

    text: str = Field(min_length=1, max_length=2000)


class EmbedResponse(_Frozen):
    """What embedding one question took.

    Attributes:
        dim: The dimension of the vector.
        embed_ms: The time it took, in milliseconds.
    """

    dim: int
    embed_ms: float


class SearchResponse(_Frozen):
    """What a search returns.

    Attributes:
        hits: The best passages, best first.
        allowed: How many passages the filter allows, when the store counts them.
        timings: Where the time went.
    """

    hits: tuple[Hit, ...]
    allowed: int | None
    timings: Timings


class PartyViewRequest(_Frozen):
    """One question asked of each of the eight parties.

    Attributes:
        text: The question.
        vector_seed: The seed of its synthetic vector.
        filter: A filter applied to every party's search; it may not name a party.
        ann: Search parameters of an approximate vector index.
    """

    text: str = Field(min_length=1, max_length=2000)
    vector_seed: int = Field(ge=0)
    filter: SearchFilter = SearchFilter()
    ann: AnnParams = AnnParams()

    @model_validator(mode="after")
    def _no_party(self) -> PartyViewRequest:
        if self.filter.party is not None:
            raise ValueError("The view of the eight parties chooses the party itself.")
        return self


class PartyViewResponse(_Frozen):
    """The best passage of each party, in the site's order of the parties.

    Attributes:
        hits: The best hit of each party, or ``None`` when the party has none.
        timings: Where the time went.
    """

    hits: dict[str, Hit | None]
    timings: Timings
