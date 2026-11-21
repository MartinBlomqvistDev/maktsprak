"""Data contracts for the search corpus."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from search.parties import VALID_PARTIES

#: How the source encodes the text of a speech the corpus keeps; see
#: ``search/corpus/parse.py``.
Layout = Literal["html", "lines"]

#: Sverigedemokraterna entered the Riksdag after the election in 2010.
FIRST_YEAR_WITH_SD = 2010

#: The labels of sessions before 2000/01 follow another form.
FIRST_SUPPORTED_YEAR = 2000

#: The address of one speech on the Riksdag's open-data host, data.riksdagen.se.
#: The page holds only that speech, with its protocol, the kind of debate,
#: "Anf. <number>" and the speaker above the text. How a sample of these
#: addresses answered, and what was found at the protocol's own pages at
#: www.riksdagen.se, is in ``data/reports/search/deep_links.json``.
DEEP_LINK = "https://data.riksdagen.se/anforande/{dok_id}-{number}/html"


def text_sha256(paragraphs: Sequence[str]) -> str:
    """Hash a speech's paragraphs, one per line, as the corpus identifies its text.

    The paragraphs of an :class:`Anforande` hold no line break, so the lines can
    be told apart again and two different splits of a speech never share a hash.

    Args:
        paragraphs: The speaker's paragraphs, in order.

    Returns:
        The SHA-256 of the paragraphs joined by line feeds, in hex.
    """
    return hashlib.sha256("\n".join(paragraphs).encode("utf-8")).hexdigest()


def deep_link(dok_id: str, anf_nummer: int) -> str:
    """Return the address of the page that holds one speech.

    The address ends in the speech's key and ``/html``:
    ``deep_link("HC09139", 36)`` is
    ``https://data.riksdagen.se/anforande/HC09139-36/html``.

    Args:
        dok_id: The protocol the speech belongs to.
        anf_nummer: The speech's number within the protocol.

    Returns:
        The address of the speech's page.
    """
    return DEEP_LINK.format(dok_id=dok_id, number=anf_nummer)


def session_label(start_year: int) -> str:
    """Name the session that opens in the autumn of *start_year*.

    Args:
        start_year: The calendar year the session opens.

    Returns:
        The label the Riksdag uses, for example ``2002/03``.

    Raises:
        ValueError: For years before 2000, where the labels follow another form.
    """
    if start_year < FIRST_SUPPORTED_YEAR:
        raise ValueError(
            f"The session opening in {start_year} is before {FIRST_SUPPORTED_YEAR}/01 "
            "and is not supported."
        )
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def session_start_year(session: str) -> int:
    """Return the year a session opens.

    Args:
        session: The session label, for example ``2002/03``.

    Returns:
        The calendar year the session opens.

    Raises:
        ValueError: If the label is malformed or names a session before 2000/01.
    """
    start, _, end = session.partition("/")
    well_formed = len(start) == 4 and start.isdigit() and len(end) == 2 and end.isdigit()
    if not well_formed or session != session_label(int(start)):
        raise ValueError(f"Not a session label: {session!r}. Expected the form '2002/03'.")
    return int(start)


def session_bounds(rm: str) -> tuple[date, date]:
    """Return the first and last day a protocol of session *rm* can carry.

    A session opens in September, or early October after an election, and runs
    until the next one opens a year later.

    Args:
        rm: The session label, for example ``2014/15``.

    Returns:
        The first of September of the year the session opens, and the thirtieth
        of September of the year after.

    Raises:
        ValueError: If *rm* is not a session label.
    """
    start_year = session_start_year(rm)
    return date(start_year, 9, 1), date(start_year + 1, 9, 30)


def _is_clean_line(text: str) -> bool:
    """Tell whether *text* is one line with a visible character and no whitespace at its ends.

    Zero-width spaces, byte order marks and soft hyphens are format characters
    (Unicode category Cf) that ``str.strip`` does not remove, so a line made of
    nothing else is not accepted either.

    Args:
        text: The speaker, paragraph or note to check.

    Returns:
        Whether *text* follows the rule.
    """
    has_visible = any(not char.isspace() and unicodedata.category(char) != "Cf" for char in text)
    return has_visible and text == text.strip() and len(text.splitlines()) == 1


class Anforande(BaseModel):
    """One speech in the chamber, as the Riksdag publishes it.

    The fields that identify the speech and its protocol keep or shorten the
    Riksdag's own names: ``dok_id``, ``dok_nummer`` and ``rel_dok_id`` as
    published, ``rm`` for ``dok_rm`` and ``anf_nummer`` for
    ``anforande_nummer``. A speech cannot be constructed in a state the corpus
    does not allow: every rule below is checked on creation.

    Attributes:
        anf_id: The key, ``{dok_id}-{anf_nummer}``, for example ``HC09139-36``.
        dok_id: The protocol the speech belongs to.
        rm: The parliamentary session, for example ``2024/25``.
        dok_nummer: The protocol's number within the session.
        anf_nummer: The speech's number within the protocol.
        date: The date of the protocol.
        speaker_raw: The speaker as published, title and party included. It is
            one line with a visible character and no whitespace at either end.
        person_id: The Riksdag's id for the speaker, when it gives one.
        party: One of the eight parties, after historical renames.
        is_reply: Whether the speech was a reply.
        title: The heading of the debate.
        activity: The kind of debate, when the source states it.
        rel_dok_id: The document the debate concerns, when the source states it.
        layout: How the source encoded the text.
        paragraphs: The speaker's paragraphs, in order. Only these are quoted. Each is
            one line with a visible character and no whitespace at either end.
        notes: Paragraphs that are the record's own, set aside and never quoted.
            Each follows the same rule as a paragraph.
        text_sha256: The hash of ``paragraphs``, see :func:`text_sha256`.
        url: The address of the page that holds the speech, see :func:`deep_link`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anf_id: str
    dok_id: str
    rm: str
    dok_nummer: int
    anf_nummer: int
    date: date
    speaker_raw: str
    person_id: str | None
    party: str
    is_reply: bool
    title: str | None
    activity: str | None
    rel_dok_id: str | None
    layout: Layout
    paragraphs: tuple[str, ...]
    notes: tuple[str, ...]
    text_sha256: str
    url: str

    @model_validator(mode="after")
    def _check_invariants(self) -> Anforande:
        if self.anf_id != f"{self.dok_id}-{self.anf_nummer}":
            raise ValueError(f"{self.anf_id}: the key does not match its protocol and number.")
        if self.party not in VALID_PARTIES:
            raise ValueError(f"{self.anf_id}: {self.party!r} is not one of the eight parties.")
        if not _is_clean_line(self.speaker_raw):
            raise ValueError(
                f"{self.anf_id}: the speaker must hold a visible character, sit on one line "
                "and have no whitespace at its ends."
            )
        if not self.paragraphs:
            raise ValueError(f"{self.anf_id}: a speech needs at least one paragraph.")
        for kind, texts in (("paragraph", self.paragraphs), ("note", self.notes)):
            for index, text in enumerate(texts):
                if not _is_clean_line(text):
                    raise ValueError(
                        f"{self.anf_id}: {kind} {index} must hold a visible character, "
                        "sit on one line and have no whitespace at its ends."
                    )
        if self.text_sha256 != text_sha256(self.paragraphs):
            raise ValueError(f"{self.anf_id}: the hash does not match the paragraphs.")
        if self.party == "SD" and self.date.year < FIRST_YEAR_WITH_SD:
            raise ValueError(f"{self.anf_id}: Sverigedemokraterna before {FIRST_YEAR_WITH_SD}.")
        first_day, last_day = session_bounds(self.rm)
        if not first_day <= self.date <= last_day:
            raise ValueError(f"{self.anf_id}: {self.date} lies outside session {self.rm}.")
        if self.url != deep_link(self.dok_id, self.anf_nummer):
            raise ValueError(f"{self.anf_id}: the address does not match its protocol and number.")
        return self


def passage_id(anf_id: str, first: int, last: int) -> str:
    """Return the key of the passage that runs from paragraph *first* to *last* of a speech.

    Both indexes are zero-based and both paragraphs belong to the passage, so
    ``passage_id("HC09139-36", 0, 2)`` is ``HC09139-36:0-2``.

    Args:
        anf_id: The key of the speech.
        first: The index of the passage's first paragraph in the speech.
        last: The index of its last paragraph.

    Returns:
        The key of the passage.
    """
    return f"{anf_id}:{first}-{last}"


class Passage(BaseModel):
    """A run of consecutive whole paragraphs from one speech.

    The search indexes hold passages, and a quotation is always one or more whole
    paragraphs of a single speech. A passage names its paragraphs by position
    in the speech and carries the fields a search filters on, so a hit can be
    narrowed without reading the speech. A passage cannot be constructed in a
    state the corpus does not allow: every rule below is checked on creation.

    Attributes:
        passage_id: The key, ``{anf_id}:{first}-{last}``, for example ``HC09139-36:0-2``.
        anf_id: The speech the passage belongs to.
        first: The zero-based index of the first paragraph in the speech.
        last: The zero-based index of the last paragraph, which is included.
        words: How many words those paragraphs hold, split on whitespace.
        party: One of the eight parties, taken from the speech.
        date: The date of the speech's protocol.
        person_id: The Riksdag's id for the speaker, when the speech has one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str
    anf_id: str
    first: int
    last: int
    words: int
    party: str
    date: date
    person_id: str | None

    @model_validator(mode="after")
    def _check_invariants(self) -> Passage:
        if self.first < 0 or self.last < 0:
            raise ValueError(f"{self.passage_id}: a paragraph index cannot be negative.")
        if self.first > self.last:
            raise ValueError(f"{self.passage_id}: the first paragraph comes after the last.")
        if self.passage_id != passage_id(self.anf_id, self.first, self.last):
            raise ValueError(
                f"{self.passage_id}: the key does not match its speech and paragraphs."
            )
        if self.words < 1:
            raise ValueError(f"{self.passage_id}: a passage needs at least one word.")
        if self.party not in VALID_PARTIES:
            raise ValueError(f"{self.passage_id}: {self.party!r} is not one of the eight parties.")
        return self
