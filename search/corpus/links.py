"""Checking that a speech's deep link leads to the speech.

Every quotation links to the page that holds its speech on the Riksdag's open-data
host, ``https://data.riksdagen.se/anforande/<anf_id>/html``. The page is small: the
protocol's title, the kind of debate and its heading, ``Anf. <number>`` with the
speaker, and the speech's text. This module draws a sample of speeches, requests each
address, and judges the one response it keeps per address: the status is 200, the
page names the speech and its speaker, and the opening words of every paragraph of
the speech occur in the page's text in order. Each page is also judged against the
speech before and the speech after in the same protocol, each where the corpus holds
one. A page that passes one of those judgements holds that neighbour as well: the
address is off by one, or the page holds the neighbour and the speech together, as a
page of the whole protocol would. Such a page does not pass. ``LIMITATION`` names the
pages that pass although they hold more than the speech or lack part of it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import unicodedata
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import date
from html.parser import HTMLParser

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from search.models import Layout, deep_link

logger = logging.getLogger(__name__)

#: How much of a paragraph is looked for in the page.
OPENING_CHARS = 60
#: How much of a page that did not pass is kept in the report.
EXCERPT_CHARS = 200

#: Elements whose content is not text a reader sees.
_HIDDEN = frozenset({"script", "style", "template"})

BUDGET_EXHAUSTED = "request budget exhausted"


class SpeechRef(BaseModel):
    """A speech as far as the sample needs to know it.

    Attributes:
        anf_id: The key, for example ``HC09139-36``.
        dok_id: The protocol the speech belongs to.
        anf_nummer: The speech's number within the protocol.
        rm: The parliamentary session.
        date: The date of the protocol.
        layout: How the source encoded the speech.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anf_id: str
    dok_id: str
    anf_nummer: int
    rm: str
    date: date
    layout: Layout

    @property
    def url(self) -> str:
        """The address of the speech, computed from its protocol and number."""
        return deep_link(self.dok_id, self.anf_nummer)


class SpeechText(BaseModel):
    """What a page is compared with: a speech's number, speaker and paragraphs.

    Attributes:
        anf_id: The key, for example ``HC09139-36``.
        anf_nummer: The speech's number within the protocol.
        speaker: The speaker as published, title and party included.
        paragraphs: The speech's paragraphs, in order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anf_id: str
    anf_nummer: int
    speaker: str
    paragraphs: tuple[str, ...] = Field(min_length=1)


class PageJudgement(BaseModel):
    """What one fetched page shows about one speech.

    Attributes:
        names_speech: Whether the page names the speech: ``Anf.`` and its number,
            followed by its speaker.
        first_opening_found: Whether the opening words of the speech's first
            paragraph occur in the page's text.
        every_opening_in_order: Whether the opening words of every paragraph of the
            speech occur in the page's text in order, each after the end of the
            match for the paragraph before.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    names_speech: bool
    first_opening_found: bool
    every_opening_in_order: bool

    @model_validator(mode="after")
    def _check_order(self) -> PageJudgement:
        if self.every_opening_in_order and not self.first_opening_found:
            raise ValueError(
                "The openings cannot all be found in order when the first is not found."
            )
        return self

    @property
    def is_match(self) -> bool:
        """Whether the page passes all three judgements."""
        return self.names_speech and self.first_opening_found and self.every_opening_in_order


NOT_FOUND = PageJudgement(
    names_speech=False, first_opening_found=False, every_opening_in_order=False
)


def sample_key(seed: int, name: str) -> str:
    """Rank *name* under *seed*: the same inputs give the same key on any machine."""
    return hashlib.sha256(f"{seed}|{name}".encode()).hexdigest()


def draw_sample(
    speeches: Iterable[SpeechRef],
    seed: int,
    per_session: int = 2,
    extra_recent_sessions: int = 2,
) -> list[SpeechRef]:
    """Draw speeches from distinct protocols, spread over the sessions.

    Each session gives *per_session* speeches, each from a different protocol.
    The *extra_recent_sessions* most recent sessions give one more. The draw
    depends only on the seed and on which speeches exist, never on the order
    they were given in.

    Args:
        speeches: Every speech to draw from.
        seed: Decides which speeches are drawn.
        per_session: How many speeches to take from each session.
        extra_recent_sessions: How many of the latest sessions give one more.

    Returns:
        The sample, ordered by session, then date, then the rank that
        :func:`sample_key` gives the speech's key under the seed. A session with
        fewer protocols than asked for gives all it has.

    Raises:
        ValueError: If a count is negative or *per_session* is zero.
    """
    if per_session < 1:
        raise ValueError(f"Each session must give at least one speech; got {per_session}.")
    if extra_recent_sessions < 0:
        raise ValueError(f"The extra sessions cannot be negative; got {extra_recent_sessions}.")

    by_session: dict[str, dict[str, list[SpeechRef]]] = defaultdict(lambda: defaultdict(list))
    for speech in speeches:
        by_session[speech.rm][speech.dok_id].append(speech)

    sessions = sorted(by_session)
    recent = set(sessions[-extra_recent_sessions:]) if extra_recent_sessions else set()
    sample: list[SpeechRef] = []
    for session in sessions:
        protocols = sorted(by_session[session], key=lambda dok_id: sample_key(seed, dok_id))
        take = per_session + (1 if session in recent else 0)
        for dok_id in protocols[:take]:
            sample.append(
                min(by_session[session][dok_id], key=lambda s: sample_key(seed, s.anf_id))
            )
    return sorted(sample, key=lambda s: (s.rm, s.date, sample_key(seed, s.anf_id)))


def neighbours_of(speech: SpeechRef, texts: Mapping[str, SpeechText]) -> tuple[SpeechText, ...]:
    """Return the speeches before and after *speech* in its protocol.

    Args:
        speech: The sampled speech.
        texts: The text of speeches by key, which must include the neighbours.

    Returns:
        The speech with the previous number and the speech with the next number,
        in that order, each where *texts* holds it: two, one or none.
    """
    keys = (f"{speech.dok_id}-{speech.anf_nummer + step}" for step in (-1, 1))
    return tuple(texts[key] for key in keys if key in texts)


def squeeze(text: str) -> str:
    """Remove every whitespace character and every invisible formatting character.

    The corpus puts a paragraph together from source lines and drops the soft
    hyphen, so the page and the corpus can differ in where whitespace and
    invisible characters fall.
    """
    return "".join(
        char for char in text if not char.isspace() and unicodedata.category(char) != "Cf"
    )


class _TextReader(HTMLParser):
    """Collect a page's text, leaving out scripts and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.chunks.append(data)


def page_text(html: str) -> str:
    """Return the text of a page: tags removed, entities decoded, scripts and styles left out."""
    reader = _TextReader()
    reader.feed(html)
    reader.close()
    return "".join(reader.chunks)


def opening_words(paragraph: str) -> str:
    """Return the opening words of a paragraph, in the form the page is searched for.

    That is the paragraph's first ``OPENING_CHARS`` characters, with whitespace and
    invisible formatting characters removed by :func:`squeeze`.
    """
    return squeeze(paragraph[:OPENING_CHARS])


def judge_page(html: str, speech: SpeechText) -> PageJudgement:
    """Judge a fetched page against one speech.

    The page's text is compared with the corpus's on the first ``OPENING_CHARS``
    characters of every paragraph, with whitespace and invisible formatting
    characters removed on both sides. The openings are looked for in order,
    each from the end of the match for the paragraph before. A paragraph that
    is missing from the page or out of place fails the judgement unless its
    opening words occur elsewhere in the stretch of page that the search for it
    covers, as ``LIMITATION`` states. A paragraph whose opening words are empty
    after whitespace is removed counts as not found.

    Args:
        html: The page as the server returned it.
        speech: The speech the page is meant to hold.

    Returns:
        What the page shows.
    """
    text = squeeze(page_text(html))
    names = squeeze(f"Anf. {speech.anf_nummer} {speech.speaker}") in text
    found = 0
    position = 0
    for paragraph in speech.paragraphs:
        opening = opening_words(paragraph)
        # An empty opening would be found at any position, so it counts as not found.
        at = text.find(opening, position) if opening else -1
        if at < 0:
            break
        found += 1
        position = at + len(opening)
    return PageJudgement(
        names_speech=names,
        first_opening_found=found > 0,
        every_opening_in_order=found == len(speech.paragraphs),
    )


def excerpt_of(html: str) -> str:
    """Return the first ``EXCERPT_CHARS`` characters of a page's text, whitespace collapsed."""
    return " ".join(page_text(html).split())[:EXCERPT_CHARS]


class BudgetExhausted(Exception):
    """The run has used every request it was allowed."""


class RequestBudget:
    """A count of HTTP requests that stops the run at a limit.

    Every request counts, a retry as much as a first try.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        """Count one request, or raise if the limit has been reached."""
        if self.used >= self.limit:
            raise BudgetExhausted(BUDGET_EXHAUSTED)
        self.used += 1


class HostThrottle:
    """Keeps requests to one host at least a set time apart, however many run at once."""

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval = min_interval_seconds
        self._sleep = sleep
        self._now = now
        self._lock = asyncio.Lock()
        self._last: float | None = None

    async def wait(self) -> None:
        """Return when the next request may be sent."""
        async with self._lock:
            if self._last is not None:
                remaining = self._last + self.min_interval - self._now()
                if remaining > 0:
                    await self._sleep(remaining)
            self._last = self._now()


class PageResponse(BaseModel):
    """What came back for one page.

    Attributes:
        status: The HTTP status of the last response, or ``None`` if there was none.
        html: The body of that response, when the status is 200.
        error: Why the page could not be fetched, when it could not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: int | None = None
    html: str | None = None
    error: str | None = None


#: Elements of the API's XML record of a speech, which its formatted page does not carry.
XML_RECORD_FIELDS = ("<anforande_nummer>", "<anforandetext>")


def is_xml_record(html: str) -> bool:
    """Tell whether a body holds the elements of the API's XML record of a speech.

    Args:
        html: The body as the server returned it.

    Returns:
        Whether every element of ``XML_RECORD_FIELDS`` occurs in the body.
    """
    return all(field in html for field in XML_RECORD_FIELDS)


def _is_transient(status: int) -> bool:
    """Tell a status worth another try from one that will not change."""
    return status == httpx.codes.TOO_MANY_REQUESTS or status >= httpx.codes.INTERNAL_SERVER_ERROR


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    throttle: HostThrottle,
    budget: RequestBudget,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> PageResponse:
    """Fetch one page, and try again after a failure that may not recur.

    A server error or a throttling answer is tried again, after a pause that
    grows with each attempt. So is a request that fails without a status: a
    timeout, a dropped connection, a body that cannot be decoded. Any other
    status, such as 404, is returned at once. A redirect is not followed: the
    address must answer for itself. A failed request comes back as a result
    that names the failure, so one page does not stop a run.

    Args:
        client: The HTTP client. It must not follow redirects itself.
        url: The page to fetch.
        throttle: Spaces the requests to the host.
        budget: Counts the requests and stops the run at its limit.
        attempts: How many times to try.
        backoff_seconds: The pause after the first failed attempt.
        sleep: How to pause.

    Returns:
        The last response, or the reason there was none. When the budget runs
        out before a retry, the result of the last attempt is kept and its
        error also names the budget.
    """
    result = PageResponse(error="not attempted")
    for attempt in range(1, attempts + 1):
        try:
            budget.spend()
            await throttle.wait()
            response = await client.get(url)
        except BudgetExhausted:
            if attempt == 1:
                return PageResponse(error=BUDGET_EXHAUSTED)
            return PageResponse(status=result.status, error=f"{result.error}; {BUDGET_EXHAUSTED}")
        except httpx.RequestError as error:
            result = PageResponse(error=type(error).__name__)
        else:
            ok = response.status_code == httpx.codes.OK
            result = PageResponse(
                status=response.status_code,
                html=response.text if ok else None,
                error=None if ok else f"HTTP {response.status_code}",
            )
            if ok or not _is_transient(response.status_code):
                return result
        if attempt < attempts:
            logger.warning(
                "Attempt %d of %d failed for %s: %s", attempt, attempts, url, result.error
            )
            await sleep(backoff_seconds * attempt)
    return result


class PageCheck(BaseModel):
    """The outcome for one sampled speech.

    Attributes:
        anf_id: The speech.
        session: Its parliamentary session.
        date: The date of its protocol.
        layout: How the source encoded it.
        url: The address that was fetched.
        status: The HTTP status of the page, or ``None`` if none came back.
        names_speech: Whether the page names the speech and its speaker.
        first_opening_found: Whether the page holds the opening words of the
            speech's first paragraph.
        every_opening_in_order: Whether it holds the opening words of every
            paragraph of the speech, in order.
        passed: Whether the status is 200, all three judgements hold and, when
            the speech has neighbours, the page fails the judgement against
            every one of them.
        neighbour_anf_ids: The speeches before and after the speech in its
            protocol that the corpus holds, in that order, which the page is
            judged against as well. Empty when the corpus holds neither.
        neighbour_rejected: Whether the page fails the judgement against every
            neighbour, so that it holds none of them. ``None`` when there is no
            neighbour or no page.
        xml_record: Whether the page that came back holds the elements of the
            API's XML record of a speech, ``XML_RECORD_FIELDS``.
        page_excerpt: The start of the page's text when the page came back and
            did not pass, so a failure can be read without fetching again.
        error: Why the page could not be fetched, when it could not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anf_id: str
    session: str
    date: date
    layout: Layout
    url: str
    status: int | None
    names_speech: bool
    first_opening_found: bool
    every_opening_in_order: bool
    passed: bool
    neighbour_anf_ids: tuple[str, ...]
    neighbour_rejected: bool | None
    xml_record: bool
    page_excerpt: str | None
    error: str | None

    @model_validator(mode="after")
    def _check_passed(self) -> PageCheck:
        holds_speech = (
            self.status == httpx.codes.OK
            and self.names_speech
            and self.first_opening_found
            and self.every_opening_in_order
        )
        neighbours_ruled_out = not self.neighbour_anf_ids or self.neighbour_rejected is True
        if self.passed != (holds_speech and neighbours_ruled_out):
            raise ValueError(
                "passed must be true exactly when the status is 200, all three judgements "
                "hold and, when the speech has neighbours, the page fails the judgement "
                "against every one of them."
            )
        return self


def check_for(
    speech: SpeechRef,
    text: SpeechText,
    response: PageResponse,
    neighbours: Sequence[SpeechText] = (),
) -> PageCheck:
    """Judge the page that came back for one speech.

    Args:
        speech: The sampled speech.
        text: What the page must hold: the speech's number, speaker and paragraphs.
        response: What the fetch returned.
        neighbours: The speeches before and after it in its protocol, each of
            which the page must fail the judgement against.

    Returns:
        The entry for the report. A page that did not come back counts as
        showing nothing.
    """
    judgement = NOT_FOUND
    rejected: bool | None = None
    passed = False
    xml_record = False
    excerpt: str | None = None
    if response.html is not None:
        judgement = judge_page(response.html, text)
        xml_record = is_xml_record(response.html)
        if neighbours:
            rejected = not any(judge_page(response.html, n).is_match for n in neighbours)
        passed = response.status == httpx.codes.OK and judgement.is_match and rejected is not False
        if not passed:
            excerpt = excerpt_of(response.html)
    return PageCheck(
        anf_id=speech.anf_id,
        session=speech.rm,
        date=speech.date,
        layout=speech.layout,
        url=speech.url,
        status=response.status,
        names_speech=judgement.names_speech,
        first_opening_found=judgement.first_opening_found,
        every_opening_in_order=judgement.every_opening_in_order,
        passed=passed,
        neighbour_anf_ids=tuple(n.anf_id for n in neighbours),
        neighbour_rejected=rejected,
        xml_record=xml_record,
        page_excerpt=excerpt,
        error=response.error,
    )


class Totals(BaseModel):
    """Counts over a set of checked speeches.

    Attributes:
        sampled: How many speeches were in the set.
        fetched: How many pages came back with status 200.
        never_requested: How many pages were never requested, because the
            request budget ran out first.
        retry_cut_by_budget: How many pages were requested and failed, and were
            not requested again because the request budget ran out.
        names_speech: How many pages name the speech and its speaker.
        first_opening_found: How many hold the opening words of the first paragraph.
        every_opening_in_order: How many hold the opening words of every
            paragraph, in order.
        passed: How many have status 200, pass every judgement and hold none of
            their neighbouring speeches.
        neighbour_checked: How many pages were also judged against at least one
            neighbouring speech.
        neighbour_rejected: How many of those failed the judgement against every
            neighbour, so that they hold none of their neighbours.
        passed_neighbour_checked: How many of the pages that passed were judged
            against at least one neighbouring speech.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sampled: int
    fetched: int
    never_requested: int
    retry_cut_by_budget: int
    names_speech: int
    first_opening_found: int
    every_opening_in_order: int
    passed: int
    neighbour_checked: int
    neighbour_rejected: int
    passed_neighbour_checked: int


def totals_of(checks: Sequence[PageCheck]) -> Totals:
    """Count what a set of checks found."""
    cut = f"; {BUDGET_EXHAUSTED}"
    return Totals(
        sampled=len(checks),
        fetched=sum(1 for c in checks if c.status == httpx.codes.OK),
        never_requested=sum(1 for c in checks if c.error == BUDGET_EXHAUSTED),
        retry_cut_by_budget=sum(1 for c in checks if c.error and c.error.endswith(cut)),
        names_speech=sum(c.names_speech for c in checks),
        first_opening_found=sum(c.first_opening_found for c in checks),
        every_opening_in_order=sum(c.every_opening_in_order for c in checks),
        passed=sum(c.passed for c in checks),
        neighbour_checked=sum(1 for c in checks if c.neighbour_rejected is not None),
        neighbour_rejected=sum(1 for c in checks if c.neighbour_rejected),
        passed_neighbour_checked=sum(
            1 for c in checks if c.passed and c.neighbour_rejected is not None
        ),
    )


def totals_by_session(checks: Sequence[PageCheck]) -> dict[str, Totals]:
    """Count what the checks found, session by session in chronological order."""
    grouped: dict[str, list[PageCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.session].append(check)
    return {session: totals_of(grouped[session]) for session in sorted(grouped)}


def totals_by_layout(checks: Sequence[PageCheck]) -> dict[str, Totals]:
    """Count what the checks found, by how the source encoded the speech."""
    grouped: dict[str, list[PageCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.layout].append(check)
    return {layout: totals_of(grouped[layout]) for layout in sorted(grouped)}


def verdict_of(totals: Totals) -> str:
    """Describe the totals in words, for the report's verdict.

    Each count follows a label and a colon, so that the sentence reads the same
    whatever the count.
    """
    budget = ""
    if totals.never_requested:
        budget += (
            " Sampled pages never requested, because the request budget ran out: "
            f"{totals.never_requested}."
        )
    if totals.retry_cut_by_budget:
        budget += (
            " Sampled pages that failed and were not requested again, because the request "
            f"budget ran out: {totals.retry_cut_by_budget}."
        )
    if totals.fetched == 0:
        return "No page could be fetched, so the deep links are not checked." + budget
    neighbour = (
        "Pages judged against at least one neighbouring speech: "
        f"{totals.neighbour_checked}; of these, pages that hold none of their neighbours: "
        f"{totals.neighbour_rejected}. "
        if totals.neighbour_checked
        else ""
    )
    return (
        f"Speeches sampled: {totals.sampled}. Pages that came back with status 200: "
        f"{totals.fetched}. Of those, pages that name the speech and its speaker: "
        f"{totals.names_speech}; that hold the opening words of the speech's first "
        f"paragraph: {totals.first_opening_found}; that hold the opening words of every "
        f"paragraph of the speech in order: {totals.every_opening_in_order}. {neighbour}"
        f"Pages that pass every judgement: {totals.passed}; of these, pages judged against "
        f"at least one neighbouring speech: {totals.passed_neighbour_checked}." + budget
    )


class OtherAddressForm(BaseModel):
    """An address the Riksdag publishes for a speech that does not lead to the speech.

    Attributes:
        address: The form of the address.
        result: What was found when it was tried.
        how_established: How the result was established.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    address: str
    result: str
    how_established: str


OTHER_ADDRESS_FORMS = (
    OtherAddressForm(
        address=(
            "https://www.riksdagen.se/sv/dokument-och-lagar/dokument/protokoll/"
            "protokoll_<dok_id>/#anf<number>, the form the Riksdag's API returns in its "
            "protokoll_url_www field"
        ),
        result=(
            "On 23 of the 24 pages fetched, the protocol page is a landing page that carries "
            "neither the anchor nor the speech's text. The anchor was on 1 of the 24, the one "
            "page (2005/06) that holds its protocol's full text. On 2026-11-20, a browser opened "
            "one page, that of protocol 2024/25:139: once loaded, it had no element with the id "
            "of speech 36, did not scroll, and did not hold the speech's words."
        ),
        how_established=(
            "A script fetched the protocol page of one speech from each of the 24 sessions and "
            "looked for an element with the id and for an <a> element with the name; a browser "
            "then opened the page of protocol 2024/25:139."
        ),
    ),
    OtherAddressForm(
        address=(
            "https://www.riksdagen.se/sv/dokument-och-lagar/dokument/protokoll/"
            "protokoll_<dok_id>/html/#anf<number>, the full-text page below the protocol page"
        ),
        result=(
            "The anchor was on 4 of 8 pages, all four from protocols before 2014/15; the four "
            "pages from 2015/16 on had none, although all eight held the speech's opening "
            "words. On 2026-11-20, a browser opened one page, that of protocol 2003/04:100: "
            "it scrolled to speech 132, and the page has 197 named anchors."
        ),
        how_established=(
            "A script fetched the full-text page of 8 of the 24 protocols, spread over the "
            "sessions, and looked for the anchor as above; a browser then opened the page of "
            "protocol 2003/04:100."
        ),
    ),
)

SAMPLING = (
    "Each session gives speeches from distinct protocols, and the most recent sessions give "
    "one more. Protocols and speeches are ranked by the SHA-256 of the seed and their id, so "
    "the draw depends on the seed and the corpus alone."
)
JUDGEMENT = (
    "A page passes when its status is 200, it names the speech (Anf. and the number, followed "
    "by the speaker as the corpus holds it), and the first "
    f"{OPENING_CHARS} characters of every paragraph of the speech occur in its text in order, "
    "each after the end of the match before it. Both sides are compared after removing tags, "
    "decoding entities and removing all whitespace and invisible formatting characters, "
    "because the corpus puts some paragraphs together from several source lines. The page "
    "is also judged the same way against the speech before and the speech after it in its "
    "protocol, each where the corpus holds one, and a page that passes either of those "
    "judgements as well does not pass."
)
LIMITATION = (
    "One response per address was examined, and only the HTML the server returned; nothing "
    "was rendered. An address that failed in a way that may not recur, with a server error, "
    "a throttling answer or no status at all, was requested again, up to the number of "
    "attempts the run allowed. Only the opening words of each paragraph are compared. A page "
    "that drops or changes text inside a paragraph, after its opening words, therefore "
    "passes. A paragraph that is missing from the page or out of place goes unnoticed when "
    "its opening words occur between the match for the paragraph before it and the opening "
    "words of the paragraph after it; for the first paragraph, anywhere earlier on the page; "
    "for the last, anywhere later. Only the speeches next to the speech in its protocol are "
    "looked for, and only whole: a page that also holds a speech that is not a neighbour, a "
    "second copy of the speech, or a neighbour present only in part or headed with another "
    "title for its speaker, passes. The server can "
    "also answer with the API's XML record of the speech in place of the formatted page; "
    'such a page holds the speech\'s text but has no "Anf." heading, so it fails on the '
    "name. How many addresses answer this way is not known."
)


def xml_record_note(checks: Sequence[PageCheck]) -> str:
    """Name the entries of a run whose page looks like the API's XML record of the speech.

    Such an entry has status 200, the opening words of every paragraph of the
    speech found, the name not found, and a body that holds the elements in
    ``XML_RECORD_FIELDS``.

    Args:
        checks: One entry per sampled speech.

    Returns:
        A sentence that names those entries, or an empty string when there are none.
    """
    names = [
        c.anf_id
        for c in checks
        if c.status == httpx.codes.OK
        and c.every_opening_in_order
        and not c.names_speech
        and c.xml_record
    ]
    if not names:
        return ""
    fields = " and ".join(XML_RECORD_FIELDS)
    return (
        " Entries of this run whose page has status 200, holds the opening words of every "
        "paragraph of the speech, lacks the name, and holds the elements "
        f"{fields}, as the API's XML record does: {', '.join(names)}."
    )


class DeepLinkReport(BaseModel):
    """The result of checking a sample of deep links.

    Attributes:
        inputs_sha256: The build report's hash over the dataset files the corpus
            was built from, which identifies the corpus the sample was drawn from.
        corpus_speeches: How many speeches that corpus holds.
        seed: The seed of the draw.
        per_session: How many speeches each session gave.
        extra_recent_sessions: How many of the latest sessions gave one more.
        sampling: How the draw works.
        judgement: What a page must show to pass.
        request_limit: The most HTTP requests the run was allowed, retries included.
        requests_made: How many it made.
        verdict: What the totals show, in words.
        limitation: What the check does not cover, and which entries of the run
            look like the API's XML record of the speech.
        totals: The counts over every sampled speech.
        by_session: The same counts, session by session.
        by_layout: The same counts by how the source encoded the speech.
        entries: One entry per sampled speech.
        other_address_forms: The address forms that do not lead to the speech, and
            what was found when each was tried.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs_sha256: str
    corpus_speeches: int
    seed: int
    per_session: int
    extra_recent_sessions: int
    sampling: str
    judgement: str
    request_limit: int
    requests_made: int
    verdict: str
    limitation: str
    totals: Totals
    by_session: dict[str, Totals]
    by_layout: dict[str, Totals]
    entries: tuple[PageCheck, ...]
    other_address_forms: tuple[OtherAddressForm, ...]


def report_of(
    checks: Sequence[PageCheck],
    *,
    inputs_sha256: str,
    corpus_speeches: int,
    seed: int,
    per_session: int,
    extra_recent_sessions: int,
    budget: RequestBudget,
) -> DeepLinkReport:
    """Assemble the report from the checked speeches and the request count.

    Args:
        checks: One entry per sampled speech.
        inputs_sha256: The build report's hash of the corpus's inputs.
        corpus_speeches: How many speeches the corpus holds.
        seed: The seed of the draw.
        per_session: How many speeches each session gave.
        extra_recent_sessions: How many of the latest sessions gave one more.
        budget: The request count of the run.

    Returns:
        The report. Its limitation is ``LIMITATION``, followed by the sentence
        of :func:`xml_record_note` when the run has entries it names.
    """
    totals = totals_of(checks)
    return DeepLinkReport(
        inputs_sha256=inputs_sha256,
        corpus_speeches=corpus_speeches,
        seed=seed,
        per_session=per_session,
        extra_recent_sessions=extra_recent_sessions,
        sampling=SAMPLING,
        judgement=JUDGEMENT,
        request_limit=budget.limit,
        requests_made=budget.used,
        verdict=verdict_of(totals),
        limitation=LIMITATION + xml_record_note(checks),
        totals=totals,
        by_session=totals_by_session(checks),
        by_layout=totals_by_layout(checks),
        entries=tuple(checks),
        other_address_forms=OTHER_ADDRESS_FORMS,
    )
