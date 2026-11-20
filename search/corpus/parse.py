"""Parsing the Riksdag's speech records into paragraphs the corpus can quote.

The source encodes a speech's text in three ways, depending on its age:

``html``
    From 2014/15: ``<p>`` elements, with ``<em>`` and ``<strong>`` inline.
``lines``
    2003/04 to 2013/14: plain text, one paragraph per line, where an emphasised
    run of words sits on a line of its own.
``wrapped``
    2002/03: plain text broken at the width of the printed column, with words
    hyphenated across lines and no mark for where a paragraph ends.

Words are never changed here. Markup is removed, entities are decoded and
whitespace is collapsed; in the ``lines`` layout the lines that emphasis split
are put back together. The ``wrapped`` layout cannot be read into paragraphs
without repairing hyphenated words by rule, so it is reported and left out.

A speech record also carries text that is the record's own: applause, a note
that the debate was closed, an interjection by somebody else. Those paragraphs
are set aside as notes, so that they can never be quoted as the speaker's
words. Every rule below was written against a count of what it matches in the
whole source.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from html.parser import HTMLParser
from typing import Literal, NamedTuple

SourceLayout = Literal["html", "lines", "wrapped"]

#: Characters a finished sentence, quotation or parenthesis can end with.
_TERMINAL = frozenset(".!?:;)\"'”’")
#: Punctuation that can only continue what came before it.
_CLOSING = frozenset(".,;:!?)")

#: Residue of a Word field code for the margin heading, left where a page broke:
#: ``STYLEREF Kantrubrik \* MERGEFORMAT Svar på interpellationer``.
FIELD_CODE_PREFIX = "STYLEREF "
FIELD_CODE_SEPARATOR = "MERGEFORMAT "

#: "Överläggningen var härmed avslutad.", in the handful of forms it takes.
_CLOSING_NOTE_WORDS = ("härmed", "avslutad")
_CLOSING_NOTE_MAX_CHARS = 100
#: "I detta anförande instämde Anna Andersson (S)."
_CONCURRENCE_PREFIX = "I detta anförande instämde"
#: The record narrates what the chair did, in the third person and past tense:
#: "Andre vice talmannen konstaterade att interpellanten inte var närvarande."
_CHAIRS = (
    "Talmannen",
    "Förste vice talmannen",
    "Andre vice talmannen",
    "Tredje vice talmannen",
)
_CHAIR_VERBS = ("konstaterade", "meddelade")
_CHAIR_NARRATION = tuple(f"{chair} {verb} " for chair in _CHAIRS for verb in _CHAIR_VERBS)
#: "Kammaren biföll konstitutionsutskottets förslag och valde därmed ..."
_CHAMBER_NARRATION = ("Kammaren biföll ", "Kammaren valde ")
#: "Då Anna Andersson, som framställt interpellationen, anmält att hon var
#: förhindrad att närvara vid sammanträdet medgav tredje vice talmannen att ..."
_LEAVE_OPENING = "Då "
_LEAVE_WORDS = (" medgav ", "talmannen")


class SpeechText(NamedTuple):
    """A speech's text, split into what the speaker said and what the record added.

    Attributes:
        paragraphs: The speaker's paragraphs, in order.
        notes: The record's own paragraphs, in order.
        furniture: How many paragraphs were page furniture and dropped.
    """

    paragraphs: tuple[str, ...]
    notes: tuple[str, ...]
    furniture: int


def normalise(text: str) -> str:
    """Collapse every run of whitespace to one space and trim the ends."""
    return " ".join(text.split())


def detect_layout(text: str) -> SourceLayout:
    """Tell which of the three encodings a speech's text uses."""
    if "<p>" in text or "<p " in text:
        return "html"
    if "\r\n" in text:
        return "lines"
    return "wrapped"


class _ParagraphCollector(HTMLParser):
    """Collect the text of each ``<p>`` element; inline tags leave no trace."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.stray: list[str] = []
        self._open: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            self._close()
            self._open = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self._close()

    def handle_data(self, data: str) -> None:
        if self._open is None:
            self.stray.append(data)
        else:
            self._open.append(data)

    def _close(self) -> None:
        if self._open is not None:
            self.paragraphs.append("".join(self._open))
            self._open = None


def html_paragraphs(text: str) -> list[str]:
    """Read the paragraphs of a speech in the ``html`` layout.

    Args:
        text: The speech's text as published.

    Returns:
        The normalised text of each non-empty ``<p>`` element, in order.

    Raises:
        ValueError: If any text sits outside the ``<p>`` elements. The source
            has none, and dropping it in silence would lose the speaker's words.
    """
    collector = _ParagraphCollector()
    collector.feed(text)
    collector.close()
    stray = normalise("".join(collector.stray))
    if stray:
        raise ValueError(f"Text outside the paragraph elements: {stray[:80]!r}")
    return [paragraph for raw in collector.paragraphs if (paragraph := normalise(raw))]


def _continues(previous: str, line: str) -> bool:
    """Tell whether *line* carries on the paragraph that *previous* belongs to.

    The source puts an emphasised run on its own line, and the pieces keep the
    spaces that joined them: ``"ha bara "``, ``"en"``, ``" kund eller "``. A
    line that opens with a space or with closing punctuation therefore
    continues, and so does one that follows a line left hanging on a space in
    mid-sentence. A line that ends a sentence and is followed by a line with no
    space in front is a paragraph break, which is the ordinary case.
    """
    if line[0].isspace() or line[0] in _CLOSING:
        return True
    trimmed = previous.rstrip()
    return previous != trimmed and trimmed[-1] not in _TERMINAL


def line_paragraphs(text: str) -> list[str]:
    """Read the paragraphs of a speech in the ``lines`` layout.

    Args:
        text: The speech's text as published.

    Returns:
        The normalised paragraphs, in order. Lines split by emphasis are joined
        exactly as they stand, with no character added between them.
    """
    groups: list[str] = []
    for line in text.split("\r\n"):
        if not line.strip():
            continue
        if groups and _continues(groups[-1], line):
            groups[-1] += line
        else:
            groups.append(line)
    return [normalise(group) for group in groups]


def read_paragraphs(text: str) -> list[str] | None:
    """Read a speech's text into normalised paragraphs, the record's included.

    Args:
        text: The speech's text as published.

    Returns:
        The paragraphs in order, or ``None`` for the ``wrapped`` layout, which
        the corpus does not read.

    Raises:
        ValueError: If the ``html`` layout carries text outside its paragraphs.
    """
    layout = detect_layout(text)
    if layout == "wrapped":
        return None
    return html_paragraphs(text) if layout == "html" else line_paragraphs(text)


def margin_heading(paragraph: str) -> str | None:
    """Return the margin heading a field-code residue carries, if it is one."""
    if not paragraph.startswith(FIELD_CODE_PREFIX):
        return None
    _, found, heading = paragraph.partition(FIELD_CODE_SEPARATOR)
    return heading.strip() if found and heading.strip() else None


def _is_furniture(paragraph: str, headings: Collection[str]) -> bool:
    """Tell page furniture from text: a field-code residue or a repeated heading."""
    return paragraph.startswith(FIELD_CODE_PREFIX) or paragraph in headings


def _closes_the_speech(paragraph: str) -> bool:
    """Tell whether the speaker's words end where *paragraph* begins."""
    if len(paragraph) <= _CLOSING_NOTE_MAX_CHARS:
        lowered = paragraph.lower().rstrip(".")
        if _CLOSING_NOTE_WORDS[0] in lowered and lowered.endswith(_CLOSING_NOTE_WORDS[1]):
            return True
    if paragraph.startswith((_CONCURRENCE_PREFIX, *_CHAIR_NARRATION, *_CHAMBER_NARRATION)):
        return True
    return paragraph.startswith(_LEAVE_OPENING) and all(word in paragraph for word in _LEAVE_WORDS)


def _is_inserted_note(paragraph: str) -> bool:
    """Tell whether *paragraph* is an insertion by the record: ``(Applåder)``."""
    return paragraph.startswith("(") and paragraph.endswith(")")


def split_speech_from_record(paragraphs: Sequence[str], headings: Collection[str]) -> SpeechText:
    """Set the record's own paragraphs aside from the speaker's.

    Three kinds of paragraph are not the speaker's words:

    * page furniture, which is dropped: the residue of a field code, or a
      paragraph that is nothing but a heading of the record;
    * an insertion in parentheses, which becomes a note: applause, a decision,
      an interjection by somebody else;
    * the paragraph that closes the speech, and everything after it: that the
      debate was over, who concurred, what the chair or the chamber did.

    Args:
        paragraphs: The normalised paragraphs of one speech record, in order.
        headings: The record's headings, debate titles and margin headings
            alike. A paragraph identical to one of them is furniture.

    Returns:
        The speaker's paragraphs, the record's notes, and the furniture count.
    """
    speech: list[str] = []
    notes: list[str] = []
    furniture = 0
    closed = False
    for paragraph in paragraphs:
        if _is_furniture(paragraph, headings):
            furniture += 1
        elif closed or _closes_the_speech(paragraph):
            closed = True
            notes.append(paragraph)
        elif _is_inserted_note(paragraph):
            notes.append(paragraph)
        else:
            speech.append(paragraph)
    return SpeechText(tuple(speech), tuple(notes), furniture)
