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

Words are never changed here. Markup is removed, entities are decoded,
whitespace is collapsed and the soft hyphen is dropped; in the ``lines`` layout
the lines that emphasis split are put back together exactly as they stand, with
no character added between them. The ``wrapped`` layout cannot be read into
paragraphs without repairing hyphenated words by rule, so it is reported and
left out.

A speech record also carries text that is the record's own: applause, a note
that the debate was closed, an interjection by somebody else, the page's
heading. Those paragraphs are set aside, so that they can never be quoted as
the speaker's words. A paragraph is set aside whole or kept whole, with one
exception: where another person's interjection or the protocol's page footer
runs into the speaker's words, the paragraph is cut at the edge of that text,
and the speaker's part is kept exactly as it stands. An insertion of the
record's inside one of the speaker's paragraphs is left as published: 4 kept
speeches carry one, ``(Applåder)`` in three and ``(Applåd)`` in one, between
two sentences in three of the speeches and at the end of its paragraph in one.
The comments below give what each rule matches in the whole source, and each
paragraph set aside carries the name of its rule.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser
from typing import Literal, NamedTuple

SourceLayout = Literal["html", "lines", "wrapped"]

#: Characters a finished sentence, quotation or parenthesis can end with.
_TERMINAL = frozenset(".!?:;)\"'”’")
#: Punctuation that can only continue what came before it.
_CLOSING = frozenset(".,;:!?)")

#: The soft hyphen marks where a word may break and is invisible anywhere else.
#: It occurs 4 times inside paragraphs in the whole source; no zero-width
#: character or byte-order mark occurs inside any text.
_SOFT_HYPHEN = "\u00ad"

#: The inline elements the source uses: ``em`` 17 558 times and ``strong`` 136.
#: The only other element is ``p``, 1 151 090 times.
_INLINE_TAGS = frozenset({"em", "strong"})

#: Residue of a Word field code for the margin heading, left where a page broke:
#: ``STYLEREF Kantrubrik \* MERGEFORMAT Svar på interpellationer``.
FIELD_CODE_PREFIX = "STYLEREF "
FIELD_CODE_SEPARATOR = "MERGEFORMAT "

#: Rules that mark page furniture, which is dropped. Every other rule marks a
#: note of the record's, which is kept beside the speech.
FURNITURE_RULES = frozenset(
    {
        "field_code",
        "heading",
        "heading_pieces",
        "heading_with_subtitle",
        "heading_line_break",
        "title_fragment",
        "title_misprint",
        "page_footer",
    }
)

# The lines layout.

#: A short line between a line that ends a sentence and a line that continues
#: it is the head of a sentence whose first words are emphasised:
#: ``"utmaningar. "``, ``"En"``, ``" mening handlar om ..."``. Of the 14 106
#: lines in the source that stand as an emphasised run inside a sentence,
#: 14 028 are shorter than this. In the position of the head, leaving aside
#: salutations and lines after a colon, 106 lines are this long or longer: 85
#: end a sentence and are followed by a line that opens a paragraph by the test
#: below, and 21 end no sentence and are continued by the line after them.
#: After a colon the bound does not apply: the 4 longer runs there are three
#: titles and one quotation that the sentence introduces.
_EMPHASIS_MAX_CHARS = 100
#: A salutation never continues the sentence before it, and opens a paragraph
#: when it follows a whole sentence.
_SALUTATIONS = ("Fru talman!", "Herr talman!")
#: A line that opens with a space continues the paragraph before it, except
#: where both are whole sentences at least this long: 722 lines in the source
#: pass that test and are new paragraphs begun with a stray space,
#: ``" Fru talman! Det utvidgade transportörsansvaret har ..."``. One more, in
#: H109126-1, is a new paragraph because it opens with a salutation.
_PARAGRAPH_MIN_CHARS = 60
#: The character the source put where a list item's bullet was lost.
_LOST_BULLET = "?"

# Headings in the html layout.

#: The longest paragraph that can be a piece of a heading.
_HEADING_PIECE_MAX_CHARS = 200
#: The most paragraphs a heading is broken over: ``"Gransknings-"``,
#: ``"betänkandeVissa frågor om regeringens ansvar"``, ``"för förvaltningen"``,
#: ``"och statliga bolag (kapitel 3)"``.
_HEADING_MAX_PIECES = 8
#: The fewest letters and digits a heading needs to be matched in pieces.
_HEADING_MIN_SQUASHED = 12
#: The fewest letters and digits a piece of the record's own title needs.
_TITLE_FRAGMENT_MIN_SQUASHED = 8
#: The hyphen and the en dash, which a heading and its broken copy use in each
#: other's place.
_DASHES = ("-", "\u2013")

# The record's own paragraphs.

#: "Överläggningen var härmed avslutad.", in the handful of forms it takes.
_CLOSING_NOTE_WORDS = ("härmed", "avslutad")
_CLOSING_NOTE_MAX_CHARS = 100
#: The same note with "därmed" or the misspelling "härmad", matched only as the
#: whole four-word note, since a loose test for those words would also find a
#: speaker's sentence: 10 paragraphs.
_CLOSING_NOTE_SUBJECTS = frozenset(
    {"Överläggningen", "Interpellationsdebatten", "Frågestunden", "Debatten"}
)
_CLOSING_NOTE_ADVERBS = frozenset({"därmed", "härmed", "härmad"})
#: "I detta anförande instämde Johan Pehrson (fp)."
_CONCURRENCE_PREFIX = "I detta anförande instämde"
#: The record narrates what the chair did, in the third person and past tense:
#: "Andre vice talmannen konstaterade att interpellanten inte var närvarande i
#: kammaren och förklarade överläggningen avslutad." In 3 speeches the chair is
#: the ålderspresident.
_CHAIRS = (
    "Talmannen",
    "Förste vice talmannen",
    "Andre vice talmannen",
    "Tredje vice talmannen",
    "Ålderspresidenten",
    "Tjänstgörande ålderspresidenten",
)
_CHAIR_VERBS = ("konstaterade", "meddelade")
_CHAIR_NARRATION = tuple(f"{chair} {verb} " for chair in _CHAIRS for verb in _CHAIR_VERBS)
#: "Kammaren biföll konstitutionsutskottets förslag och valde därmed ...". A
#: speaker can recall a decision in the same words ("Kammaren biföll i våras en
#: motion ..."), so only the forms the record uses are matched.
_CHAMBER_NARRATION = (
    "Kammaren biföll konstitutionsutskottets förslag",
    "Kammaren valde med acklamation ",
    "Kammaren valde i enlighet med ",
)
#: "Då Åsa Domeij, som framställt interpellationen, anmält att hon var
#: förhindrad att närvara vid sammanträdet medgav tredje vice talmannen att
#: Ingegerd Saarinen i stället fick delta i överläggningen." In 37 speeches the
#: sentence takes another form. In 34 the chair closes the debate, "...
#: förklarade talmannen överläggningen avslutad.", with any of the four chairs
#: from talmannen to tredje vice talmannen and with "överläggningen" or
#: "interpellationsdebatten". In 3 the leave is given by another chair, "...
#: medgav tjänstgörande ålderspresidenten att ...".
_LEAVE_OPENING = "Då "
_LEAVE_CHAIRS = ("talmannen", "ålderspresidenten")
_LEAVE_GRANTED = " medgav "
_LEAVE_CLOSED = (" förklarade ", "avslutad.")
#: The record of an election that follows the nominator's speech in 11
#: speeches, with the lists of members and the counts of the ballot after it:
#: "Företogs val av 17 ledamöter i utskotten.", "Val av 17 ledamöter i samtliga
#: utskott företogs." and "Efter härmed avslutad överläggning företogs val med
#: slutna sedlar."
_ELECTION_PREFIXES = ("Företogs val", "Efter härmed avslutad överläggning")
_ELECTION_PASSIVE = ("Val av ", " företogs.")
#: The record reads a submitted document into the protocol, with its
#: signatures: "Följande skrivelse hade kommit in:", "Från
#: konstitutionsutskottet hade följande skrivelse inkommit:", "Den överlämnade
#: skrivelsen hade följande lydelse:". The rule matches in 18 records of the
#: source: 17 speeches that the corpus keeps and one record outside the eight
#: parties, H909123-89. In one more record, HB09135-194, the reading follows
#: the narration of a demand and is set aside with everything after that
#: narration. The whole sentence is matched, since a speaker can end a sentence
#: about a letter with a colon too.
_DOCUMENT_READ = re.compile(
    r"(?:Från .+ hade följande skrivelse (?:inkommit|kommit in)"
    r"|Följande skrivelser? (?:med föreslagna namn )?hade kommit in(?: från .+)?"
    r"|(?:Den överlämnade skrivelsen|De inkomna skrivelserna) hade följande lydelse):"
)
#: "Håkan Svenneling m.fl. (V) hade, med stöd av 2 kap. 22 § första stycket
#: regeringsformen, genom skrivelser framställt yrkande om att ...", once. The
#: record's past perfect is matched: ", med stöd av " occurs in 266 texts of
#: the source, and in a speaker's paragraph in 260 of them.
_DEMAND_WORDS = (" hade, med stöd av ", " framställt yrkande ")
#: "Konstitutionsutskottet hade enhälligt föreslagit att professorn ... skulle
#: väljas till ny justitieombudsman.", where no document was read before it: 2
#: speeches.
_COMMITTEE_NARRATION = "Konstitutionsutskottet hade enhälligt föreslagit att "
#: The record of a vote, glued to the speech before it in 1 speech.
_VOTE = "Votering:"

#: An interjection by another person that opens a paragraph and runs on into
#: the next ones (22 times) or is never closed (once):
#: ``(GUNILLA CARLSSON i Tyresö (m): Jag sade att ...``,
#: ``(TALMANNEN: Det är inte tillåtet att störa från läktaren. ...``.
_INTERJECTION_OPENING = re.compile(
    r"\((?:[A-ZÅÄÖÉÜ][A-ZÅÄÖÉÜa-zåäöéü\-. ]*[A-ZÅÄÖÉÜa-zåäöéü] \((?:[A-Za-z]+|-)\)"
    r"|[A-ZÅÄÖ ]*TALMANNEN|(?:TJÄNSTGÖRANDE )?ÅLDERSPRESIDENTEN)\s*:"
)
#: The most paragraphs after the opening one that an interjection runs over.
_INTERJECTION_REACH = 3
#: An interjection the record gives without parentheses, once:
#: "JENS HOLM (V): Jodå. Det kan jag."
_UNBRACKETED_INTERJECTION = re.compile(
    r"[A-ZÅÄÖ]{2,}(?: [A-ZÅÄÖ-]{2,})+ \((?:S|M|C|V|FP|KD|MP|SD|L|s|m|c|v|fp|kd|mp|sd|l)\):"
)
#: The page footer that split one speaker's sentence across a page break, once:
#: "... att ta sig Vid protokollet", "ANDERS NORIN", "/Eva-Lena Ekman mänskliga
#: rättigheter. ..."
_FOOTER_SIGNATURE = " Vid protokollet"
_FOOTER_CLERK = "/"
#: The clerk signs with a first name and a surname.
_CLERK_WORDS = 2


class SetAside(NamedTuple):
    """A paragraph that is not the speaker's, with the rule that says so.

    Attributes:
        rule: The name of the rule that set the paragraph aside.
        text: The paragraph.
    """

    rule: str
    text: str


class SpeechText(NamedTuple):
    """A speech's text, split into what the speaker said and what the record added.

    Attributes:
        paragraphs: The speaker's paragraphs, in order.
        set_aside: The paragraphs that are not the speaker's, in order.
    """

    paragraphs: tuple[str, ...]
    set_aside: tuple[SetAside, ...]

    @property
    def notes(self) -> tuple[str, ...]:
        """The record's notes, in order: every paragraph set aside that is not page furniture."""
        return tuple(item.text for item in self.set_aside if item.rule not in FURNITURE_RULES)

    @property
    def furniture(self) -> int:
        """How many paragraphs were page furniture and dropped."""
        return sum(item.rule in FURNITURE_RULES for item in self.set_aside)


class Headings(NamedTuple):
    """The record's headings, in the forms the heading rules compare.

    Attributes:
        exact: The headings as collected.
        lowered: The headings in lower case.
        squashed: The headings reduced to their letters and digits in lower
            case, for those with enough of them to be matched in pieces.
        respaced: The same headings in lower case, without hyphens, en dashes or
            spaces.
        pairs: The debate titles run together with their subtitles, reduced
            to their letters and digits in lower case, as the margin heading
            of a page prints the two.
    """

    exact: frozenset[str]
    lowered: frozenset[str]
    squashed: frozenset[str]
    respaced: frozenset[str]
    pairs: frozenset[str]


def index_headings(headings: Iterable[str], pairs: Iterable[tuple[str, str]] = ()) -> Headings:
    """Prepare the record's headings for the heading rules.

    Args:
        headings: The debate titles, subtitles and margin headings.
        pairs: The debate titles that come with a subtitle, each with its
            subtitle.

    Returns:
        The headings in every form the heading rules compare.
    """
    exact = frozenset(headings)
    matched = [heading for heading in exact if len(_squash(heading)) >= _HEADING_MIN_SQUASHED]
    joined = (_title_with_subtitle(title, subtitle) for title, subtitle in pairs)
    return Headings(
        exact=exact,
        lowered=frozenset(heading.lower() for heading in exact),
        squashed=frozenset(_squash(heading) for heading in matched),
        respaced=frozenset(_respace(heading) for heading in matched),
        pairs=frozenset(pair for pair in joined if pair),
    )


NO_HEADINGS = index_headings(())


def normalise(text: str) -> str:
    """Collapse every run of whitespace to one space, trim the ends, drop the soft hyphen.

    The soft hyphen only marks where a line may break, so dropping it gives the
    text a reader sees; it is the one invisible character the source carries.

    Args:
        text: The text to normalise.

    Returns:
        The normalised text, empty if nothing visible is left.
    """
    return " ".join(text.replace(_SOFT_HYPHEN, "").split())


def detect_layout(text: str) -> SourceLayout:
    """Tell which of the three encodings a speech's text uses.

    The paragraph element is found in upper or lower case, as an HTML parser
    reads it.

    Args:
        text: The speech's text as published.

    Returns:
        ``html`` if the text holds a ``<p>`` element, ``lines`` if it holds a
        ``\\r\\n``, and ``wrapped`` otherwise.
    """
    lowered = text.lower()
    if "<p>" in lowered or "<p " in lowered:
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
        elif tag not in _INLINE_TAGS:
            raise ValueError(f"Unknown element <{tag}>: it could join or split words.")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self._close()
        elif tag not in _INLINE_TAGS:
            raise ValueError(f"Unknown element </{tag}>: it could join or split words.")

    def handle_data(self, data: str) -> None:
        if self._open is None:
            self.stray.append(data)
        else:
            self._open.append(data)

    def close(self) -> None:
        """Read what is left, and end a last paragraph that the source leaves open."""
        super().close()
        self._close()

    def _close(self) -> None:
        if self._open is not None:
            self.paragraphs.append("".join(self._open))
            self._open = None


def html_paragraphs(text: str) -> list[str]:
    """Read the paragraphs of a speech in the ``html`` layout.

    A ``<p>`` that is never closed ends where the next one opens or where the
    text ends, as in HTML itself.

    Args:
        text: The speech's text as published.

    Returns:
        The normalised text of each non-empty ``<p>`` element, in order.

    Raises:
        ValueError: If any text sits outside the ``<p>`` elements, or an
            element other than ``p``, ``em`` and ``strong`` occurs. The source
            has neither, and reading past them in silence could lose, join or
            split the speaker's words.
    """
    collector = _ParagraphCollector()
    collector.feed(text)
    collector.close()
    stray = normalise("".join(collector.stray))
    if stray:
        raise ValueError(f"Text outside the paragraph elements: {stray[:80]!r}")
    return [paragraph for raw in collector.paragraphs if (paragraph := normalise(raw))]


def _is_whole_insertion(text: str) -> bool:
    """Tell whether *text* is one parenthesis from its first character to its last."""
    if not text.startswith("("):
        return False
    depth = 0
    for position, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            return position == len(text) - 1
    return False


def _is_bullet(group: Sequence[str], line: str) -> bool:
    """Tell whether *line* is a list item whose bullet the source replaced with ``?``.

    The ``?`` is a bullet when a letter follows it (``"?nytt filmavtal"``), or
    when it follows a full stop, a colon or another such item, after which a
    question mark cannot stand: 3 records in the source, with 14 items.
    """
    if line[0] != _LOST_BULLET:
        return False
    previous = group[-1]
    return (
        line[1:2].isalpha()
        or previous.rstrip()[-1] in ".:"
        or (previous[0] == _LOST_BULLET and previous[1:2].isalpha())
    )


def _opens_paragraph(group: Sequence[str], line: str) -> bool:
    """Tell whether *line*, which opens with a space, is a new paragraph all the same.

    The source sometimes begins a paragraph with a stray space. It is a new
    paragraph when the group before it is one line that ends a sentence, both
    that line and *line* are long, and *line* opens with a capital, so that
    neither can be an emphasised run inside a sentence. A salutation after a
    line that ends a sentence opens a paragraph whatever the lengths: ``" Herr
    talman! Jag yrkar bifall till finansutskottet förslag. "``, once in the
    source.
    """
    if len(group) != 1:
        return False
    previous = group[0].strip()
    stripped = line.strip()
    if previous[-1] not in ".!?":
        return False
    return stripped.startswith(_SALUTATIONS) or (
        len(previous) >= _PARAGRAPH_MIN_CHARS
        and len(stripped) >= _PARAGRAPH_MIN_CHARS
        and stripped[0].isupper()
    )


def _is_emphasised_head(previous: str, line: str, following: str | None) -> bool:
    """Tell whether *line*, after a finished sentence, is an emphasised run inside a sentence.

    The source writes ``"utmaningar. "``, ``"En"``, ``" mening handlar ..."``:
    the line is short, and the line after it continues it. After a colon the
    run can be a title or a quotation of any length that the sentence
    introduces: ``"ta ställning till en grönbok: "``, ``"System för
    exportkontroll ... i en föränderlig värld"``, ``"."``.
    """
    if following is None or not (following[0].isspace() or following[0] in _CLOSING):
        return False
    stripped = line.strip()
    if stripped.startswith(_SALUTATIONS):
        return False
    return previous.rstrip().endswith(":") or len(stripped) < _EMPHASIS_MAX_CHARS


def _continues(group: Sequence[str], line: str, following: str | None) -> bool:
    """Tell whether *line* carries on the paragraph that *group* has begun.

    The source puts an emphasised run on its own line, and the pieces keep the
    spaces that joined them: ``"ha bara "``, ``"en"``, ``" kund eller "``. A
    line that opens with a space or with closing punctuation therefore
    continues, unless it is a new paragraph that the source began with a stray
    space (see :func:`_opens_paragraph`). Three other lines continue as well:
    one that follows a line left hanging on a space in mid-sentence; the tail of
    a compound whose head was emphasised (``"public service"``,
    ``"-kanalerna"``); and an emphasised run that opens a sentence after a
    finished one (see :func:`_is_emphasised_head`). A line that ends a sentence
    and is followed by a line with no space in front is a paragraph break, which
    is the ordinary case. A paragraph that is wholly one parenthesis, such as
    another person's interjection, ends with its closing parenthesis, so that
    the speaker's words after it are a paragraph of their own: ``"(TALMANNEN:
    Absolut!)"``, ``". att helt enkelt låsa fast ..."``. A list item whose
    bullet the source replaced with ``?`` stands alone as well.

    Args:
        group: The lines of the paragraph so far.
        line: The line to place.
        following: The next non-blank line, if there is one.

    Returns:
        True if *line* belongs to the paragraph, False if it opens a new one.
    """
    previous = group[-1]
    if _is_whole_insertion("".join(group).strip()) or _is_bullet(group, line):
        return False
    if line[0].isspace():
        return not _opens_paragraph(group, line)
    if line[0] in _CLOSING:
        return True
    if line[0] == "-" and line[1:2].isalpha() and previous[-1].isalnum():
        return True
    trimmed = previous.rstrip()
    if previous == trimmed:
        return False
    return trimmed[-1] not in _TERMINAL or _is_emphasised_head(previous, line, following)


def _is_lost_boundary(group: Sequence[str], line: str, following: str | None) -> bool:
    """Tell whether a break before *line* sits inside a sentence whose spaces the source lost.

    The source sometimes drops the spaces around an emphasised run, so that
    neither line carries one: ``"Man talar om w"``, ``"ider Europe"``, ``" och
    g"``. Whether a space belongs at such a join cannot be told from the text,
    so the break is kept and counted.
    """
    previous = group[-1]
    return (
        following is not None
        and (following[0].isspace() or following[0] in _CLOSING)
        and previous[-1].isalpha()
        and line[0].isalpha()
    )


def _line_groups(text: str) -> tuple[list[list[str]], int]:
    """Group the lines of a ``lines`` text into paragraphs, and count the lost boundaries.

    A line with nothing left after :func:`normalise` is blank and skipped.
    """
    lines = [html.unescape(line) for line in text.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if normalise(line)]
    groups: list[list[str]] = []
    lost = 0
    for index, line in enumerate(lines):
        following = lines[index + 1] if index + 1 < len(lines) else None
        if groups and _continues(groups[-1], line, following):
            groups[-1].append(line)
            continue
        if groups and _is_lost_boundary(groups[-1], line, following):
            lost += 1
        groups.append([line])
    return groups, lost


def line_paragraphs(text: str) -> list[str]:
    """Read the paragraphs of a speech in the ``lines`` layout.

    Lines end with ``\\r\\n``; 6 texts also carry a bare ``\\n``, which ends a
    line as well. Entities are decoded, as in the ``html`` layout: the source
    writes ``&amp;`` 954 times in 598 texts, for ``Vin &amp; Sprit`` and the
    like, and no other entity.

    Args:
        text: The speech's text as published.

    Returns:
        The normalised paragraphs, in order. Lines split by emphasis are joined
        exactly as they stand, with no character added between them.
    """
    groups, _ = _line_groups(text)
    return [normalise("".join(group)) for group in groups]


def lost_emphasis_boundaries(text: str) -> int:
    """Count the breaks inside a sentence that a ``lines`` text keeps for lack of a space.

    See :func:`_is_lost_boundary`.

    Args:
        text: The speech's text as published.

    Returns:
        The number of such breaks, 0 for any layout other than ``lines``.
    """
    if detect_layout(text) != "lines":
        return 0
    _, lost = _line_groups(text)
    return lost


def read_paragraphs(text: str) -> list[str] | None:
    """Read a speech's text into normalised paragraphs, the record's included.

    Args:
        text: The speech's text as published.

    Returns:
        The paragraphs in order, or ``None`` for the ``wrapped`` layout, which
        the corpus does not read.

    Raises:
        ValueError: If the ``html`` layout carries text outside its paragraphs
            or an element it does not use.
    """
    layout = detect_layout(text)
    if layout == "wrapped":
        return None
    return html_paragraphs(text) if layout == "html" else line_paragraphs(text)


def margin_heading(paragraph: str) -> str | None:
    """Return the margin heading that a field-code residue carries.

    Args:
        paragraph: A normalised paragraph of a speech record.

    Returns:
        The heading after ``MERGEFORMAT``, or ``None`` if the paragraph does not
        open with a field code or the code carries no heading.
    """
    if not paragraph.startswith(FIELD_CODE_PREFIX):
        return None
    _, found, heading = paragraph.partition(FIELD_CODE_SEPARATOR)
    return heading.strip() if found and heading.strip() else None


def _is_field_code(paragraph: str) -> bool:
    """Tell whether *paragraph* is the residue of a field code.

    The residue opens the paragraph, or follows the first half of a margin
    heading that the page hyphenated: ``Gransknings- STYLEREF Kantrubrik \\*
    MERGEFORMAT Fel! Ogiltig självreferens i bokmärke.``, once in the source.

    Raises:
        ValueError: If the residue follows anything else, which could be the
            speaker's words and needs a person to look at it.
    """
    if paragraph.startswith(FIELD_CODE_PREFIX):
        return True
    head, found, rest = paragraph.partition(FIELD_CODE_PREFIX)
    if not found or FIELD_CODE_SEPARATOR not in rest:
        return False
    if len(head.split()) == 1 and head.rstrip().endswith("-"):
        return True
    raise ValueError(f"Field-code residue inside a paragraph: {paragraph[:80]!r}")


def _squash(text: str) -> str:
    """Reduce *text* to its letters and digits, in lower case."""
    return "".join(char for char in text.lower() if char.isalnum())


def _respace(text: str) -> str:
    """Put *text* in lower case without hyphens, en dashes or spaces.

    A title written with an en dash comes through the page with a hyphen:
    ``"Strategisk export-kontroll 2015 - krigsmateriel ..."``.
    """
    lowered = text.lower()
    for dash in _DASHES:
        lowered = lowered.replace(dash, "")
    return "".join(lowered.split())


def _title_with_subtitle(title: str, subtitle: str) -> str | None:
    """Run a debate title and its subtitle together and reduce them to letters and digits.

    Returns:
        The joined text, or ``None`` unless both are given and the joined text
        has enough letters and digits to be matched.
    """
    joined = _squash(title + subtitle)
    return joined if title and subtitle and len(joined) >= _HEADING_MIN_SQUASHED else None


def _drops_one(text: str, whole: str) -> bool:
    """Tell whether *text* is *whole* with exactly one character left out."""
    return len(whole) == len(text) + 1 and any(
        whole[:index] + whole[index + 1 :] == text for index in range(len(whole))
    )


def _heading_pieces(
    paragraphs: Sequence[str], headings: Headings, titles: Sequence[str]
) -> dict[int, str]:
    """Find the paragraphs that are pieces of a heading the page broke, by index.

    In the ``html`` layout the margin heading of a page can come through as
    ordinary paragraphs that no heading equals, because the printed line broke
    it. Five rules find them, in this order:

    * ``heading_pieces``: two to eight consecutive paragraphs that together
      spell a heading of the record, or a debate title and its subtitle run
      together: ``"Bastjänstgöring"``, ``"för läkare"``; 95 paragraphs in 34
      speeches;
    * ``heading_line_break``: one paragraph that differs from a heading only by
      a hyphen or a space where the line broke: ``"Ett starkt straffrättsligt
      skydd för den person-liga integriteten"``; 279 paragraphs;
    * ``heading_with_subtitle``: one paragraph that spells a debate title and
      its subtitle run together, those of the speech itself or of another
      debate in the record: ``"Gransknings-betänkandeRegeringens ansvarför
      förvaltningen (kapitel 4)"``; 34 paragraphs;
    * ``title_fragment``: one paragraph, without the punctuation that ends a
      sentence, that is a piece of the speech's own title and subtitle:
      ``"Aktuell debatt om"``; 31 paragraphs;
    * ``title_misprint``: one paragraph, without that punctuation, that is the
      speech's own title or subtitle with one letter missing: ``"Val av
      chefsjustite-ombudsman"``, once.

    A paragraph of the speaker's can equal a heading in all but case: in the
    ``html`` layout 9 paragraphs of the speeches the corpus keeps do, with 8
    distinct texts, ``"ekonomisk tillväxt"`` among them. That is why a single
    paragraph is matched against a heading by its letters only when the line
    break shows in it. For the same reason a title without a subtitle is never
    matched by its letters alone: ``"Stöd till Ukraina."`` is a speaker's.
    """
    found: dict[int, str] = {}
    own = _squash("".join(titles))
    own_pair = _title_with_subtitle(*titles) if len(titles) == 2 else None
    pairs = headings.pairs | {own_pair} if own_pair else headings.pairs
    targets = headings.squashed | pairs
    if len(own) >= _HEADING_MIN_SQUASHED:
        targets |= {own}
    misprinted = [whole for whole in map(_squash, titles) if len(whole) >= _HEADING_MIN_SQUASHED]
    squashed = [
        _squash(paragraph) if len(paragraph) <= _HEADING_PIECE_MAX_CHARS else None
        for paragraph in paragraphs
    ]
    for start in range(len(paragraphs)):
        if start in found or squashed[start] is None:
            continue
        joined = ""
        longest = None
        for end in range(start, min(start + _HEADING_MAX_PIECES, len(paragraphs))):
            piece = squashed[end]
            if piece is None:
                break
            joined += piece
            if end > start and joined in targets:
                longest = end
        if longest is not None:
            found.update(dict.fromkeys(range(start, longest + 1), "heading_pieces"))
    for index, paragraph in enumerate(paragraphs):
        piece = squashed[index]
        if index in found or piece is None:
            continue
        if _respace(paragraph) in headings.respaced and paragraph.lower() not in headings.lowered:
            found[index] = "heading_line_break"
        elif piece in pairs:
            found[index] = "heading_with_subtitle"
        elif paragraph[-1] in _TERMINAL:
            continue
        elif len(piece) >= _TITLE_FRAGMENT_MIN_SQUASHED and piece in own:
            found[index] = "title_fragment"
        elif any(_drops_one(piece, whole) for whole in misprinted):
            found[index] = "title_misprint"
    return found


def _furniture_rule(paragraph: str, layout: SourceLayout, headings: Headings) -> str | None:
    """Name the rule that makes *paragraph* page furniture, if one does.

    A heading is matched in the ``html`` layout only. In the ``lines`` layout
    the one paragraph that equals a heading is the speaker's, an item of a
    list: ``"Biologisk mångfald"``.
    """
    if _is_field_code(paragraph):
        return "field_code"
    return "heading" if layout == "html" and paragraph in headings.exact else None


def _closes_with_note(paragraph: str) -> bool:
    """Tell whether *paragraph* is the note that the debate was over."""
    if len(paragraph) <= _CLOSING_NOTE_MAX_CHARS:
        lowered = paragraph.lower().rstrip(".")
        if _CLOSING_NOTE_WORDS[0] in lowered and lowered.endswith(_CLOSING_NOTE_WORDS[1]):
            return True
    words = paragraph.split()
    return (
        len(words) == 4
        and words[0] in _CLOSING_NOTE_SUBJECTS
        and words[1] == "var"
        and words[2] in _CLOSING_NOTE_ADVERBS
        and words[3] == "avslutad."
    )


def _is_leave(paragraph: str) -> bool:
    """Tell whether *paragraph* records that an absent interpellant was excused.

    The chair is named in every such note, which keeps a speaker's "Då
    förklarade jag ..." out of it.
    """
    if not paragraph.startswith(_LEAVE_OPENING):
        return False
    if not any(chair in paragraph for chair in _LEAVE_CHAIRS):
        return False
    closed = _LEAVE_CLOSED[0] in paragraph and paragraph.endswith(_LEAVE_CLOSED[1])
    return _LEAVE_GRANTED in paragraph or closed


def _closing_rule(paragraph: str) -> str | None:
    """Name the rule by which the speaker's words end where *paragraph* begins."""
    if _closes_with_note(paragraph):
        return "closing_note"
    if paragraph.startswith(_CONCURRENCE_PREFIX):
        return "concurrence"
    if paragraph.startswith(_CHAIR_NARRATION):
        return "chair_narration"
    if paragraph.startswith(_CHAMBER_NARRATION):
        return "chamber_narration"
    if _is_leave(paragraph):
        return "leave_of_absence"
    if paragraph.startswith(_ELECTION_PREFIXES) or (
        paragraph.startswith(_ELECTION_PASSIVE[0]) and paragraph.endswith(_ELECTION_PASSIVE[1])
    ):
        return "election"
    if _DOCUMENT_READ.fullmatch(paragraph):
        return "document_read"
    if all(word in paragraph for word in _DEMAND_WORDS):
        return "demand_narration"
    if paragraph.startswith(_COMMITTEE_NARRATION):
        return "committee_narration"
    if paragraph == _VOTE:
        return "vote"
    return None


def _note_rule(paragraph: str) -> str | None:
    """Name the rule that makes *paragraph* an insertion by the record, if one does."""
    if _is_whole_insertion(paragraph):
        return "insertion"
    if _UNBRACKETED_INTERJECTION.match(paragraph):
        return "unbracketed_interjection"
    return None


def _interjection_end(paragraphs: Sequence[str], start: int) -> tuple[int, int] | None:
    """Find where an interjection that opens *paragraphs[start]* ends.

    The interjection ends at the parenthesis that closes the one it opens
    with, in the same paragraph or in one of the next few. It never runs into a
    paragraph that opens a parenthesis of its own.

    Returns:
        The index of the paragraph that closes it and the length of the part of
        that paragraph that belongs to it, or ``None`` when it is not closed
        within reach.
    """
    depth = 0
    for index in range(start, min(start + 1 + _INTERJECTION_REACH, len(paragraphs))):
        paragraph = paragraphs[index]
        if index > start and paragraph.startswith("("):
            return None
        for position, char in enumerate(paragraph):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                return index, position + 1
    return None


def _structure(
    paragraphs: Sequence[str], layout: SourceLayout, headings: Headings, titles: Sequence[str]
) -> list[tuple[str | None, str]]:
    """Mark what can only be told from more than one paragraph, and cut where needed.

    Returns the speech as a list of parts in order, each with the rule that
    sets it aside or ``None`` where the other rules still have to decide.
    """
    pending = list(paragraphs)
    pieces = _heading_pieces(pending, headings, titles) if layout == "html" else {}
    parts: list[tuple[str | None, str]] = []
    index = 0
    while index < len(pending):
        paragraph = pending[index]
        if index in pieces:
            parts.append((pieces[index], paragraph))
            index += 1
            continue
        if _INTERJECTION_OPENING.match(paragraph) and not _is_whole_insertion(paragraph):
            end = _interjection_end(pending, index)
            if end is None:
                parts.append(("interjection_unclosed", paragraph))
                index += 1
                continue
            last, cut = end
            rule = "interjection_span" if last > index else "interjection_cut"
            parts.extend((rule, pending[inner]) for inner in range(index, last))
            parts.append((rule, pending[last][:cut]))
            rest = pending[last][cut:].strip()
            if rest:
                pending[last] = rest
                index = last
            else:
                index = last + 1
            continue
        if (
            paragraph.endswith(_FOOTER_SIGNATURE)
            and index + 2 < len(pending)
            and pending[index + 1].isupper()
            and pending[index + 2].startswith(_FOOTER_CLERK)
        ):
            clerk, rest = _clerk_and_rest(pending[index + 2])
            parts.append((None, paragraph.removesuffix(_FOOTER_SIGNATURE)))
            footer = (_FOOTER_SIGNATURE.strip(), pending[index + 1], clerk)
            parts.extend(("page_footer", piece) for piece in footer)
            if rest:
                pending[index + 2] = rest
                index += 2
            else:
                index += 3
            continue
        parts.append((None, paragraph))
        index += 1
    return parts


def _clerk_and_rest(paragraph: str) -> tuple[str, str]:
    """Split a paragraph that opens with the clerk's signature into the signature and the rest.

    The signature is ``/`` and the clerk's first name and surname; the
    speaker's sentence goes on after it, in lower case: ``/Eva-Lena Ekman
    mänskliga rättigheter. ...``.

    Raises:
        ValueError: If the signature is not two capitalised words, or the text
            after it does not open in lower case. A third capitalised word
            could be part of the name or the speaker's next word, and only a
            person can tell which.
    """
    words = paragraph.removeprefix(_FOOTER_CLERK).split(" ")
    name, rest = words[:_CLERK_WORDS], words[_CLERK_WORDS:]
    if len(name) < _CLERK_WORDS or not all(word[:1].isupper() for word in name):
        raise ValueError(f"No clerk's signature at the page footer: {paragraph[:80]!r}")
    if rest and not rest[0][:1].islower():
        raise ValueError(f"The clerk's signature runs into a capital: {paragraph[:80]!r}")
    return _FOOTER_CLERK + " ".join(name), " ".join(rest)


def split_speech_from_record(
    paragraphs: Sequence[str],
    layout: SourceLayout,
    headings: Headings = NO_HEADINGS,
    titles: Sequence[str] = (),
) -> SpeechText:
    """Set the record's own paragraphs aside from the speaker's.

    Four kinds of text are not the speaker's words:

    * page furniture, which is dropped: the residue of a field code, a heading
      of the record, whole or in the pieces the page broke it into, and the
      protocol's page footer;
    * an insertion in parentheses, which becomes a note: applause, a decision,
      an interjection by somebody else, also one that runs over several
      paragraphs or into the speaker's next words;
    * an interjection the record gives without parentheses;
    * the paragraph that closes the speech, and everything after it: that the
      debate was over, who concurred, what the chair or the chamber did, an
      election, a document read into the protocol, a vote.

    After the closing paragraph only a field code or a whole heading is told
    apart from the rest, as furniture. The rules for the pieces of a heading
    look for furniture among the speaker's words, and in the record's own text
    that follows they would take an emphasised word of its narration for a
    piece of the title: ``"... valde därmed ... till"``, ``"justitieombudsman"``.

    Args:
        paragraphs: The normalised paragraphs of one speech record, in order.
        layout: The layout the paragraphs were read from. Headings are matched
            in the ``html`` layout only.
        headings: The record's headings, debate titles and margin headings
            alike, see :func:`index_headings`.
        titles: The speech's own debate title and subtitle, in that order.

    Returns:
        The speaker's paragraphs, and the others with the rule that set each
        of them aside.

    Raises:
        ValueError: If a field-code residue sits inside what could be speech,
            or the clerk's signature in a page footer runs into words that
            could be the speaker's.
    """
    speech: list[str] = []
    set_aside: list[SetAside] = []
    closed = False
    for marked, paragraph in _structure(paragraphs, layout, headings, titles):
        rule = _furniture_rule(paragraph, layout, headings)
        if rule is None and closed:
            rule = "after_closing"
        if rule is None:
            rule = marked
        if rule is None:
            rule = _closing_rule(paragraph)
            closed = rule is not None
        if rule is None:
            rule = _note_rule(paragraph)
        if rule is None:
            speech.append(paragraph)
        else:
            set_aside.append(SetAside(rule, paragraph))
    return SpeechText(tuple(speech), tuple(set_aside))
