"""Building the speech corpus from the raw datasets.

Every record in the source ends up in exactly one place: as a speech in the
corpus, or in a count under the reason it was left out. Nothing is dropped
without being counted.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict

from search.corpus.fetch import Manifest, verify_raw_files
from search.corpus.parse import (
    FIELD_CODE_PREFIX,
    Headings,
    SetAside,
    SourceLayout,
    detect_layout,
    index_headings,
    lost_emphasis_boundaries,
    margin_heading,
    normalise,
    read_paragraphs,
    split_speech_from_record,
)
from search.corpus.store import write_speeches
from search.models import Anforande, deep_link, session_start_year, text_sha256
from search.parties import canonical_party

RawRecord = dict[str, Any]

BUILD_REPORT_NAME = "anforanden_build.json"

OUTSIDE_THE_EIGHT = "outside_the_eight_parties"
NO_TEXT = "no_text"
HARD_WRAPPED = "hard_wrapped_source"
ONLY_RECORD_TEXT = "only_record_text"
TEST_RECORD = "test_record"

#: A record whose speaker field names another party in parentheses than its
#: party field, so that the record contradicts itself on whose speech it is.
ANOTHER_PARTY_IN_SPEAKER = "speaker_names_another_party"

#: A speech kept whose text has a break inside a sentence where the source lost
#: the spaces around an emphasised run; see ``parse.lost_emphasis_boundaries``.
LOST_EMPHASIS_BOUNDARY = "lost_emphasis_boundary"

#: A speech kept whose speaker field names none of the eight parties in
#: parentheses, so that only the party field gives its party.
NO_PARTY_IN_SPEAKER = "speaker_without_party_label"

#: Protocols that are tests of the publishing system. H909999 is protocol
#: 2021/22:999, dated 2022-03-29, while the session's other protocols are
#: numbered up to 142. Each of its 6 records carries the name and id of a
#: member who speaks elsewhere in the session. Records 2 to 5 hold keyboard
#: mash only, 13 to 24 words each, such as "Sdf lksjdklf". Record 1 holds a
#: variant of the speech H90989-42, which a member of another party held the
#: same day: 1 020 of the record's 1 232 words occur in that speech in the same
#: order, and its last three paragraphs are one word of keyboard mash each,
#: such as "Askjdhajksdhjka". Record 6 holds the 12 paragraphs of H90988-1 and
#: then the 7 of H90988-2, field-code residue apart, word for word, under the
#: name of a member who held neither of them.
TEST_PROTOCOLS = frozenset({"H909999"})

#: A heading field that opens like a speech is a speaker's sentence: the 23
#: records GU09126-230 to -252 carry the avsnittsrubrik "Fru talman! Jag vill
#: börja med att yrka bifall till reservationerna 1, 2, 4 och 5."
_ADDRESSES = ("Herr talman", "Fru talman")

_REPLY = {"Y": True, "N": False}
_NO_ACTIVITY = frozenset({"", "-"})
_TOP_HEADINGS = 20


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SessionTally(_Frozen):
    """What the build made of one session's dataset.

    Attributes:
        session: The parliamentary session, for example ``2014/15``.
        source_records: How many speech records the dataset holds.
        speeches: How many of them became speeches in the corpus.
        excluded: How many were left out, by reason.
        layouts: How many records use each text layout, among those with text
            from one of the eight parties.
        first_date: The date of the earliest speech kept.
        last_date: The date of the latest speech kept.
        paragraphs: How many of the speaker's paragraphs the speeches kept hold.
        words: How many words those paragraphs hold.
    """

    session: str
    source_records: int
    speeches: int
    excluded: dict[str, int]
    layouts: dict[str, int]
    first_date: date | None
    last_date: date | None
    paragraphs: int
    words: int


class BuildReport(_Frozen):
    """What the build made of the whole source.

    The speaker field of a record names the member's party in parentheses, and
    the build holds it against the record's party field, both read through
    :func:`search.parties.canonical_party`. A record whose speaker field names
    another party is left out and counted in ``excluded`` under
    ``speaker_names_another_party``. A speech whose speaker field names none of
    the eight parties is kept and listed in ``flagged`` under
    ``speaker_without_party_label``. Every other speech kept has a speaker
    field that names its party and no other.

    Attributes:
        inputs_sha256: A hash over the name and SHA-256 of every dataset file.
        source_records: How many speech records the datasets hold.
        speeches: How many of them became speeches in the corpus.
        excluded: How many were left out, by reason.
        excluded_party_labels: The party labels of the records outside the
            eight parties, as published, with their counts.
        hyphenated_line_ends_in_wrapped: How many lines end in a hyphen in the
            records left out for being hard-wrapped.
        set_aside: How many paragraphs are not the speaker's, by the rule that
            set them aside.
        flagged: The speeches kept with a known defect of the source that the
            build keeps as published, by kind, with their keys.
        speeches_with_notes: How many speeches kept carry at least one note.
        headings: How many distinct headings of the record were collected.
        dropped_headings: The paragraphs most often dropped for being a
            heading, with their counts.
        paragraphs: How many of the speaker's paragraphs the corpus holds.
        words: How many words those paragraphs hold.
        words_per_speech: The mean, median, ninth decile and maximum.
        speeches_by_party: The speeches kept, by party.
        sessions: The same account for each session.
    """

    inputs_sha256: str
    source_records: int
    speeches: int
    excluded: dict[str, int]
    excluded_party_labels: dict[str, int]
    hyphenated_line_ends_in_wrapped: int
    set_aside: dict[str, int]
    flagged: dict[str, tuple[str, ...]]
    speeches_with_notes: int
    headings: int
    dropped_headings: dict[str, int]
    paragraphs: int
    words: int
    words_per_speech: dict[str, float]
    speeches_by_party: dict[str, int]
    sessions: tuple[SessionTally, ...]


class Outcome(NamedTuple):
    """What became of one source record.

    Attributes:
        speech: The speech, if the record became one.
        excluded_as: The reason the record was left out, if it was.
        layout: The text layout, when the record had text to read.
        set_aside: The paragraphs that are not the speaker's.
        flags: The known defects of the source the speech keeps as published.
    """

    speech: Anforande | None
    excluded_as: str | None
    layout: str | None
    set_aside: tuple[SetAside, ...]
    flags: tuple[str, ...] = ()


def iter_records(dataset: Path) -> Iterator[RawRecord]:
    """Yield the speech records in one dataset file, in the order of the archive.

    Args:
        dataset: The zip of one session's dataset.

    Yields:
        Each record's ``anforande`` object as published.
    """
    with zipfile.ZipFile(dataset) as archive:
        for name in archive.namelist():
            yield json.loads(archive.read(name).decode("utf-8-sig"))["anforande"]


def heading_field(record: RawRecord, field: str) -> str | None:
    """Read a heading field of a record.

    Args:
        record: The record as published.
        field: The name of the field, ``avsnittsrubrik`` or ``underrubrik``.

    Returns:
        The normalised heading, or ``None`` if the field is empty or opens like
        a speech.
    """
    heading = normalise(record.get(field) or "")
    return None if not heading or heading.startswith(_ADDRESSES) else heading


def record_key(record: RawRecord) -> str:
    """Return the key of a record: its protocol's id and its number in the protocol.

    Args:
        record: The record as published.

    Returns:
        The key, for example ``H209100-7``.
    """
    return f"{record['dok_id']}-{record['anforande_nummer']}"


@contextmanager
def _naming(record: RawRecord) -> Iterator[None]:
    """Name the record in a ``ValueError`` raised while its text is read.

    The speech model's own messages carry the key already; those errors pass
    as they are, so the key stands once in the message.

    Args:
        record: The record whose text is read inside the block.

    Raises:
        ValueError: The error raised inside the block. Its message is given
            ``<key>: `` in front when it does not carry the key.
    """
    try:
        yield
    except ValueError as error:
        key = record_key(record)
        if key in str(error):
            raise
        raise ValueError(f"{key}: {error}") from error


def headings_in(records: Iterable[RawRecord]) -> set[str]:
    """Gather the record's headings: debate titles, and the margin headings of its pages.

    A paragraph that is identical to one of them is page furniture. A margin
    heading is read from the residue of a field code, which names it after
    ``MERGEFORMAT``: ``STYLEREF Kantrubrik \\* MERGEFORMAT Svar på
    interpellationer``. In the source, every record that holds such residue has
    the ``html`` layout, the one layout headings are matched in.

    Args:
        records: The source records to read.

    Returns:
        The distinct headings.

    Raises:
        ValueError: If the text of a record that holds field-code residue
            cannot be read into paragraphs. The message opens with the
            record's key.
    """
    headings: set[str] = set()
    for record in records:
        for field in ("avsnittsrubrik", "underrubrik"):
            if heading := heading_field(record, field):
                headings.add(heading)
        text = record.get("anforandetext") or ""
        if FIELD_CODE_PREFIX in text:
            with _naming(record):
                paragraphs = read_paragraphs(text) or ()
            for paragraph in paragraphs:
                if heading := margin_heading(paragraph):
                    headings.add(heading)
    return headings


def heading_pairs_in(records: Iterable[RawRecord]) -> set[tuple[str, str]]:
    """Gather the debate titles that come with a subtitle, each with its subtitle.

    The margin heading of a page can print a title and its subtitle as one
    line, which no single heading equals: ``"Gransknings-betänkandeRegeringens
    ansvarför förvaltningen (kapitel 4)"``. It can name another debate than the
    speech's own, in 3 speeches.

    Args:
        records: The source records to read.

    Returns:
        The distinct pairs of a title and its subtitle.
    """
    pairs: set[tuple[str, str]] = set()
    for record in records:
        title = heading_field(record, "avsnittsrubrik")
        subtitle = heading_field(record, "underrubrik")
        if title and subtitle:
            pairs.add((title, subtitle))
    return pairs


def parties_in_speaker(speaker: str) -> frozenset[str]:
    """Read the parties a speaker field names in parentheses.

    The field gives the member's party after the name, in either case and
    under the label of its time, with or without a title before the name and a
    reply marker after the label: ``"Andre vice talman KERSTIN HEINEMANN
    (fp)"``, ``"FREDRIK REINFELDT (m) replik"``, ``"ANNA KINBERG BATRA (M)
    (replik)"``.

    Args:
        speaker: The speaker field of a record.

    Returns:
        The canonical codes of the eight parties that stand in parentheses in
        the field. Empty if no pair of parentheses holds one of them.
    """
    named: set[str] = set()
    for opened in speaker.split("(")[1:]:
        label, closed, _ = opened.partition(")")
        party = canonical_party(label) if closed else None
        if party is not None:
            named.add(party)
    return frozenset(named)


def read_record(record: RawRecord, headings: Headings, *, first_session: str) -> Outcome:
    """Turn one source record into a speech, or name the reason it is left out.

    Args:
        record: The record as published.
        headings: The record's headings, see :func:`headings_in`,
            :func:`heading_pairs_in` and
            :func:`search.corpus.parse.index_headings`.
        first_session: The earliest session the corpus keeps, for example
            ``2003/04``. A record of an earlier session is left out as
            hard-wrapped.

    Returns:
        The speech, or the reason it was excluded, with what was set aside.

    Raises:
        ValueError: If the record breaks a rule the corpus depends on, see
            :func:`_read_speech`. The message carries the record's key.
    """
    party = canonical_party(record.get("parti") or "")
    if party is None:
        return Outcome(None, OUTSIDE_THE_EIGHT, None, ())
    text = record.get("anforandetext") or ""
    if not text.strip():
        return Outcome(None, NO_TEXT, None, ())
    layout = detect_layout(text)
    if record["dok_id"] in TEST_PROTOCOLS:
        return Outcome(None, TEST_RECORD, layout, ())
    with _naming(record):
        return _read_speech(record, party, text, layout, headings, first_session)


def _read_speech(
    record: RawRecord,
    party: str,
    text: str,
    layout: SourceLayout,
    headings: Headings,
    first_session: str,
) -> Outcome:
    """Read a record of one of the eight parties that has text and is no test.

    Args:
        record: The record as published.
        party: The canonical code of the record's party field.
        text: The record's text as published.
        layout: The layout of the text.
        headings: The record's headings.
        first_session: The earliest session the corpus keeps.

    Returns:
        The speech, or the reason it was excluded, with what was set aside.

    Raises:
        ValueError: If the record is of a session before *first_session* and
            its text has another layout than ``wrapped``, if it is of a later
            session and its layout is ``wrapped``, if its text cannot be read
            or split, if its reply flag is unknown, or if the speech model
            refuses one of its fields.
    """
    session = record["dok_rm"]
    if session_start_year(session) < session_start_year(first_session):
        if layout != "wrapped":
            raise ValueError(
                f"the text has the {layout} layout in {session}, a session before "
                f"{first_session}. The corpus leaves those sessions out because their text "
                "is hard-wrapped, which does not hold for this record."
            )
        return Outcome(None, HARD_WRAPPED, layout, ())
    if layout == "wrapped":
        raise ValueError(
            f"the text is hard-wrapped in {session}, a session the corpus keeps. "
            "Its paragraphs cannot be told apart."
        )
    speaker = normalise(record["talare"])
    named = parties_in_speaker(speaker)
    if named - {party}:
        return Outcome(None, ANOTHER_PARTY_IN_SPEAKER, layout, ())
    title = heading_field(record, "avsnittsrubrik")
    titles = (title or "", heading_field(record, "underrubrik") or "")
    split = split_speech_from_record(read_paragraphs(text) or (), layout, headings, titles)
    if not split.paragraphs:
        return Outcome(None, ONLY_RECORD_TEXT, layout, split.set_aside)

    dok_id = record["dok_id"]
    number = int(record["anforande_nummer"])
    if record["replik"] not in _REPLY:
        raise ValueError(f"unknown reply flag {record['replik']!r}.")
    activity = normalise(record.get("kammaraktivitet") or "")
    speech = Anforande(
        anf_id=f"{dok_id}-{number}",
        dok_id=dok_id,
        rm=record["dok_rm"],
        dok_nummer=int(record["dok_nummer"]),
        anf_nummer=number,
        date=date.fromisoformat(record["dok_datum"][:10]),
        speaker_raw=speaker,
        person_id=record.get("intressent_id") or None,
        party=party,
        is_reply=_REPLY[record["replik"]],
        title=title,
        activity=None if activity in _NO_ACTIVITY else activity,
        rel_dok_id=record.get("rel_dok_id") or None,
        layout=layout,
        paragraphs=split.paragraphs,
        notes=split.notes,
        text_sha256=text_sha256(split.paragraphs),
        url=deep_link(dok_id, number),
    )
    lost_boundary = bool(lost_emphasis_boundaries(text))
    found = ((LOST_EMPHASIS_BOUNDARY, lost_boundary), (NO_PARTY_IN_SPEAKER, not named))
    flags = tuple(flag for flag, holds in found if holds)
    return Outcome(speech, None, layout, split.set_aside, flags)


def _hyphenated_line_ends(text: str) -> int:
    """Count the lines of a hard-wrapped text that end in a hyphen."""
    return sum(line.rstrip().endswith("-") for line in text.split("\n"))


def _spread(values: list[int]) -> dict[str, float]:
    """Summarise a list of counts: mean, median, ninth decile and maximum."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 1),
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


class _Tally:
    """The running account of a build, turned into a report at the end."""

    def __init__(self) -> None:
        self.excluded: Counter[str] = Counter()
        self.party_labels: Counter[str] = Counter()
        self.set_aside: Counter[str] = Counter()
        self.flagged: dict[str, list[str]] = {}
        self.dropped_headings: Counter[str] = Counter()
        self.by_party: Counter[str] = Counter()
        self.words_per_speech: list[int] = []
        self.speeches_with_notes = 0
        self.hyphenated_line_ends = 0
        self.sessions: list[SessionTally] = []
        self.seen: set[str] = set()

    def read_session(
        self,
        session: str,
        records: Iterable[RawRecord],
        headings: Headings,
        first_session: str,
    ) -> list[Anforande]:
        """Read one session's records into speeches, sorted, and account for the rest.

        Args:
            session: The session of the dataset.
            records: The dataset's records.
            headings: The record's headings.
            first_session: The earliest session the corpus keeps.

        Returns:
            The speeches kept, by date, protocol and number.

        Raises:
            ValueError: If the dataset holds no record, if a record names
                another session than the dataset's, if a key occurs twice, or
                if a record breaks a rule the corpus depends on, see
                :func:`read_record`.
        """
        speeches: list[Anforande] = []
        excluded: Counter[str] = Counter()
        layouts: Counter[str] = Counter()
        source_records = paragraphs = words = 0
        for record in records:
            source_records += 1
            if record.get("dok_rm") != session:
                raise ValueError(
                    f"{record_key(record)}: the record names session {record.get('dok_rm')} "
                    f"in the dataset of session {session}."
                )
            outcome = read_record(record, headings, first_session=first_session)
            if outcome.layout:
                layouts[outcome.layout] += 1
            for item in outcome.set_aside:
                self.set_aside[item.rule] += 1
                if item.rule == "heading":
                    self.dropped_headings[item.text] += 1
            if outcome.speech is None:
                excluded[outcome.excluded_as or ""] += 1
                if outcome.excluded_as == OUTSIDE_THE_EIGHT:
                    self.party_labels[normalise(record.get("parti") or "") or "(none)"] += 1
                if outcome.excluded_as == HARD_WRAPPED:
                    self.hyphenated_line_ends += _hyphenated_line_ends(record["anforandetext"])
                continue
            speech = outcome.speech
            if speech.anf_id in self.seen:
                raise ValueError(f"{speech.anf_id} occurs twice. The key must be unique.")
            self.seen.add(speech.anf_id)
            for flag in outcome.flags:
                self.flagged.setdefault(flag, []).append(speech.anf_id)
            speeches.append(speech)
            speech_words = sum(len(paragraph.split()) for paragraph in speech.paragraphs)
            self.words_per_speech.append(speech_words)
            self.by_party[speech.party] += 1
            self.speeches_with_notes += bool(speech.notes)
            paragraphs += len(speech.paragraphs)
            words += speech_words
        if not source_records:
            raise ValueError(f"The dataset of session {session} holds no record.")
        speeches.sort(key=lambda speech: (speech.date, speech.dok_nummer, speech.anf_nummer))
        self.excluded.update(excluded)
        self.sessions.append(
            SessionTally(
                session=session,
                source_records=source_records,
                speeches=len(speeches),
                excluded=dict(sorted(excluded.items())),
                layouts=dict(sorted(layouts.items())),
                first_date=speeches[0].date if speeches else None,
                last_date=speeches[-1].date if speeches else None,
                paragraphs=paragraphs,
                words=words,
            )
        )
        return speeches

    def report(self, inputs_sha256: str, headings: int) -> BuildReport:
        """Freeze the account into the report."""
        return BuildReport(
            inputs_sha256=inputs_sha256,
            source_records=sum(session.source_records for session in self.sessions),
            speeches=len(self.seen),
            excluded=dict(sorted(self.excluded.items())),
            excluded_party_labels=dict(self.party_labels.most_common()),
            hyphenated_line_ends_in_wrapped=self.hyphenated_line_ends,
            set_aside=dict(sorted(self.set_aside.items())),
            flagged={flag: tuple(sorted(keys)) for flag, keys in sorted(self.flagged.items())},
            speeches_with_notes=self.speeches_with_notes,
            headings=headings,
            dropped_headings=dict(self.dropped_headings.most_common(_TOP_HEADINGS)),
            paragraphs=sum(session.paragraphs for session in self.sessions),
            words=sum(session.words for session in self.sessions),
            words_per_speech=_spread(self.words_per_speech),
            speeches_by_party=dict(sorted(self.by_party.items())),
            sessions=tuple(self.sessions),
        )


def inputs_digest(manifest: Manifest) -> str:
    """Hash the name and SHA-256 of every dataset file, to tie a build to its inputs.

    Args:
        manifest: The datasets the build reads.

    Returns:
        The SHA-256 over one ``file:sha256`` line per dataset, in hex.
    """
    lines = "\n".join(f"{entry.file}:{entry.sha256}" for entry in manifest.files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def check_sessions(manifest: Manifest, sessions: Sequence[str]) -> None:
    """Check that the manifest holds the dataset of every session, and of no other.

    Args:
        manifest: The datasets that were fetched.
        sessions: The sessions the build reads.

    Raises:
        ValueError: If the manifest lacks one of the sessions, or holds a
            session that is not among them.
    """
    fetched = manifest.by_session()
    missing = [session for session in sessions if session not in fetched]
    if missing:
        raise ValueError(
            f"The manifest lacks: {', '.join(missing)}. "
            "Run step 01, 01_fetch_anforanden.py, to fetch the missing datasets."
        )
    unexpected = sorted(set(fetched) - set(sessions))
    if unexpected:
        raise ValueError(
            "The manifest holds a dataset outside the sessions the settings name: "
            f"{', '.join(unexpected)}."
        )


def build_corpus(
    manifest: Manifest,
    raw_dir: Path,
    destination: Path,
    *,
    sessions: Sequence[str],
    first_session: str,
) -> BuildReport:
    """Build the speech corpus from the datasets of the sessions the settings name.

    The datasets are read for the record's headings, and for its titles with
    their subtitles, before they are read for the speeches: telling a heading
    from a paragraph of speech needs them all.

    Args:
        manifest: The datasets that were fetched.
        raw_dir: The directory they were fetched to.
        destination: The Parquet file to write the speeches to.
        sessions: Every session the build reads, from the first one fetched to
            the last, see :func:`search.corpus.fetch.sessions_between`.
        first_session: The earliest session the corpus keeps. The records of
            the sessions before it are read, counted and left out.

    Returns:
        The account of what was kept and what was left out.

    Raises:
        FileNotFoundError: If a dataset file is missing.
        ValueError: If *first_session* is not one of *sessions*, the manifest
            does not hold exactly the sessions given, a file does not match
            the manifest, a session's dataset holds no record, a record names
            another session than its dataset's, a key occurs twice, or a
            record breaks a rule the corpus depends on. Nothing is written in
            that case.
    """
    if first_session not in sessions:
        raise ValueError(
            f"The corpus's first session, {first_session}, is not one of the sessions "
            "the build reads."
        )
    check_sessions(manifest, sessions)
    verify_raw_files(manifest, raw_dir)
    datasets = [(entry.session, raw_dir / entry.file) for entry in manifest.files]
    headings: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for _, dataset in datasets:
        headings |= headings_in(iter_records(dataset))
        pairs |= heading_pairs_in(iter_records(dataset))
    known_headings = index_headings(headings, pairs)

    tally = _Tally()
    write_speeches(
        destination,
        (
            tally.read_session(session, iter_records(dataset), known_headings, first_session)
            for session, dataset in datasets
        ),
    )
    return tally.report(inputs_digest(manifest), len(known_headings.exact))
