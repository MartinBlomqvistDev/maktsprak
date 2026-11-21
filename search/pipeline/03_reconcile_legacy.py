"""Step 03: reconcile the speech corpus against the older corpus from the protocol PDFs.

Pairs the speeches of the older corpus with speeches of the new one, protocol
by protocol and party by party, and scores each pair by its words. The report,
which is tracked, says what share of the older corpus's speeches the new corpus
covers, how alike the paired texts are, and where they differ most.

The report answers to the rule that ``docs/search/PREREGISTRATION.md`` states
under "Corpus": the datasets must cover a set share of the older corpus's
speeches, and the texts must not differ systematically. The script compares
the coverage with that share, ``reconcile.coverage_threshold`` in
``search/config.yaml``. It gives no verdict on whether the texts differ
systematically; the overlap and surplus figures are there for that judgement.

The sessions in scope run from ``corpus.first_session`` to
``source.last_session``. An older protocol of any other session is counted
apart, as out of scope. A session in scope without a speech in the new corpus
stops the script before anything is written.

Coverage tests one direction. A pair counts when most of the new speech is
found in the older text: at least ``reconcile.min_sequence_share`` of its pairs
of adjacent tokens. A new speech that has lost words can therefore pair, and
coverage does not show whether the new speeches are complete. The other
direction is described by ``old_in_new`` and by the surplus figures, which
count what the older text holds beyond the new speech. Where both corpora
carry the same words, the reconciliation cannot tell the record's text from
the speaker's. That is why the paragraphs set aside as the record's own are
checked by reading them when the corpus is built.

Usage::

    python search/pipeline/03_reconcile_legacy.py
"""

from __future__ import annotations

import argparse
import heapq
import logging
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import ReconcileSettings
from search.corpus.align import (
    SLUG_LIKENESS,
    AlignParams,
    Match,
    ProtocolAlignment,
    Speech,
    Unmatched,
    legacy_key,
    legacy_slug,
    new_speaker_slug,
    prefix_offset,
    reconcile_protocol,
    session_of,
    speaker_slug,
    split_legacy_text,
    tokenise,
)
from search.corpus.fetch import sessions_between
from search.corpus.store import SPEECHES_NAME, write_report

logger = logging.getLogger("reconcile_legacy")

REPORT_NAME = "reconciliation.json"

#: Decimals kept in shares and summary figures.
DIGITS = 4

NEW_COLUMNS = ["anf_id", "dok_id", "rm", "date", "speaker_raw", "party", "paragraphs", "notes"]
OLD_COLUMNS = ["id", "protocol_id", "protocol_date", "speaker", "party", "text"]

SCOPE_DEFINITION = (
    "The sessions in scope run from corpus.first_session to source.last_session in "
    "search/config.yaml; sessions_in_scope gives the first and the last. An older speech is "
    "in scope when its protocol belongs to one of them, and the protocol's session is read "
    "from its id. Older protocols of other sessions lie outside the sessions the corpus "
    "covers and are counted in old_corpus.out_of_scope. Coverage is the share of the older "
    "corpus's in-scope speeches that are paired with a speech in the new corpus. The reverse "
    "share counts the new corpus's speeches dated from the first to the last date of the "
    "older corpus's in-scope protocols, and the share of them paired with an older speech."
)

PAIRING_DEFINITION = (
    "Pairs are made only within one protocol and one party, and only between slugs that "
    "agree on the speaker's name. Two slugs that both end in a place must name the same "
    "place. A slug that ends in a place agrees with one that does not only when the two, "
    f"the place left out, are at least {SLUG_LIKENESS} alike, letter by letter. Otherwise the "
    "given name and surname that end the new slug must be found among the older slug's "
    f"tokens, or the two names must be at least {SLUG_LIKENESS} alike. Two members of one "
    "party with the same name agree where neither slug gives a place, or where one gives it "
    "and the other holds the name alone. A pair needs at least min_shared_tokens shared "
    "tokens, and at least min_sequence_share of the new speech's pairs of adjacent tokens "
    "must be found in the older text. The test runs in that direction only. A new speech "
    "that has lost words can therefore pair, and coverage does not show whether the new "
    "speeches are complete. The other direction is described by overlap.old_in_new and by "
    "the figures under surplus, which count what the older text holds beyond the new "
    "speech. Where both corpora carry the same words, the reconciliation cannot tell the "
    "record's text from the speaker's. That is why the paragraphs set aside as the record's "
    "own are checked by reading them when the corpus is built."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Share(_Frozen):
    """A share with the counts it was computed from."""

    numerator: int
    denominator: int
    share: float


class Spread(_Frozen):
    """The distribution of a share over the pairs: mean, median, 5th and 1st percentile, minimum."""

    mean: float
    median: float
    p05: float
    p01: float
    minimum: float


class OutOfScope(_Frozen):
    """Older protocols of one session outside the sessions the corpus covers.

    Attributes:
        session: The session, read from the protocol id.
        protocols: How many protocols.
        rows: How many records of the older corpus.
        speeches: How many speeches those records hold.
        first_date: The earliest protocol date.
        last_date: The latest protocol date.
    """

    session: str
    protocols: int
    rows: int
    speeches: int
    first_date: date
    last_date: date


class OldCorpus(_Frozen):
    """The older corpus, in total and in scope.

    Attributes:
        rows: Records in the older corpus, one per speaker and party in a protocol.
        speeches: The speeches those records hold.
        out_of_scope: The protocols of sessions outside the sessions the corpus
            covers, by session.
        in_scope_rows: Records in protocols of the sessions in scope.
        in_scope_speeches: The speeches those records hold.
        in_scope_protocols: How many protocols are in scope.
        first_date: The earliest date of an in-scope protocol.
        last_date: The latest date of an in-scope protocol.
        slug_disagreements: In-scope records whose id carries another slug than
            their speaker's name gives.
    """

    rows: int
    speeches: int
    out_of_scope: tuple[OutOfScope, ...]
    in_scope_rows: int
    in_scope_speeches: int
    in_scope_protocols: int
    first_date: date
    last_date: date
    slug_disagreements: int


class NewCorpus(_Frozen):
    """The new corpus, in total and within the dates the older corpus covers.

    Attributes:
        speeches: Speeches in the new corpus.
        protocols: Protocols in the new corpus.
        sessions: Its first and last session.
        in_window_speeches: Speeches in protocols dated from the first to the
            last date of the older corpus's in-scope protocols.
        in_window_protocols: How many protocols that is.
        protocol_date_disagreements: Protocols on both sides whose dates differ.
    """

    speeches: int
    protocols: int
    sessions: tuple[str, str]
    in_window_speeches: int
    in_window_protocols: int
    protocol_date_disagreements: int


class OneSided(_Frozen):
    """Protocols present on one side only.

    Attributes:
        protocols: How many protocols.
        speeches: How many speeches they hold on the side they are on.
        by_year: Protocols per calendar year.
        ids: The protocol ids, sorted, at most ``reconcile.listed_ids`` of them.
        ids_not_listed: How many ids were left out of the list.
    """

    protocols: int
    speeches: int
    by_year: dict[str, int]
    ids: tuple[str, ...]
    ids_not_listed: int


class LeftOver(_Frozen):
    """Speeches without a pair, on one side, in protocols both sides have.

    Attributes:
        speeches: How many speeches.
        by_reason: How many by reason, as :class:`search.corpus.align.Unmatched`
            assigns it: ``too_short`` when the speech has fewer tokens than a
            pair needs in common, so that it can never be paired;
            ``party_absent`` when the other side has no speech of that party in
            the protocol; ``no_content_match`` for a speech of neither kind.
        found_in: How many of them have their words in a speech on the other
            side after all, by where: under another speaker's name in the
            same party, filed under another party, or in a speech that is
            already paired. A diagnosis; it makes no pair.
        listed: Keys, with party, reason and where the words were found, at most
            ``reconcile.listed_ids``.
        not_listed: How many were left out of the list.
    """

    speeches: int
    by_reason: dict[str, int]
    found_in: dict[str, int]
    listed: tuple[str, ...]
    not_listed: int


class YearRow(_Frozen):
    """Coverage in one calendar year.

    Attributes:
        year: The calendar year of the protocol date.
        old_speeches: Older speeches in scope.
        old_matched: Of them, paired.
        coverage: ``old_matched / old_speeches``.
        new_speeches: New speeches within the window.
        new_matched: Of them, paired.
        reverse_share: ``new_matched / new_speeches``.
        pairs_missing_words: Pairs in which more than ``missing_tokens`` of
            the new speech's tokens are absent from the older speech.
        missing_token_share: The new speeches' tokens absent from the older
            speech, as a share of all tokens of the new speeches paired.
    """

    year: int
    old_speeches: int
    old_matched: int
    coverage: float
    new_speeches: int
    new_matched: int
    reverse_share: float
    pairs_missing_words: int
    missing_token_share: float


class Overlap(_Frozen):
    """How alike the paired texts are, over all pairs.

    Attributes:
        new_in_old: Share of the new speech's tokens found in the older speech.
        old_in_new: Share of the older speech's tokens found in the new
            speech's paragraphs; the direction that pairing does not test.
        old_in_new_with_notes: The same, against paragraphs and notes.
        sequence_share: Share of the new speech's pairs of adjacent tokens
            found in the older speech; the measure pairs are made on.
        head_agrees: Share of pairs whose first ``edge_tokens`` tokens agree.
        tail_agrees: Share of pairs whose last ``edge_tokens`` tokens agree.
        identical: Share of pairs whose token sequences are the same.
        old_extends_new: Share of pairs where the older speech is the whole new
            speech followed by more text.
        new_fully_found: Share of pairs where every token of the new speech is
            found in the older speech.
        missing_words: Share of pairs in which more than ``missing_tokens`` of
            the new speech's tokens are absent from the older speech.
        missing_tokens: New speeches' tokens not found in the older speech, in
            total, and as a share of all the new speeches' tokens.
    """

    new_in_old: Spread
    old_in_new: Spread
    old_in_new_with_notes: Spread
    sequence_share: Spread
    head_agrees: Share
    tail_agrees: Share
    identical: Share
    old_extends_new: Share
    new_fully_found: Share
    missing_words: Share
    missing_tokens: Share


class Surplus(_Frozen):
    """What the paired older speeches hold beyond the new speeches, in tokens.

    A pair's surplus is the number of the older speech's tokens less the
    number it shares with the new speech, each token counted as often as it
    occurs. The figures add the surplus up over the pairs. A note is a
    paragraph of the new speech that the build set aside as the record's own
    and kept beside the speech.

    Attributes:
        tokens: The surplus against the new speeches' paragraphs, as a share
            of all tokens of the paired older speeches.
        pairs_over_threshold: Pairs whose surplus against the paragraphs is
            more than ``surplus_tokens``, as a share of the pairs.
        tokens_with_notes: The surplus against paragraphs and notes taken
            together, as a share of all tokens of the paired older speeches.
        pairs_over_threshold_with_notes: Pairs whose surplus against
            paragraphs and notes taken together is more than
            ``surplus_tokens``, as a share of the pairs.
        explained_by_notes: The part of ``tokens`` that the notes hold, which
            is ``tokens`` less ``tokens_with_notes``, as a share of ``tokens``.
            A paragraph of the speech that a rule set aside in error is a
            note too, and its tokens are counted here.
        pairs_with_notes: Pairs whose new speech has notes with at least one
            token in them.
    """

    tokens: Share
    pairs_over_threshold: Share
    tokens_with_notes: Share
    pairs_over_threshold_with_notes: Share
    explained_by_notes: Share
    pairs_with_notes: int


class Deviation(_Frozen):
    """One pair whose texts differ, with its sizes and shares."""

    old_key: str
    new_key: str
    method: str
    new_tokens: int
    old_tokens: int
    note_tokens: int
    missing: int
    surplus: int
    surplus_with_notes: int
    new_in_old: float
    old_in_new_with_notes: float
    sequence_share: float


class ReconciliationReport(_Frozen):
    """The reconciliation of the new corpus against the older corpus.

    Attributes:
        scope: Which speeches are in scope, and how coverage and the reverse
            share are defined.
        pairing: What makes a pair, and what coverage therefore does not show.
        sessions_in_scope: The first and the last session in scope.
        old_corpus: The older corpus in total, out of scope and in scope.
        new_corpus: The new corpus in total and within the window.
        coverage: Older in-scope speeches paired; the share the coverage rule
            is applied to.
        coverage_threshold: The share the coverage rule asks for.
        coverage_meets_threshold: Whether the coverage reaches it.
        coverage_in_shared_protocols: Coverage among protocols both sides have.
        reverse: New speeches in the window paired with an older speech.
        reverse_in_shared_protocols: The same, among protocols both sides have.
        methods: Pairs by how they were found.
        old_only_protocols: Older in-scope protocols the new corpus lacks.
        new_only_protocols: New protocols in the window the older corpus lacks.
        unmatched_old: Older speeches without a pair in shared protocols.
        unmatched_new: New speeches without a pair in shared protocols.
        years: Coverage per calendar year.
        overlap: How alike the paired texts are.
        surplus: What the older speeches have that the new ones do not.
        top_missing_tokens: The new speeches' tokens most often not found in
            the older speech, with counts.
        top_surplus_tokens: The older speeches' tokens most often found in
            neither the paragraphs nor the notes, with counts.
        largest_deviations: The pairs with most differing tokens, counted as
            ``missing + surplus_with_notes``.
        largest_missing: The pairs with most of the new speech's tokens absent
            from the older speech.
        params: The alignment's parameters.
    """

    scope: str
    pairing: str
    sessions_in_scope: tuple[str, str]
    old_corpus: OldCorpus
    new_corpus: NewCorpus
    coverage: Share
    coverage_threshold: float
    coverage_meets_threshold: bool
    coverage_in_shared_protocols: Share
    reverse: Share
    reverse_in_shared_protocols: Share
    methods: dict[str, int]
    old_only_protocols: OneSided
    new_only_protocols: OneSided
    unmatched_old: LeftOver
    unmatched_new: LeftOver
    years: tuple[YearRow, ...]
    overlap: Overlap
    surplus: Surplus
    top_missing_tokens: dict[str, int]
    top_surplus_tokens: dict[str, int]
    largest_deviations: tuple[Deviation, ...]
    largest_missing: tuple[Deviation, ...]
    params: AlignParams


def share(numerator: int, denominator: int) -> Share:
    """Build a share, rounded, keeping its counts."""
    value = round(numerator / denominator, DIGITS) if denominator else 0.0
    return Share(numerator=numerator, denominator=denominator, share=value)


def meets_threshold(numerator: int, denominator: int, threshold: float) -> bool:
    """Say whether a share reaches a threshold, compared exactly.

    The threshold is read from its decimal form as a fraction, so a share of
    exactly 99% meets 0.99 whatever the denominator. A denominator of zero
    meets no threshold.
    """
    if not denominator:
        return False
    return Fraction(numerator, denominator) >= Fraction(str(threshold))


def spread(values: list[float]) -> Spread:
    """Summarise shares over the pairs.

    A percentile ``q`` is the value at index ``int(q * (n - 1))`` of the ``n``
    shares sorted in ascending order.
    """
    ordered = sorted(values)
    if not ordered:
        return Spread(mean=0.0, median=0.0, p05=0.0, p01=0.0, minimum=0.0)

    def rank(q: float) -> float:
        return round(ordered[int(q * (len(ordered) - 1))], DIGITS)

    return Spread(
        mean=round(statistics.fmean(ordered), DIGITS),
        median=round(statistics.median(ordered), DIGITS),
        p05=rank(0.05),
        p01=rank(0.01),
        minimum=round(ordered[0], DIGITS),
    )


def align_params(settings: ReconcileSettings) -> AlignParams:
    """Take the alignment's parameters from the settings."""
    return AlignParams(
        edge_tokens=settings.edge_tokens,
        min_sequence_share=settings.min_sequence_share,
        min_shared_tokens=settings.min_shared_tokens,
        missing_tokens=settings.missing_tokens,
        surplus_tokens=settings.surplus_tokens,
    )


def one_sided(
    ids: Iterable[str], year_of: dict[str, int], speeches: dict[str, int], listed: int
) -> OneSided:
    """Summarise protocols present on one side only, listing at most ``listed`` ids."""
    ordered = sorted(ids)
    by_year = Counter(str(year_of[protocol]) for protocol in ordered)
    return OneSided(
        protocols=len(ordered),
        speeches=sum(speeches[protocol] for protocol in ordered),
        by_year=dict(sorted(by_year.items())),
        ids=tuple(ordered[:listed]),
        ids_not_listed=max(len(ordered) - listed, 0),
    )


def index_by(values: Sequence[str]) -> dict[str, list[int]]:
    """Map each value to the rows it occurs in, in row order."""
    rows: dict[str, list[int]] = defaultdict(list)
    for row, value in enumerate(values):
        rows[value].append(row)
    return rows


def deviation(match: Match) -> Deviation:
    """Describe a pair for the lists of deviations."""
    score = match.score
    return Deviation(
        old_key=match.old_key,
        new_key=match.new_key,
        method=match.method,
        new_tokens=score.new_tokens,
        old_tokens=score.old_tokens,
        note_tokens=score.note_tokens,
        missing=score.missing,
        surplus=score.surplus,
        surplus_with_notes=score.surplus_with_notes,
        new_in_old=round(score.new_in_old, DIGITS),
        old_in_new_with_notes=round(score.old_in_new_with_notes, DIGITS),
        sequence_share=round(score.sequence_share, DIGITS),
    )


class _Tally:
    """The running account of the reconciliation, turned into a report at the end."""

    def __init__(self, params: AlignParams, listed_ids: int) -> None:
        self.params = params
        self.listed_ids = listed_ids
        self.methods: Counter[str] = Counter()
        self.shares: dict[str, list[float]] = defaultdict(list)
        self.flags: Counter[str] = Counter()
        self.pairs = 0
        self.tokens: Counter[str] = Counter()
        self.missing_tokens: Counter[str] = Counter()
        self.surplus_tokens: Counter[str] = Counter()
        self.differing: list[Match] = []
        self.by_year: dict[int, Counter[str]] = defaultdict(Counter)
        self.unmatched: dict[str, list[Unmatched]] = {"old": [], "new": []}

    def add(self, result: ProtocolAlignment, speeches: dict[str, Speech], year: int) -> None:
        """Take in one protocol's alignment, with its speeches by key and its year."""
        self.unmatched["old"].extend(result.unmatched_old)
        self.unmatched["new"].extend(result.unmatched_new)
        for match in result.matches:
            self._add_pair(match, speeches)
            score = match.score
            self.by_year[year]["new_tokens"] += score.new_tokens
            self.by_year[year]["missing"] += score.missing
            self.by_year[year]["missing_words"] += score.missing > self.params.missing_tokens

    def _add_pair(self, match: Match, speeches: dict[str, Speech]) -> None:
        score = match.score
        self.pairs += 1
        self.methods[match.method] += 1
        self.shares["new_in_old"].append(score.new_in_old)
        self.shares["old_in_new"].append(score.old_in_new)
        self.shares["old_in_new_with_notes"].append(score.old_in_new_with_notes)
        self.shares["sequence_share"].append(score.sequence_share)
        self.flags["head_agrees"] += score.head_agrees
        self.flags["tail_agrees"] += score.tail_agrees
        self.flags["identical"] += score.identical
        self.flags["old_extends_new"] += score.old_extends_new
        self.flags["new_fully_found"] += score.missing == 0
        self.flags["missing_words"] += score.missing > self.params.missing_tokens
        self.flags["over"] += score.surplus > self.params.surplus_tokens
        self.flags["over_with_notes"] += score.surplus_with_notes > self.params.surplus_tokens
        self.flags["with_notes"] += score.note_tokens > 0
        self.tokens["new"] += score.new_tokens
        self.tokens["old"] += score.old_tokens
        self.tokens["missing"] += score.missing
        self.tokens["surplus"] += score.surplus
        self.tokens["surplus_with_notes"] += score.surplus_with_notes

        new, old = speeches[match.new_key], speeches[match.old_key]
        if score.missing:
            self.missing_tokens.update(Counter(new.tokens) - Counter(old.tokens))
        if score.surplus_with_notes:
            self.surplus_tokens.update(Counter(old.tokens) - Counter(new.tokens + new.note_tokens))
        if score.missing or score.surplus_with_notes:
            self.differing.append(match)

    def left_over(self, side: str) -> LeftOver:
        """Summarise the speeches without a pair on one side."""
        items = sorted(self.unmatched[side])
        listed = tuple(
            f"{item.key} {item.party} {item.reason} {item.found_in or '-'}"
            for item in items[: self.listed_ids]
        )
        return LeftOver(
            speeches=len(items),
            by_reason=dict(sorted(Counter(item.reason for item in items).items())),
            found_in=dict(
                sorted(Counter(item.found_in for item in items if item.found_in).items())
            ),
            listed=listed,
            not_listed=max(len(items) - self.listed_ids, 0),
        )

    def overlap(self) -> Overlap:
        """Summarise how alike the paired texts are."""
        return Overlap(
            new_in_old=spread(self.shares["new_in_old"]),
            old_in_new=spread(self.shares["old_in_new"]),
            old_in_new_with_notes=spread(self.shares["old_in_new_with_notes"]),
            sequence_share=spread(self.shares["sequence_share"]),
            head_agrees=share(self.flags["head_agrees"], self.pairs),
            tail_agrees=share(self.flags["tail_agrees"], self.pairs),
            identical=share(self.flags["identical"], self.pairs),
            old_extends_new=share(self.flags["old_extends_new"], self.pairs),
            new_fully_found=share(self.flags["new_fully_found"], self.pairs),
            missing_words=share(self.flags["missing_words"], self.pairs),
            missing_tokens=share(self.tokens["missing"], self.tokens["new"]),
        )

    def surplus(self) -> Surplus:
        """Summarise what the older speeches have that the new ones do not."""
        return Surplus(
            tokens=share(self.tokens["surplus"], self.tokens["old"]),
            pairs_over_threshold=share(self.flags["over"], self.pairs),
            tokens_with_notes=share(self.tokens["surplus_with_notes"], self.tokens["old"]),
            pairs_over_threshold_with_notes=share(self.flags["over_with_notes"], self.pairs),
            explained_by_notes=share(
                self.tokens["surplus"] - self.tokens["surplus_with_notes"], self.tokens["surplus"]
            ),
            pairs_with_notes=self.flags["with_notes"],
        )

    def largest(self, size: int, by_missing: bool) -> tuple[Deviation, ...]:
        """List the pairs that differ most, largest first and in protocol order on a tie.

        Args:
            size: How many pairs to list.
            by_missing: Whether to rank by the new speech's tokens absent from
                the older speech alone. Otherwise the older speech's surplus
                against paragraphs and notes is added to them.

        Returns:
            At most ``size`` pairs, each with at least one differing token.
        """

        def size_of(match: Match) -> int:
            score = match.score
            return score.missing if by_missing else score.missing + score.surplus_with_notes

        ranked = heapq.nlargest(size, self.differing, key=size_of)
        return tuple(deviation(match) for match in ranked if size_of(match) > 0)


def new_speeches(table: pa.Table) -> dict[str, list[Speech]]:
    """Read one protocol's new speeches into the alignment's form, by party."""
    by_party: dict[str, list[Speech]] = defaultdict(list)
    for row in table.to_pylist():
        by_party[row["party"]].append(
            Speech(
                key=row["anf_id"],
                slug=new_speaker_slug(row["speaker_raw"]),
                tokens=tokenise(" ".join(row["paragraphs"])),
                note_tokens=tokenise(" ".join(row["notes"])),
            )
        )
    return by_party


def old_speeches(table: pa.Table, protocol_id: str) -> tuple[dict[str, list[Speech]], int]:
    """Read one protocol's older records into speeches, by party.

    Returns:
        The speeches by party, and how many records have an id whose slug
        differs from the slug of their speaker's name.
    """
    by_party: dict[str, list[Speech]] = defaultdict(list)
    disagreements = 0
    for row in table.to_pylist():
        slug = legacy_slug(row["id"], protocol_id, row["party"])
        disagreements += slug != speaker_slug(row["speaker"])
        for position, text in enumerate(split_legacy_text(row["text"])):
            by_party[row["party"]].append(
                Speech(key=legacy_key(row["id"], position), slug=slug, tokens=tokenise(text))
            )
    return by_party, disagreements


def reconcile(
    new_path: Path,
    old_path: Path,
    settings: ReconcileSettings,
    first_session: str,
    last_session: str,
) -> ReconciliationReport:
    """Reconcile the two corpora and account for every speech on both sides.

    Both corpora are read into Arrow memory once; Python objects are made for
    one protocol at a time.

    Args:
        new_path: The new corpus, ``anforanden.parquet``.
        old_path: The older corpus, ``speeches_full.parquet``.
        settings: The alignment's parameters, the coverage threshold and the
            caps on the lists in the report.
        first_session: The first session in scope, ``corpus.first_session``.
        last_session: The last session in scope, ``source.last_session``.

    Returns:
        The report.

    Raises:
        ValueError: If a session label is malformed or the first session is
            later than the last, a session in scope has no speech in the new
            corpus, the protocol ids of the new corpus do not follow one
            numbering of the sessions, the older corpus has no protocol of a
            session in scope, or an older record's id is malformed.
    """
    sessions = sessions_between(first_session, last_session)
    params = align_params(settings)
    new_table = pq.read_table(new_path, columns=NEW_COLUMNS)
    old_table = pq.read_table(old_path, columns=OLD_COLUMNS)
    logger.info("Read %d new speeches and %d older records", new_table.num_rows, old_table.num_rows)

    new_ids = new_table.column("dok_id").to_pylist()
    new_rm = new_table.column("rm").to_pylist()
    new_dates = new_table.column("date").to_pylist()
    new_sessions = set(new_rm)
    absent = [session for session in sessions if session not in new_sessions]
    if absent:
        raise ValueError(
            f"The new corpus holds no speech of these sessions in scope: {', '.join(absent)}."
        )
    offsets: dict[int, str] = {}
    for dok_id, rm in zip(new_ids, new_rm, strict=True):
        offsets.setdefault(prefix_offset(dok_id, rm), dok_id)
    if len(offsets) != 1:
        raise ValueError(
            "The protocol ids follow more than one numbering of the sessions; "
            f"the first protocol of each: {sorted(offsets.values())}."
        )
    offset = next(iter(offsets))
    scope = set(sessions)
    new_rows = index_by(new_ids)
    new_date_of = {dok_id: new_dates[rows[0]] for dok_id, rows in new_rows.items()}

    old_ids = old_table.column("protocol_id").to_pylist()
    old_dates = [stamp.date() for stamp in old_table.column("protocol_date").to_pylist()]
    old_texts = old_table.column("text")
    old_rows = index_by(old_ids)
    old_date_of = {protocol: old_dates[rows[0]] for protocol, rows in old_rows.items()}
    speech_counts = {
        protocol: sum(len(split_legacy_text(old_texts[row].as_py())) for row in rows)
        for protocol, rows in old_rows.items()
    }

    out_of_scope: dict[str, list[str]] = defaultdict(list)
    in_scope: list[str] = []
    for protocol in old_rows:
        session = session_of(protocol, offset)
        if session in scope:
            in_scope.append(protocol)
        else:
            out_of_scope[session].append(protocol)
    if not in_scope:
        raise ValueError(
            f"The older corpus holds no protocol of the sessions {first_session} to {last_session}."
        )
    first_day = min(old_date_of[protocol] for protocol in in_scope)
    last_day = max(old_date_of[protocol] for protocol in in_scope)
    in_window = {dok_id for dok_id, day in new_date_of.items() if first_day <= day <= last_day}
    new_counts = {dok_id: len(rows) for dok_id, rows in new_rows.items()}

    shared = sorted(
        set(in_scope) & in_window, key=lambda protocol: (old_date_of[protocol], protocol)
    )
    tally = _Tally(params, settings.listed_ids)
    old_by_year: Counter[int] = Counter()
    old_matched_by_year: Counter[int] = Counter()
    new_by_year: Counter[int] = Counter()
    new_matched_by_year: Counter[int] = Counter()
    slug_disagreements = 0
    for protocol in in_scope:
        old_by_year[old_date_of[protocol].year] += speech_counts[protocol]
    for dok_id in in_window:
        new_by_year[new_date_of[dok_id].year] += new_counts[dok_id]

    for done, protocol in enumerate(shared, start=1):
        old, disagreements = old_speeches(old_table.take(old_rows[protocol]), protocol)
        new = new_speeches(new_table.take(new_rows[protocol]))
        slug_disagreements += disagreements
        result = reconcile_protocol(old, new, params)
        speeches = {s.key: s for side in (old, new) for group in side.values() for s in group}
        tally.add(result, speeches, new_date_of[protocol].year)
        old_matched_by_year[old_date_of[protocol].year] += len(result.matches)
        new_matched_by_year[new_date_of[protocol].year] += len(result.matches)
        if done % 250 == 0:
            logger.info("%d of %d shared protocols, %d pairs", done, len(shared), tally.pairs)

    for protocol in set(in_scope) - in_window:
        old, disagreements = old_speeches(old_table.take(old_rows[protocol]), protocol)
        slug_disagreements += disagreements

    old_in_scope = sum(speech_counts[protocol] for protocol in in_scope)
    old_in_shared = sum(speech_counts[protocol] for protocol in shared)
    new_in_window = sum(new_counts[dok_id] for dok_id in in_window)
    new_in_shared = sum(new_counts[dok_id] for dok_id in shared)
    years = sorted(set(old_by_year) | set(new_by_year))
    out_rows = {
        session: sum(len(old_rows[protocol]) for protocol in protocols)
        for session, protocols in out_of_scope.items()
    }
    sessions_sorted = sorted(new_sessions)
    date_disagreements = sum(old_date_of[protocol] != new_date_of[protocol] for protocol in shared)

    old_corpus = OldCorpus(
        rows=old_table.num_rows,
        speeches=sum(speech_counts.values()),
        out_of_scope=tuple(
            OutOfScope(
                session=session,
                protocols=len(protocols),
                rows=out_rows[session],
                speeches=sum(speech_counts[protocol] for protocol in protocols),
                first_date=min(old_date_of[protocol] for protocol in protocols),
                last_date=max(old_date_of[protocol] for protocol in protocols),
            )
            for session, protocols in sorted(out_of_scope.items())
        ),
        in_scope_rows=sum(len(old_rows[protocol]) for protocol in in_scope),
        in_scope_speeches=old_in_scope,
        in_scope_protocols=len(in_scope),
        first_date=first_day,
        last_date=last_day,
        slug_disagreements=slug_disagreements,
    )
    new_corpus = NewCorpus(
        speeches=new_table.num_rows,
        protocols=len(new_rows),
        sessions=(sessions_sorted[0], sessions_sorted[-1]),
        in_window_speeches=new_in_window,
        in_window_protocols=len(in_window),
        protocol_date_disagreements=date_disagreements,
    )
    coverage = share(tally.pairs, old_in_scope)
    return ReconciliationReport(
        scope=SCOPE_DEFINITION,
        pairing=PAIRING_DEFINITION,
        sessions_in_scope=(first_session, last_session),
        old_corpus=old_corpus,
        new_corpus=new_corpus,
        coverage=coverage,
        coverage_threshold=settings.coverage_threshold,
        coverage_meets_threshold=meets_threshold(
            tally.pairs, old_in_scope, settings.coverage_threshold
        ),
        coverage_in_shared_protocols=share(tally.pairs, old_in_shared),
        reverse=share(tally.pairs, new_in_window),
        reverse_in_shared_protocols=share(tally.pairs, new_in_shared),
        methods=dict(sorted(tally.methods.items())),
        old_only_protocols=one_sided(
            set(in_scope) - in_window,
            {protocol: old_date_of[protocol].year for protocol in in_scope},
            speech_counts,
            settings.listed_ids,
        ),
        new_only_protocols=one_sided(
            in_window - set(in_scope),
            {dok_id: new_date_of[dok_id].year for dok_id in in_window},
            new_counts,
            settings.listed_ids,
        ),
        unmatched_old=tally.left_over("old"),
        unmatched_new=tally.left_over("new"),
        years=tuple(
            YearRow(
                year=year,
                old_speeches=old_by_year[year],
                old_matched=old_matched_by_year[year],
                coverage=share(old_matched_by_year[year], old_by_year[year]).share,
                new_speeches=new_by_year[year],
                new_matched=new_matched_by_year[year],
                reverse_share=share(new_matched_by_year[year], new_by_year[year]).share,
                pairs_missing_words=tally.by_year[year]["missing_words"],
                missing_token_share=share(
                    tally.by_year[year]["missing"], tally.by_year[year]["new_tokens"]
                ).share,
            )
            for year in years
        ),
        overlap=tally.overlap(),
        surplus=tally.surplus(),
        top_missing_tokens=dict(tally.missing_tokens.most_common(settings.listed_tokens)),
        top_surplus_tokens=dict(tally.surplus_tokens.most_common(settings.listed_tokens)),
        largest_deviations=tally.largest(settings.listed_deviations, by_missing=False),
        largest_missing=tally.largest(settings.listed_missing, by_missing=True),
        params=params,
    )


def main() -> None:
    """Parse the command line and reconcile the two corpora."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    new_path = settings.data.search / SPEECHES_NAME
    old_path = settings.data.legacy_corpus
    for path in (new_path, old_path):
        if not path.is_file():
            raise SystemExit(f"{path} is missing.")

    report = reconcile(
        new_path,
        old_path,
        settings.reconcile,
        settings.corpus.first_session,
        settings.source.last_session,
    )
    write_report(report, settings.tracked.reports / REPORT_NAME)

    logger.info(
        "Coverage %d of %d = %.4f (threshold %s met: %s)",
        report.coverage.numerator,
        report.coverage.denominator,
        report.coverage.share,
        report.coverage_threshold,
        report.coverage_meets_threshold,
    )
    logger.info(
        "Reverse %d of %d = %.4f",
        report.reverse.numerator,
        report.reverse.denominator,
        report.reverse.share,
    )
    logger.info("Report written to %s", settings.tracked.reports / REPORT_NAME)


if __name__ == "__main__":
    main()
