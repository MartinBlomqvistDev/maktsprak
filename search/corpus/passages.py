"""Packing a speech's paragraphs into passages.

The indexes hold passages, and a quotation is always one or more whole
paragraphs of a single speech. A passage is therefore a run of consecutive
whole paragraphs from one speech: a paragraph is never split, never dropped,
and never shared between two passages.

A speech is packed by a balanced partition, which decides how many passages the
speech needs and cuts the speech into that many of nearly equal size. Filling
each passage up to the target in turn would leave whatever remains as the last
passage, however short. See :func:`partition` for the rule.

A word is a run of characters between whitespace, the same count the build
report uses.
"""

from __future__ import annotations

import logging
import statistics
from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import date
from itertools import accumulate
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from search.corpus.store import iter_speeches, write_passages
from search.models import Anforande, Passage, passage_id

logger = logging.getLogger("passages")

PASSAGES_REPORT_NAME = "passages.json"

#: Passages are written this many at a time, one Parquet row group each.
WRITE_BATCH_PASSAGES = 20_000

#: A passage with fewer words than this is counted as short in the report.
SHORT_BELOW_WORDS = 50

#: A passage with more words than this is counted as long in the report.
LONG_ABOVE_WORDS = 320

#: A passage with more words than this is counted as very long in the report.
VERY_LONG_ABOVE_WORDS = 512

#: How many of the largest passages the report names.
LARGEST_KEPT = 10

#: The first year of each period the report groups passages into, by the year
#: of the speech. A period ends with the year before the next period starts.
#: The last period is open-ended: its label ends with the latest year among the
#: speeches packed.
PERIOD_FIRST_YEARS: tuple[int, ...] = (2003, 2010, 2018)


def count_words(paragraph: str) -> int:
    """Count the words of a paragraph, split on whitespace."""
    return len(paragraph.split())


def paragraph_word_counts(speech: Anforande) -> list[int]:
    """Count the words of each paragraph of a speech, in order."""
    return [count_words(paragraph) for paragraph in speech.paragraphs]


def _nearest_boundary(
    cumulative: Sequence[int], scaled_ideal: int, scale: int, low: int, high: int
) -> int:
    """Pick the boundary in ``[low, high]`` whose cumulative words are nearest an ideal.

    The ideal is ``scaled_ideal / scale`` words, kept as a pair of integers so
    that a tie is decided by arithmetic and never by rounding. When two
    boundaries are equally near, the earlier one is taken.
    """
    index = bisect_left(cumulative, scaled_ideal, low, high + 1, key=lambda words: scale * words)
    candidates = [boundary for boundary in (index - 1, index) if low <= boundary <= high]
    return min(candidates, key=lambda boundary: abs(scale * cumulative[boundary] - scaled_ideal))


def partition(word_counts: Sequence[int], target_words: int) -> list[tuple[int, int]]:
    """Cut a speech into runs of consecutive paragraphs of nearly equal size.

    With ``W`` words in the speech and a target of ``T``, the speech is cut into
    ``k = max(1, round(W / T))`` passages, where a half rounds up: 300 words at a
    target of 200 make two passages, 299 make one. A speech with fewer than
    ``k`` paragraphs gets one passage per paragraph.

    The ``j``-th cut, for ``j`` from 1 to ``k - 1``, goes at the paragraph
    boundary whose cumulative word count is nearest ``j * W / k``. Two boundaries
    equally near the ideal are a tie, and the earlier boundary wins, which makes
    the earlier passage the shorter. A cut is moved if it would otherwise leave
    a passage empty: it never lies at or before the previous cut, and it leaves
    at least one paragraph for each passage still to come.

    A paragraph is never split, so a paragraph longer than the target ends up
    as a passage of its own, or together with neighbours, wherever the balance
    puts the nearest cut. The result depends only on the word counts and the
    target.

    Every paragraph of a speech holds at least one word, so a count below one
    is not a valid input and is refused.

    Args:
        word_counts: The words in each paragraph, in order. Each is at least one.
        target_words: The size to pack towards, in words.

    Returns:
        One ``(first, last)`` pair per passage, the index of its first and of
        its last paragraph, both included. The pairs are consecutive and cover
        every paragraph exactly once.

    Raises:
        ValueError: If the target is below one word, there are no paragraphs,
            or a paragraph has fewer than one word.
    """
    if target_words < 1:
        raise ValueError(f"The target must be at least one word, not {target_words}.")
    paragraphs = len(word_counts)
    if paragraphs == 0:
        raise ValueError("A speech needs at least one paragraph to be packed.")
    if min(word_counts) < 1:
        raise ValueError("Every paragraph needs at least one word to be packed.")

    cumulative = [0, *accumulate(word_counts)]
    total = cumulative[-1]
    wanted = max(1, (2 * total + target_words) // (2 * target_words))
    passages = min(wanted, paragraphs)

    cuts = [0]
    for cut_number in range(1, passages):
        low = cuts[-1] + 1
        high = paragraphs - (passages - cut_number)
        cuts.append(_nearest_boundary(cumulative, cut_number * total, passages, low, high))
    cuts.append(paragraphs)
    return [(start, end - 1) for start, end in zip(cuts, cuts[1:], strict=False)]


def pack(speech: Anforande, target_words: int) -> tuple[Passage, ...]:
    """Pack a speech into passages, see :func:`partition` for the rule.

    Args:
        speech: The speech to pack.
        target_words: The size to pack towards, in words.

    Returns:
        The speech's passages, in order. Each carries the speech's party, date
        and speaker, which a search filters on.

    Raises:
        ValueError: If the target is below one word, see :func:`partition`. A
            speech always has paragraphs and every paragraph has a word, so
            the target is the only input that can be refused.
    """
    counts = paragraph_word_counts(speech)
    return tuple(
        Passage(
            passage_id=passage_id(speech.anf_id, first, last),
            anf_id=speech.anf_id,
            first=first,
            last=last,
            words=sum(counts[first : last + 1]),
            party=speech.party,
            date=speech.date,
            person_id=speech.person_id,
        )
        for first, last in partition(counts, target_words)
    )


def passage_paragraphs(speech: Anforande, passage: Passage) -> tuple[str, ...]:
    """Return the paragraphs of a passage, as they stand in its speech.

    Args:
        speech: The speech the passage was packed from.
        passage: The passage.

    Returns:
        The passage's paragraphs, whole and in order.

    Raises:
        ValueError: If the passage belongs to another speech, reaches past the
            speech's last paragraph, or counts another number of words than its
            paragraphs hold. A wrong count means that the speech's text is not
            the text the passage was packed from.
    """
    if passage.anf_id != speech.anf_id:
        raise ValueError(f"{passage.passage_id} does not belong to speech {speech.anf_id}.")
    if passage.last >= len(speech.paragraphs):
        raise ValueError(
            f"{passage.passage_id} reaches past the {len(speech.paragraphs)} paragraphs "
            f"of {speech.anf_id}."
        )
    paragraphs = speech.paragraphs[passage.first : passage.last + 1]
    held = sum(count_words(paragraph) for paragraph in paragraphs)
    if held != passage.words:
        raise ValueError(
            f"{passage.passage_id} counts {passage.words} words; its paragraphs hold {held}."
        )
    return paragraphs


def check_cover(speech: Anforande, passages: Sequence[Passage]) -> None:
    """Check that passages cover a speech: every paragraph once, in order.

    Args:
        speech: The speech.
        passages: Its passages.

    Raises:
        ValueError: If a passage belongs to another speech or carries another
            party, date or speaker than the speech; if a passage leaves a gap
            or an overlap, reaches past the last paragraph or counts another
            number of words than its paragraphs hold; or if the passages stop
            short of the last paragraph.
    """
    next_first = 0
    for passage in passages:
        if (passage.party, passage.date, passage.person_id) != (
            speech.party,
            speech.date,
            speech.person_id,
        ):
            raise ValueError(f"{passage.passage_id} carries other filter fields than its speech.")
        if passage.first != next_first:
            raise ValueError(
                f"{passage.passage_id} starts at paragraph {passage.first}, "
                f"where {next_first} was expected."
            )
        passage_paragraphs(speech, passage)
        next_first = passage.last + 1
    if next_first != len(speech.paragraphs):
        raise ValueError(
            f"The passages of {speech.anf_id} end at paragraph {next_first - 1} "
            f"of {len(speech.paragraphs)}."
        )


class _Frozen(BaseModel):
    """Base for report models: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CorpusTotals(_Frozen):
    """The size of a body of speeches, or of the passages cut from it.

    Attributes:
        speeches: The speeches.
        paragraphs: The paragraphs in them.
        words: The words in those paragraphs.
    """

    speeches: int
    paragraphs: int
    words: int


class CoverCheck(_Frozen):
    """The check that the passages hold exactly the corpus's paragraphs and words.

    Attributes:
        build_report: The totals the corpus build reported.
        read: The totals counted in the speeches as they were read for packing.
        in_passages: The totals counted in the passages: the speeches that
            have at least one passage, and the paragraphs and words they span.
        exact: Whether the three agree.
    """

    build_report: CorpusTotals
    read: CorpusTotals
    in_passages: CorpusTotals
    exact: bool


class PassagesPerSpeech(_Frozen):
    """How many passages a speech is packed into.

    Attributes:
        mean: The mean.
        median: The median.
        p90: The 90th percentile, see :func:`percentile`.
        max: The most any speech has.
    """

    mean: float
    median: float
    p90: int
    max: int


class WordsPerPassage(_Frozen):
    """How many words a passage holds.

    Percentiles are taken by :func:`percentile`.

    Attributes:
        min: The fewest.
        p10: The 10th percentile.
        median: The median.
        mean: The mean.
        p90: The 90th percentile.
        p99: The 99th percentile.
        max: The most.
    """

    min: int
    p10: int
    median: float
    mean: float
    p90: int
    p99: int
    max: int


class Band(_Frozen):
    """The passages beyond a size limit.

    Attributes:
        passages: How many passages.
        share: Their share of all passages, to six decimals.
    """

    passages: int
    share: float


class GroupTally(_Frozen):
    """The passages of one party or one period.

    Attributes:
        passages: How many passages.
        words: The words in them.
    """

    passages: int
    words: int


class LargestPassage(_Frozen):
    """One of the largest passages.

    Attributes:
        passage_id: The passage's key.
        words: Its words.
        paragraphs: The paragraphs it holds.
    """

    passage_id: str
    words: int
    paragraphs: int


class TrialTarget(_Frozen):
    """What another target size would give, counted without writing any passages.

    Attributes:
        target_words: The target.
        passages: How many passages the corpus would be packed into.
        median_words: The median words per passage.
    """

    target_words: int
    passages: int
    median_words: float


class PassagesReport(_Frozen):
    """What packing the corpus into passages made.

    Attributes:
        inputs_sha256: The hash of the corpus build's inputs, copied from its report.
        target_words: The size the passages were packed towards.
        speeches: The speeches packed.
        passages: The passages made.
        passages_per_speech: How many passages a speech has.
        words_per_passage: How many words a passage has.
        size_bands: The passages below the short limit and above the two long
            limits, keyed by the limit in words.
        passages_by_party: The passages of each party.
        passages_by_period: The passages of each period that holds a speech, by
            the speech's year.
            The label of the last period ends with the latest year among the
            speeches packed.
        largest_passages: The largest passages, largest first.
        trial_targets: What the other target sizes would give.
        cover: The check that the passages hold exactly the corpus's paragraphs and words.
    """

    inputs_sha256: str
    target_words: int
    speeches: int
    passages: int
    passages_per_speech: PassagesPerSpeech
    words_per_passage: WordsPerPassage
    size_bands: dict[str, Band]
    passages_by_party: dict[str, GroupTally]
    passages_by_period: dict[str, GroupTally]
    largest_passages: tuple[LargestPassage, ...]
    trial_targets: tuple[TrialTarget, ...]
    cover: CoverCheck


def percentile(ordered: Sequence[int], percent: int) -> int:
    """Take a percentile of sorted values: the value at index ``(n - 1) * percent // 100``.

    Args:
        ordered: The ``n`` values, smallest first. There must be at least one.
        percent: The percentile, from 0 to 100.

    Returns:
        The value at that index.
    """
    return ordered[(len(ordered) - 1) * percent // 100]


def period_of(day: date) -> int:
    """Find the period a date falls in.

    Args:
        day: The date of a speech.

    Returns:
        The first year of the period, one of ``PERIOD_FIRST_YEARS``.

    Raises:
        ValueError: If the date lies before the first period.
    """
    first_period = min(PERIOD_FIRST_YEARS)
    if day.year < first_period:
        raise ValueError(
            f"{day} lies before {first_period}, the first year of the report's periods."
        )
    return max(first_year for first_year in PERIOD_FIRST_YEARS if first_year <= day.year)


def period_label(first_year: int, latest_year: int) -> str:
    """Name a period by its first and its last year.

    Args:
        first_year: The first year of the period, one of ``PERIOD_FIRST_YEARS``.
        latest_year: The latest year among the speeches packed. It ends the
            label of the last period, which has no last year of its own.

    Returns:
        The label, for example ``2010-2017``.
    """
    later = [year for year in PERIOD_FIRST_YEARS if year > first_year]
    last_year = min(later) - 1 if later else latest_year
    return f"{first_year}-{last_year}"


def _band_key(prefix: str, limit: int) -> str:
    return f"{prefix}_{limit}_words"


class _Tally:
    """The running account of packing, turned into a report at the end."""

    def __init__(self, trial_targets: Sequence[int]) -> None:
        self.read = [0, 0, 0]
        self.in_passages = [0, 0, 0]
        self.passages_per_speech: list[int] = []
        self.words_per_passage: list[int] = []
        self.by_party: dict[str, list[int]] = {}
        self.by_period: dict[int, list[int]] = {}
        self.latest_year = 0
        self.largest: list[LargestPassage] = []
        self.trial_words: dict[int, list[int]] = {target: [] for target in trial_targets}
        self.cover: CoverCheck | None = None

    def add(self, speech: Anforande, passages: Sequence[Passage]) -> None:
        """Account for one speech and the passages it was packed into.

        Args:
            speech: The speech.
            passages: Its passages.

        Raises:
            ValueError: If the speech is dated before the first period, with
                the speech's key in the message, or if a trial target is below
                one word.
        """
        try:
            period = period_of(speech.date)
        except ValueError as error:
            raise ValueError(f"{speech.anf_id}: {error}") from error
        self.latest_year = max(self.latest_year, speech.date.year)
        counts = paragraph_word_counts(speech)
        self.read[0] += 1
        self.read[1] += len(counts)
        self.read[2] += sum(counts)

        self.in_passages[0] += bool(passages)
        self.passages_per_speech.append(len(passages))
        for passage in passages:
            paragraphs = passage.last - passage.first + 1
            self.in_passages[1] += paragraphs
            self.in_passages[2] += passage.words
            self.words_per_passage.append(passage.words)
            for tally in (
                self.by_party.setdefault(passage.party, [0, 0]),
                self.by_period.setdefault(period, [0, 0]),
            ):
                tally[0] += 1
                tally[1] += passage.words
            self._offer_largest(passage, paragraphs)

        for target, words in self.trial_words.items():
            words.extend(sum(counts[first : last + 1]) for first, last in partition(counts, target))

    def _offer_largest(self, passage: Passage, paragraphs: int) -> None:
        full = len(self.largest) >= LARGEST_KEPT
        if full and passage.words < self.largest[-1].words:
            return
        self.largest.append(
            LargestPassage(
                passage_id=passage.passage_id, words=passage.words, paragraphs=paragraphs
            )
        )
        self.largest.sort(key=lambda entry: (-entry.words, entry.passage_id))
        del self.largest[LARGEST_KEPT:]

    def verify(self, corpus: CorpusTotals) -> None:
        """Compare the three accounts of the corpus and keep the result.

        Args:
            corpus: The speeches, paragraphs and words the corpus build reported.

        Raises:
            ValueError: If the speeches, paragraphs and words counted in the
                passages differ from those counted as the speeches were read,
                or from those the corpus build reported.
        """
        read = CorpusTotals(speeches=self.read[0], paragraphs=self.read[1], words=self.read[2])
        in_passages = CorpusTotals(
            speeches=self.in_passages[0],
            paragraphs=self.in_passages[1],
            words=self.in_passages[2],
        )
        if not corpus == read == in_passages:
            raise ValueError(
                "The passages do not cover the corpus. "
                f"Build report: {corpus}. Read: {read}. In passages: {in_passages}."
            )
        self.cover = CoverCheck(build_report=corpus, read=read, in_passages=in_passages, exact=True)

    def report(self, inputs_sha256: str, target_words: int) -> PassagesReport:
        """Freeze the account into the report, once :meth:`verify` has passed.

        Args:
            inputs_sha256: The hash of the corpus build's inputs.
            target_words: The size the passages were packed towards.

        Returns:
            The report.

        Raises:
            ValueError: If :meth:`verify` has not passed, or no passage was made.
        """
        if self.cover is None or not self.words_per_passage:
            raise ValueError("There is nothing to report: no speeches were packed and verified.")
        passages = len(self.words_per_passage)
        ordered = sorted(self.words_per_passage)
        per_speech = sorted(self.passages_per_speech)

        def band(is_beyond: Callable[[int], bool]) -> Band:
            count = sum(1 for words in ordered if is_beyond(words))
            return Band(passages=count, share=round(count / passages, 6))

        return PassagesReport(
            inputs_sha256=inputs_sha256,
            target_words=target_words,
            speeches=len(per_speech),
            passages=passages,
            passages_per_speech=PassagesPerSpeech(
                mean=round(statistics.fmean(per_speech), 2),
                median=statistics.median(per_speech),
                p90=percentile(per_speech, 90),
                max=per_speech[-1],
            ),
            words_per_passage=WordsPerPassage(
                min=ordered[0],
                p10=percentile(ordered, 10),
                median=statistics.median(ordered),
                mean=round(statistics.fmean(ordered), 1),
                p90=percentile(ordered, 90),
                p99=percentile(ordered, 99),
                max=ordered[-1],
            ),
            size_bands={
                _band_key("under", SHORT_BELOW_WORDS): band(lambda w: w < SHORT_BELOW_WORDS),
                _band_key("over", LONG_ABOVE_WORDS): band(lambda w: w > LONG_ABOVE_WORDS),
                _band_key("over", VERY_LONG_ABOVE_WORDS): band(lambda w: w > VERY_LONG_ABOVE_WORDS),
            },
            passages_by_party={
                party: GroupTally(passages=count, words=words)
                for party, (count, words) in sorted(self.by_party.items())
            },
            passages_by_period={
                period_label(first_year, self.latest_year): GroupTally(passages=count, words=words)
                for first_year, (count, words) in sorted(self.by_period.items())
            },
            largest_passages=tuple(self.largest),
            trial_targets=tuple(
                TrialTarget(
                    target_words=target,
                    passages=len(sizes),
                    median_words=statistics.median(sizes),
                )
                for target, sizes in self.trial_words.items()
            ),
            cover=self.cover,
        )


def _packed_batches(
    speeches: Iterable[Anforande],
    target_words: int,
    tally: _Tally,
    corpus: CorpusTotals,
) -> Iterator[list[Passage]]:
    """Pack speeches one at a time and hand the passages on in batches.

    Each speech's passages are checked with :func:`check_cover` before they are
    handed on. When the speeches run out, the totals are checked against the
    corpus. Both checks run before the writer finishes, so a mismatch stops the
    write, leaves no partial file and leaves any earlier passages file as it was.

    Args:
        speeches: The speeches to pack.
        target_words: The size to pack towards, in words.
        tally: The running account, which every speech is added to.
        corpus: The speeches, paragraphs and words the corpus build reported.

    Yields:
        Lists of passages. Every list but the last holds at least
        ``WRITE_BATCH_PASSAGES``.

    Raises:
        ValueError: If a target is below one word, if a speech's passages do
            not cover the speech, if a speech is dated before the first period,
            or if the totals differ from the corpus's.
    """
    batch: list[Passage] = []
    for speech in speeches:
        passages = pack(speech, target_words)
        check_cover(speech, passages)
        tally.add(speech, passages)
        batch.extend(passages)
        if len(batch) >= WRITE_BATCH_PASSAGES:
            yield batch
            batch = []
    if batch:
        yield batch
    tally.verify(corpus)


def build_passages(
    speeches_path: Path,
    destination: Path,
    target_words: int,
    trial_targets: Sequence[int],
    corpus: CorpusTotals,
    inputs_sha256: str,
) -> PassagesReport:
    """Pack every speech of the corpus into passages and write them to Parquet.

    Args:
        speeches_path: The Parquet file of speeches.
        destination: The Parquet file to write the passages to.
        target_words: The size to pack towards, in words.
        trial_targets: Other sizes to count the passages of, without writing them.
        corpus: The speeches, paragraphs and words the corpus build reported.
        inputs_sha256: The hash of the corpus build's inputs, copied to the report.

    Returns:
        The account of what the packing made, with the cover check.

    Raises:
        ValueError: If the corpus build reported no speeches, if a target is
            below one word, if a speech is not covered exactly or is dated
            before the first period, or if the passages do not hold exactly the
            corpus's paragraphs and words. No passages file is written in any
            of these cases, and an earlier one is left as it was.
    """
    if corpus.speeches < 1:
        raise ValueError("The corpus build reported no speeches, so there is nothing to pack.")
    tally = _Tally(trial_targets)
    speeches = iter_speeches(speeches_path)
    written = write_passages(destination, _packed_batches(speeches, target_words, tally, corpus))
    logger.info("%d passages written", written)
    return tally.report(inputs_sha256, target_words)
