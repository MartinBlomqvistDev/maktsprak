"""Dimension-agnostic kernel for the tone/rhetoric analytics.

Every tone dimension — lexical (vi-mot-dom, absolutord), stylometric (LIX),
or structural (``hen``, frågeteckenfrekvens) — measures something different,
but they all need the *same* downstream machinery: a rate, a prior that keeps
thin cells from looking extreme, a distinctiveness score, a suppression rule,
and reproducible receipts.  Building that once, here, is the point: get the
statistics right in one place rather than eleven times.

A dimension supplies exactly one thing — a ``measure_fn`` that takes the
speech DataFrame and returns it with ``hits``/``n`` (and, for dimensions that
can point at a specific span of text, ``spans``) — and registers a
:class:`DimensionSpec`.  Everything below is shared.

Statistical notes
-----------------
The smoothing and the z-score both use the **informative Dirichlet prior** of
Monroe, Colaresi & Quinn (2008), the same method already used for the
distinctiveness word clouds (:mod:`..distinctiveness`) and the issue frames
(:mod:`..drift`).  Reusing it is deliberate: the tone numbers are then right
for exactly the same reason the word clouds are, and :func:`fightin_z` is a
thin wrapper over :func:`~..distinctiveness.weighted_log_odds` rather than a
second, parallel implementation of the same math that could drift out of sync.

The prior's strength is ``alpha * background_n`` — i.e. it scales with how
much was said in that (frame, year), not with a fixed pseudo-count.  A cell
whose own ``n`` is small relative to that prior mass is pulled toward the
all-party background rate; a large cell is barely moved.  That is the
"low-sample years must not look artificially extreme" requirement, and it is
the *only* thing standing between a party that spoke twice in 2003 and a
headline-grabbing spike.  ``alpha`` is therefore a load-bearing constant and
is flagged for a sensitivity check (see DEV_LOG §12) rather than treated as
settled.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from ..distinctiveness import weighted_log_odds

# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------

# Swedish abbreviations whose internal periods must not be read as sentence
# ends.  Masking them (rather than post-hoc re-joining) keeps the segmentation
# a single pass, and the mask is length-preserving so character offsets into
# the *original* string stay valid — which is what makes receipts able to point
# at an exact span of the real text.
_ABBREVIATIONS: tuple[str, ...] = (
    "bl.a.",
    "t.ex.",
    "d.v.s.",
    "dvs.",
    "m.m.",
    "m.fl.",
    "o.s.v.",
    "osv.",
    "fr.o.m.",
    "t.o.m.",
    "p.g.a.",
    "s.k.",
    "f.d.",
    "e.d.",
    "etc.",
    "ca.",
    "kl.",
    "nr.",
    "resp.",
    "ev.",
    "jfr.",
    "enl.",
    "ang.",
    "milj.",
    "mkr.",
    "proc.",
    "st.",
    "kr.",
)

# Longest-first, so "m.fl." is tried before "m.m." can half-match anything.
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + ")",
    re.IGNORECASE,
)

#: Placeholder for a masked period.  Single character, so offsets are preserved.
_MASK = "\x00"

# A sentence ends at .!? followed by whitespace (optionally through a closing
# quote or bracket).  The lookbehind keeps the punctuation with the sentence it
# terminates.  Requiring trailing whitespace is what stops "3.5" or "kl. 14"
# from splitting mid-number.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'’»)\]]*\s+")


def _mask_abbreviations(text: str) -> str:
    """Replace periods inside known abbreviations with :data:`_MASK`.

    Length-preserving by construction (one character in, one out), so offsets
    into the returned string index the same characters as in *text*.
    """
    return _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _MASK), text)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character offsets ``[start, end)`` of each sentence in *text*.

    Offsets index *text* itself, not a normalised copy — a receipt can slice
    the original string and highlight an exact span inside it.

    Abbreviations (``bl.a.``, ``t.ex.``, ``m.fl.`` …) do not end a sentence.
    The trade-off is deliberate: an abbreviation that genuinely *does* end a
    sentence ("... och m.m. Sedan ...") is under-split rather than over-split,
    because a missed boundary merely lengthens a receipt's context, while a
    false boundary would cut a quote in half.

    Args:
        text: Raw speech text.

    Returns:
        ``[(start, end), ...]`` in order, whitespace-trimmed, no empty spans.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    masked = _mask_abbreviations(text)
    raw: list[tuple[int, int]] = []
    start = 0
    for match in _SENT_SPLIT_RE.finditer(masked):
        raw.append((start, match.start()))
        start = match.end()
    raw.append((start, len(text)))

    spans: list[tuple[int, int]] = []
    for begin, end in raw:
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if end > begin:
            spans.append((begin, end))
    return spans


def sentence_split(text: str) -> list[str]:
    """Sentences of *text*, whitespace-trimmed (see :func:`sentence_spans`)."""
    return [text[start:end] for start, end in sentence_spans(text)]


# --------------------------------------------------------------------------
# Pattern tables
# --------------------------------------------------------------------------


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile *pattern* as a case-insensitive, word-boundary-anchored literal.

    The boundaries are the whole point.  A raw substring search — which is what
    :func:`..lexicon.apply_ton_lexicon` still does — counts ``"ta över"`` inside
    ``"prata över"`` and ``"hen"`` inside ``"Henrik"``.  ``(?<!\\w)``/``(?!\\w)``
    are Unicode-aware, so å/ä/ö count as word characters and ``"över"`` does not
    match inside ``"överens"``.

    Args:
        pattern: A literal word or phrase (not a regex — it is escaped).

    Returns:
        Compiled pattern matching *pattern* as a whole word/phrase.
    """
    return re.compile(r"(?<!\w)" + re.escape(pattern.strip().lower()) + r"(?!\w)", re.IGNORECASE)


def load_pattern_table(path: Path, key_col: str, required_cols: list[str]) -> pd.DataFrame:
    """Load a pattern/marker CSV, failing loudly on the mistakes that hide.

    Same discipline as :func:`..drift.frame_trajectories`' rejection of
    whitespace-containing stems: a pattern table that is silently wrong scores
    zero (or double) forever and nobody notices.  A duplicate key in
    particular is the bug that produced the old tone lexicon's 44
    multi-category words — the same word counted twice under two headings,
    quietly inflating both.

    Args:
        path:          CSV to load.
        key_col:       Column holding the pattern/word (checked for duplicates).
        required_cols: Columns that must be present.

    Returns:
        The table, with *key_col* stripped and lower-cased, plus a ``regex``
        column of compiled patterns.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If a required column is missing, the table is empty,
            a key is blank, or a key occurs more than once.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Pattern table not found: {path}")

    table = pd.read_csv(path)
    missing = [c for c in required_cols if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")
    if table.empty:
        raise ValueError(
            f"{path.name} is empty — a dimension with no patterns scores zero forever."
        )

    table[key_col] = table[key_col].astype(str).str.strip().str.lower()

    blank = table[table[key_col].isin({"", "nan"})]
    if not blank.empty:
        raise ValueError(f"{path.name} has {len(blank)} blank value(s) in '{key_col}'.")

    duplicated = table[key_col][table[key_col].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"{path.name} has duplicate '{key_col}' value(s), which would be counted "
            f"more than once: {duplicated}"
        )

    table["regex"] = [compile_pattern(p) for p in table[key_col]]
    return table


def find_spans(text: str, table: pd.DataFrame, key_col: str) -> list[tuple[int, int, str]]:
    """Every match of every pattern in *table*, as ``(start, end, pattern)``.

    Offsets index *text*, so a receipt can highlight the exact span.

    Args:
        text:    Raw speech text.
        table:   Output of :func:`load_pattern_table` (needs a ``regex`` column).
        key_col: Column holding the pattern string.

    Returns:
        Matches sorted by start offset.  Overlaps are *not* removed — two
        patterns may legitimately both fire on the same words, and collapsing
        them would silently under-count.
    """
    if not isinstance(text, str) or not text:
        return []
    spans: list[tuple[int, int, str]] = []
    for pattern, regex in zip(table[key_col], table["regex"], strict=True):
        for match in regex.finditer(text):
            spans.append((match.start(), match.end(), pattern))
    return sorted(spans)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def raw_rate(hits: int, n: int, per: float = 1000.0) -> float:
    """Unsmoothed rate: *hits* per *per* units of *n*.  ``0.0`` when ``n == 0``."""
    if n <= 0:
        return 0.0
    return hits / n * per


def smoothed_rate(
    hits: int,
    n: int,
    background_hits: int,
    background_n: int,
    alpha: float = 0.01,
    per: float = 1000.0,
) -> float:
    """Rate shrunk toward the pooled background rate by an informative prior.

    The estimator is::

        (hits + alpha * background_hits) / (n + alpha * background_n) * per

    which is a Beta/Dirichlet posterior mean whose prior has mean
    ``background_hits / background_n`` (the all-party rate for that year and
    frame) and strength ``alpha * background_n`` in units of *n*.

    The behaviour that matters:

    - ``n == 0``  -> returns exactly the background rate (nothing observed, so
      the estimate is the prior).
    - ``n`` small relative to ``alpha * background_n`` -> pulled hard toward the
      background.  This is what stops a party that spoke twice in 2003 from
      posting a 500-per-1000 spike off a single sentence.
    - ``n`` large -> converges on the raw rate; the prior washes out.

    Args:
        hits:            Marker hits in this cell.
        n:               Denominator for this cell (tokens or sentences).
        background_hits: Pooled hits across all groups in the same year/frame.
        background_n:    Pooled denominator across all groups, same scope.
        alpha:           Prior scale.  Effective prior strength is
            ``alpha * background_n``.  Defaults to 0.01, matching
            :func:`..distinctiveness.weighted_log_odds`.
        per:             Rate scale (default per 1 000).

    Returns:
        The smoothed rate.

    Raises:
        ValueError: If the counts are impossible (negative, or a cell larger
            than the background that is supposed to contain it).
    """
    _validate_counts(hits, n, background_hits, background_n)
    if background_n <= 0:
        return raw_rate(hits, n, per)

    denominator = n + alpha * background_n
    if denominator <= 0:
        return 0.0
    return (hits + alpha * background_hits) / denominator * per


def fightin_z(
    hits: int,
    n: int,
    background_hits: int,
    background_n: int,
    alpha: float = 0.01,
) -> float | None:
    """Fightin' Words z-score for this cell's marker rate vs. the pooled rest.

    Reuses :func:`..distinctiveness.weighted_log_odds` verbatim on a
    **two-token vocabulary** — ``MARKER`` and ``OTHER`` — so the cell's rate is
    scored against the other groups exactly the way a word is scored against
    the other parties in the distinctiveness clouds.  No second implementation
    of the same statistic: if the word clouds are right, this is right.

    The "rest" is the background *minus this cell*, so a cell is never compared
    against a pool that contains itself.

    This is metadata, never an axis: the UI may mark a point as distinguishable
    from the other parties when ``|z| > 1.96``, but the chart always plots the
    rate.

    Args:
        hits:            Marker hits in this cell.
        n:               Denominator for this cell.
        background_hits: Pooled hits across all groups, same year/frame.
        background_n:    Pooled denominator across all groups, same scope.
        alpha:           Prior scale (see :func:`smoothed_rate`).

    Returns:
        The z-score, or ``None`` where no comparison is defined: an empty cell,
        no other group to compare against, or a marker that never occurs (or
        always occurs) in the background.

    Raises:
        ValueError: If the counts are impossible.
    """
    _validate_counts(hits, n, background_hits, background_n)

    rest_n = background_n - n
    rest_hits = background_hits - hits

    # No data, no comparison group, or a degenerate background in which the
    # marker never fires (or fires on literally everything) — in all of these
    # the log-odds ratio is undefined or meaningless.  Say so rather than
    # emitting a number that looks real.
    if n <= 0 or rest_n <= 0:
        return None
    if background_hits <= 0 or background_hits >= background_n:
        return None

    group_counts: dict[str, Counter[str]] = {
        "cell": Counter({"MARKER": hits, "OTHER": n - hits}),
        "rest": Counter({"MARKER": rest_hits, "OTHER": rest_n - rest_hits}),
    }
    # min_count=0: the two-token vocabulary must survive the rare-word filter
    # that weighted_log_odds applies to real vocabularies.
    scores = weighted_log_odds(group_counts, alpha=alpha, min_count=0)
    return float(scores["cell"]["MARKER"])


def _validate_counts(hits: int, n: int, background_hits: int, background_n: int) -> None:
    """Reject count combinations that cannot arise from a correct measure_fn."""
    if hits < 0 or n < 0 or background_hits < 0 or background_n < 0:
        raise ValueError(
            f"Negative count: hits={hits}, n={n}, "
            f"background_hits={background_hits}, background_n={background_n}"
        )
    if hits > n:
        raise ValueError(f"hits ({hits}) exceeds n ({n}) — the denominator cannot be smaller.")
    if background_hits > background_n:
        raise ValueError(
            f"background_hits ({background_hits}) exceeds background_n ({background_n})."
        )
    if hits > background_hits or n > background_n:
        raise ValueError(
            f"Cell (hits={hits}, n={n}) is larger than the background "
            f"(hits={background_hits}, n={background_n}) that should contain it — "
            "the background must be the pooled total *including* this cell."
        )


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellStats:
    """One (group, year) cell of a dimension."""

    hits: int
    n: int
    speeches: int
    rate: float
    smoothed: float
    z: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        """JSON-ready form.  Raw counts are always published: a reader must be
        able to see that a rate of 4.0 came from 2 hits in 500 sentences."""
        return {
            "hits": self.hits,
            "n": self.n,
            "speeches": self.speeches,
            "rate": round(self.rate, 3),
            "smoothed": round(self.smoothed, 3),
            "z": None if self.z is None else round(self.z, 2),
        }


def aggregate_cells(
    df: pd.DataFrame,
    hit_col: str = "hits",
    n_col: str = "n",
    group_col: str = "party",
    year_col: str = "year",
    per: float = 1000.0,
    alpha: float = 0.01,
) -> dict[str, dict[int, CellStats]]:
    """Pool per-speech measurements into ``{group: {year: CellStats}}``.

    Counts are summed *before* the rate is computed — never an average of
    per-speech rates.  This matters: a one-sentence interjection carries the
    same weight as a twenty-minute speech under averaging, and the resulting
    series is noise.  (:func:`..drift.frame_trajectories` pools the same way,
    for the same reason.)

    The background for the prior and the z-score is the **pooled all-group
    total for that year** — so a party's tone is scored against what everyone
    else said that year, not against a fixed corpus-wide constant.

    Args:
        df:        Per-speech rows carrying *hit_col*, *n_col*, *group_col*,
            *year_col* (one row per speech; the output of a ``measure_fn``).
        hit_col:   Column of per-speech marker hits.
        n_col:     Column of per-speech denominators.
        group_col: Column to break out by (default: party).
        year_col:  Integer year column.
        per:       Rate scale (default per 1 000).
        alpha:     Prior scale (see :func:`smoothed_rate`).

    Returns:
        ``{group: {year: CellStats}}``, groups and years sorted.
    """
    if df.empty:
        return {}

    pooled = (
        df.groupby([group_col, year_col], observed=True)
        .agg(hits=(hit_col, "sum"), n=(n_col, "sum"), speeches=(hit_col, "size"))
        .reset_index()
    )
    background = (
        pooled.groupby(year_col, observed=True)
        .agg(bg_hits=("hits", "sum"), bg_n=("n", "sum"))
        .reset_index()
    )
    pooled = pooled.merge(background, on=year_col, how="left")

    out: dict[str, dict[int, CellStats]] = {}
    for row in pooled.itertuples(index=False):
        group = str(getattr(row, group_col))
        year = int(getattr(row, year_col))
        hits, n = int(row.hits), int(row.n)
        bg_hits, bg_n = int(row.bg_hits), int(row.bg_n)

        out.setdefault(group, {})[year] = CellStats(
            hits=hits,
            n=n,
            speeches=int(row.speeches),
            rate=raw_rate(hits, n, per),
            smoothed=smoothed_rate(hits, n, bg_hits, bg_n, alpha=alpha, per=per),
            z=fightin_z(hits, n, bg_hits, bg_n, alpha=alpha),
        )
    return {g: dict(sorted(years.items())) for g, years in sorted(out.items())}


def suppress_thin_cells(
    cells: dict[str, dict[int, CellStats]],
    min_speeches: int = 0,
    min_n: int = 0,
) -> dict[str, dict[int, CellStats]]:
    """Drop cells too thin to plot honestly.

    Emitting nothing is the right call for a cell with no real evidence behind
    it.  A smoothed value would still render as a dot on the chart, and a dot
    reads as a measurement — the prior would be doing the talking, not the
    data.  A gap in the line is the honest rendering of "we don't know".

    Two independent floors, because dimensions fail differently: a lexical
    dimension is thin when there are too few *speeches*; LIX is thin when there
    are too few *sentences*, which a speech count does not capture (eight
    one-line interjections are eight speeches and almost no text).

    Args:
        cells:        ``{group: {year: CellStats}}``.
        min_speeches: Minimum speeches in the cell.
        min_n:        Minimum denominator (tokens/sentences) in the cell.

    Returns:
        The same structure with thin cells (and any group left empty) removed.
    """
    kept: dict[str, dict[int, CellStats]] = {}
    for group, years in cells.items():
        surviving = {
            year: cell
            for year, cell in years.items()
            if cell.speeches >= min_speeches and cell.n >= min_n
        }
        if surviving:
            kept[group] = surviving
    return kept


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------

#: What a receipt can honestly claim.
#:
#: ``evidentiary`` — a pattern fired on an exact span of text.  The receipt
#: *proves* the hit: here is the sentence, here is the phrase, here is the
#: protocol it came from.  Every lexical and structural dimension is this.
#:
#: ``illustrative`` — there is no discrete hit to point at.  LIX is arithmetic
#: over a whole speech; no single sentence "is" the readability score.  The
#: receipt is a representative sentence, shown so a reader can check the
#: arithmetic by hand, and it must never be labelled as evidence.  Conflating
#: the two would be exactly the sleight-of-hand this feature exists to avoid.
ReceiptKind = Literal["evidentiary", "illustrative"]


@dataclass(frozen=True)
class Marker:
    """One receipt: a quotable, checkable instance behind an aggregate number."""

    dimension: str
    kind: ReceiptKind
    speech_id: str
    protocol_id: str
    party: str
    year: int
    context: str
    hl: tuple[int, int] | None
    pattern: str
    subtype: str | None = None
    speaker: str | None = None
    date: str | None = None
    file_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready form for ``tone_receipts.json``."""
        return {
            "dimension": self.dimension,
            "kind": self.kind,
            "speech_id": self.speech_id,
            "protocol_id": self.protocol_id,
            "party": self.party,
            "year": self.year,
            "context": self.context,
            "hl": list(self.hl) if self.hl else None,
            "pattern": self.pattern,
            "subtype": self.subtype,
            "speaker": self.speaker,
            "date": self.date,
            "file_url": self.file_url,
        }


def context_for_span(
    text: str, start: int, end: int, window: int = 1
) -> tuple[str, tuple[int, int]]:
    """Quote the sentence containing ``[start, end)``, plus *window* on each side.

    Args:
        text:   The full speech.
        start:  Match start offset in *text*.
        end:    Match end offset in *text*.
        window: How many neighbouring sentences to include for context.

    Returns:
        ``(context, (hl_start, hl_end))`` where the highlight offsets index the
        returned *context* string, ready to wrap in a ``<mark>``.
    """
    spans = sentence_spans(text)
    if not spans:
        snippet = text[start:end]
        return snippet, (0, len(snippet))

    index = next(
        (i for i, (s, e) in enumerate(spans) if s <= start < e),
        # A match straddling a boundary (or landing in trimmed whitespace):
        # fall back to the last sentence that starts at or before it.
        max((i for i, (s, _) in enumerate(spans) if s <= start), default=0),
    )
    first = max(0, index - window)
    last = min(len(spans) - 1, index + window)

    context_start = spans[first][0]
    context_end = spans[last][1]
    context = text[context_start:context_end]
    return context, (start - context_start, end - context_start)


def sample_receipts(
    df: pd.DataFrame,
    dimension: str,
    kind: ReceiptKind = "evidentiary",
    spans_col: str = "spans",
    per_cell: int = 3,
    seed: int = 13,
    group_col: str = "party",
    year_col: str = "year",
    text_col: str = "text",
    subtype_of: dict[str, str] | None = None,
) -> dict[str, dict[int, list[Marker]]]:
    """Deterministically sample receipts for each (group, year) cell.

    The sample is **random under a fixed seed**, never "the most extreme
    examples".  That choice is the difference between evidence and curated
    outrage: hand-picking the ugliest quote from each cell would produce a page
    that is technically sourced and completely misleading.  Random-with-a-seed
    is reproducible (anyone re-running the precompute gets the same receipts)
    and defensible (nobody chose them).

    Args:
        df:         Per-speech rows with *spans_col*, *group_col*, *year_col*,
            *text_col* — plus, where available, ``id``, ``protocol_id``,
            ``speaker``, ``protocol_date``, ``file_url`` for the citation.
        dimension:  Dimension id, stamped onto each marker.
        kind:       ``evidentiary`` or ``illustrative`` (see :data:`ReceiptKind`).
        spans_col:  Column of ``[(start, end, pattern), ...]`` per speech.
        per_cell:   Receipts to keep per (group, year).
        seed:       RNG seed.  Pinned in the tests: changing it changes which
            receipts the public site shows, which is not a silent refactor.
        group_col:  Group column (default: party).
        year_col:   Integer year column.
        text_col:   Column holding the speech text.
        subtype_of: Optional ``{pattern: subtype}`` map (e.g. a vi/dom
            pattern's ``outgroup_type``), stamped onto the marker.

    Returns:
        ``{group: {year: [Marker, ...]}}``.
    """
    if df.empty or spans_col not in df.columns:
        return {}

    candidates: dict[tuple[str, int], list[Marker]] = {}
    for row in df.itertuples(index=False):
        spans = getattr(row, spans_col, None)
        if not spans:
            continue
        text = getattr(row, text_col, "") or ""
        group = str(getattr(row, group_col))
        year = int(getattr(row, year_col))

        for start, end, pattern in spans:
            context, hl = context_for_span(text, start, end)
            candidates.setdefault((group, year), []).append(
                Marker(
                    dimension=dimension,
                    kind=kind,
                    speech_id=str(getattr(row, "id", "")),
                    protocol_id=str(getattr(row, "protocol_id", "")),
                    party=group,
                    year=year,
                    context=context,
                    hl=hl if kind == "evidentiary" else None,
                    pattern=pattern,
                    subtype=(subtype_of or {}).get(pattern),
                    speaker=_optional_str(getattr(row, "speaker", None)),
                    date=_optional_str(getattr(row, "protocol_date", None)),
                    file_url=_optional_str(getattr(row, "file_url", None)),
                )
            )

    out: dict[str, dict[int, list[Marker]]] = {}
    for (group, year), markers in candidates.items():
        # Seed per cell, not once globally: the receipts for one cell then do
        # not shift because an unrelated year gained a speech in the next ETL
        # run.  Sort first so dict/groupby ordering can't leak into the sample.
        markers.sort(key=lambda m: (m.speech_id, m.hl or (0, 0), m.pattern))
        rng = random.Random(f"{seed}:{dimension}:{group}:{year}")
        chosen = markers if len(markers) <= per_cell else rng.sample(markers, per_cell)
        out.setdefault(group, {})[year] = chosen
    return {g: dict(sorted(years.items())) for g, years in sorted(out.items())}


def _optional_str(value: object) -> str | None:
    """Coerce a possibly-missing DataFrame value to ``str`` or ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


# --------------------------------------------------------------------------
# Dimension registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionSpec:
    """Everything the precompute and the validation script need about a dimension.

    Adding a twelfth dimension means writing one ``measure_fn`` and one of
    these — no changes to ``build_site_data.py`` or the validator, which both
    iterate the registry.
    """

    id: str
    label_sv: str
    unit_sv: str
    technique: Literal["lexical", "stylometric", "structural"]
    #: ``df -> df`` with ``hits``, ``n`` and (evidentiary dimensions) ``spans``.
    measure_fn: Callable[[pd.DataFrame], pd.DataFrame]
    receipt_kind: ReceiptKind = "evidentiary"
    status: Literal["launch", "phase2"] = "phase2"
    supports_frames: bool = False
    min_cell_speeches: int = 8
    min_cell_n: int = 0
    per: float = 1000.0
    alpha: float = 0.01
    pattern_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        # A lexical dimension with no pattern table would score zero forever;
        # a stylometric one with a pattern table is a category error.  Both are
        # the kind of mistake that only surfaces as a flat line on the live
        # site, so reject them at construction.
        if self.technique == "lexical" and not self.pattern_paths:
            raise ValueError(f"Lexical dimension {self.id!r} has no pattern_paths.")
        if self.technique == "stylometric" and self.receipt_kind != "illustrative":
            raise ValueError(
                f"Stylometric dimension {self.id!r} must use illustrative receipts: "
                "there is no discrete hit to point at, so a receipt cannot be evidence."
            )


#: Populated by the dimension modules (``vi_dom``, ``readability``, ``inclusive``, …).
#: Empty here by design — the kernel knows nothing about any specific dimension.
TONE_DIMENSIONS: dict[str, DimensionSpec] = {}


def register(spec: DimensionSpec) -> DimensionSpec:
    """Add *spec* to :data:`TONE_DIMENSIONS`, rejecting a duplicate id."""
    if spec.id in TONE_DIMENSIONS:
        raise ValueError(f"Dimension {spec.id!r} is already registered.")
    TONE_DIMENSIONS[spec.id] = spec
    return spec
