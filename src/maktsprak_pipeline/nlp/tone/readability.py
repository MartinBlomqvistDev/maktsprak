"""LIX — how plainly a party speaks.  No word list, no judgement calls.

Every other tone dimension rests on a curated list of words, and a curated list
is always answerable to "who chose those words, and why those?".  This one is
not: it counts words, sentences, and letters.  There is nothing to audit for
bias because there is nothing to bias.  That makes it the most defensible thing
on the page, and it is worth saying so plainly in the methodology.

The measure is **LIX** (läsbarhetsindex, Björnsson 1968), the standard Swedish
readability formula — deliberately *not* Flesch-Kincaid, which is calibrated on
English syllable structure and returns meaningless numbers on Swedish::

    LIX = O/M + (L × 100 / O)

    O = words
    M = sentences
    L = long words (more than 6 letters)

Roughly: <30 very easy, 30-40 easy (fiction), 40-50 normal (newspaper prose),
50-60 difficult (officialese), >60 very difficult (legal/academic).  Chamber
debate sits high; what is interesting is not the absolute level but whether a
party has moved.

Two things that would quietly corrupt the number, and how they are handled:

**Aggregation is pooled, never averaged.**  A cell's LIX is computed from the
summed O, M and L of every speech in it — not by averaging per-speech LIX
values.  A one-line interjection ("Ja, herr talman!") has a per-speech LIX that
is pure noise, and averaging lets it shout as loudly as a twenty-minute speech.

**Very short speeches are excluded entirely.**  A speech needs at least 3
sentences and 20 words to enter the pool.  The chamber is full of one-word
objections, and they say nothing about how a party *speaks*.  The excluded
share is published rather than hidden.

Cross-language comparison is LIX's known weak point — Swedish compounds
(``arbetslöshetsersättning``) inflate the long-word count relative to English.
That is not a problem here: every comparison is Swedish against Swedish, party
against party, year against year.  Stated in the methodology so it does not have
to be discovered as a gotcha.
"""

from __future__ import annotations

import re

import pandas as pd

from .kernel import SENT_COL, CellStats, DimensionSpec, ensure_sentence_spans, register

#: A "word" for LIX purposes: a run of letters (Swedish included) or digits.
_WORD_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

#: Björnsson's threshold: a word is "long" if it has more than six letters.
LONG_WORD_MIN_LENGTH = 7

#: A speech below either floor is excluded from the pool (see module docstring).
MIN_SENTENCES = 3
MIN_WORDS = 20

#: Below this many pooled sentences a cell's LIX is not stable enough to plot.
#: Speech count is the wrong guard here — eight one-line interjections are eight
#: speeches and almost no text.
MIN_CELL_SENTENCES = 50


def lix(words: int, sentences: int, long_words: int) -> float:
    """LIX from raw counts.  ``0.0`` if there is nothing to measure."""
    if words <= 0 or sentences <= 0:
        return 0.0
    return words / sentences + (long_words * 100.0 / words)


def measure_lix(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Per-speech word / sentence / long-word counts, plus the inclusion flag.

    ``hits`` and ``n`` are set to ``long_words`` and ``words`` so the shared
    suppression and tooltip machinery keeps working (``long_words <= words``
    always holds).  The headline number is not a rate, though — it is the index,
    computed in :func:`aggregate_lix`.
    """
    out = ensure_sentence_spans(df, text_col=text_col).copy()

    words: list[int] = []
    long_words: list[int] = []
    for text in out[text_col]:
        tokens = _WORD_RE.findall(text) if isinstance(text, str) else []
        words.append(len(tokens))
        long_words.append(sum(1 for t in tokens if len(t) >= LONG_WORD_MIN_LENGTH))

    out["words"] = words
    out["long_words"] = long_words
    out["sentences"] = [len(s) for s in out[SENT_COL]]
    out["included"] = (out["sentences"] >= MIN_SENTENCES) & (out["words"] >= MIN_WORDS)

    # Kernel-compatible columns (long_words <= words, so hits <= n holds).
    out["hits"] = out["long_words"]
    out["n"] = out["words"]
    return out


def aggregate_lix(
    df: pd.DataFrame,
    group_col: str = "party",
    year_col: str = "year",
    alpha: float = 0.01,
) -> dict[str, dict[int, CellStats]]:
    """Pool counts per (group, year), then compute one LIX from the totals.

    LIX is a composite of two ratios, so it is smoothed by smoothing each ratio
    toward the pooled all-party value for that year — the same informative-prior
    idea the rest of the kernel uses, applied twice::

        smoothed_LIX = shrink(words/sentences) + 100 * shrink(long_words/words)

    ``z`` is ``None`` by design: the Fightin' Words z-score is a statement about
    a *proportion* being over-represented, and LIX is not a proportion.  Emitting
    a number there would be inventing a statistic that does not exist.
    """
    pool = df[df["included"]] if "included" in df.columns else df
    if pool.empty:
        return {}

    cells = (
        pool.groupby([group_col, year_col], observed=True)
        .agg(
            words=("words", "sum"),
            sentences=("sentences", "sum"),
            long_words=("long_words", "sum"),
            speeches=("words", "size"),
        )
        .reset_index()
    )
    background = (
        cells.groupby(year_col, observed=True)
        .agg(bg_words=("words", "sum"), bg_sent=("sentences", "sum"), bg_long=("long_words", "sum"))
        .reset_index()
    )
    cells = cells.merge(background, on=year_col, how="left")

    out: dict[str, dict[int, CellStats]] = {}
    for row in cells.itertuples(index=False):
        words, sentences, long_words = int(row.words), int(row.sentences), int(row.long_words)
        if sentences < MIN_CELL_SENTENCES:
            continue

        prior_words = alpha * int(row.bg_words)
        prior_sent = alpha * int(row.bg_sent)
        prior_long = alpha * int(row.bg_long)

        smoothed_length = (words + prior_words) / (sentences + prior_sent)
        smoothed_share = (long_words + prior_long) / (words + prior_words)

        out.setdefault(str(getattr(row, group_col)), {})[int(getattr(row, year_col))] = CellStats(
            hits=long_words,
            n=words,
            speeches=int(row.speeches),
            rate=lix(words, sentences, long_words),
            smoothed=smoothed_length + smoothed_share * 100.0,
            z=None,
            extra={
                "words": words,
                "sentences": sentences,
                "long_words": long_words,
                "avg_sentence_length": round(words / sentences, 2),
                "long_word_pct": round(long_words * 100.0 / words, 2),
            },
        )
    return {g: dict(sorted(years.items())) for g, years in sorted(out.items())}


def excluded_share(df: pd.DataFrame) -> float:
    """Fraction of speeches dropped by the inclusion floors.  Published, not hidden."""
    if df.empty or "included" not in df.columns:
        return 0.0
    return float((~df["included"]).mean())


LIX = register(
    DimensionSpec(
        id="lix",
        label_sv="Läsbarhet (LIX)",
        unit_sv="läsbarhetsindex (Björnsson) — högre = mer svårläst",
        technique="stylometric",
        measure_fn=measure_lix,
        aggregate_fn=aggregate_lix,
        receipt_kind="illustrative",
        status="launch",
        supports_frames=True,
        min_cell_speeches=0,  # the real floor is MIN_CELL_SENTENCES, applied in aggregate_lix
        pattern_paths=[],
    )
)
