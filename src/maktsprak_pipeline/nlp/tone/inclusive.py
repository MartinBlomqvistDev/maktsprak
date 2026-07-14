"""Inclusive language: ``hen``, and the chamber's own job titles.

The anchor dimension.  Everything else in the tone set can be read as a claim
about who argues badly; this one cannot be read that way by anybody.  It tracks
a documented change in the Swedish language and asks, mechanically, which
parties' speech moved with it.  Having it on the page is what makes the feature
legible as *"how political language changes"* rather than *"who is the
demagogue"* — and that reframing is not decoration, it is the point.

Two sub-measures, reported separately because the data behaves completely
differently:

**``hen``** (:func:`measure_hen`) — a single closed-class pronoun, not a curated
list, so there is no word-choice to defend.  The corpus timeline matches the
published sociolinguistics almost exactly: near-zero before 2012, a sharp rise
in 2013-14 (the year ``hen`` entered SAOL, the Swedish Academy's wordlist), then
sustained use.  Gustafsson Sendén, Bäck & Lindqvist (2015) and Lindqvist,
Renström & Gustafsson Sendén (2021) document both the adoption curve and an
attitude gradient across the political spectrum — so if the corpus shows one
too, that is a *replication*, not an accusation.

**Occupational pairs** (:func:`occupational_report`) — the ``-man`` → ``-person``
/ ``-ledamot`` substitutions.  Deliberately measured on the terms the Riksdag
actually uses about itself (``riksdagsman`` → ``riksdagsledamot``, ``talesman``
→ ``talesperson``, ``tjänsteman`` → ``tjänsteperson``), because an audit of the
obvious textbook pairs found they simply do not occur in political debate:
``brandperson``, ``flygvärd`` and ``ombudsperson`` have **zero** occurrences in
the whole 2002-2026 corpus.  That null result is reported, not buried — and
``sjuksköterska`` (for which ISOF records no settled neutral alternative) and
``brandman`` are kept in the table precisely *as* null cases, to show absence of
change where none is claimed.

The measure is a **substitution ratio**, ``neutral / (neutral + gendered)``, not
a raw count: it asks which form a speaker chose *given that they raised the
topic at all*, which controls for how much any party happens to talk about
firefighters or civil servants.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ...config import PROJECT_ROOT
from .kernel import (
    SENT_COL,
    DimensionSpec,
    compile_alternation,
    compile_pattern,
    ensure_sentence_spans,
    find_spans,
    load_pattern_table,
    register,
    sentences_with_a_hit,
)

PAIRS_PATH = PROJECT_ROOT / "data" / "lexicons" / "inclusive_occupational_pairs.csv"

#: ``hen`` and its genitive.  Not a curated list — a closed-class pronoun.
#: ``hen`` must never match inside ``Henrik``; the kernel's word-boundary
#: patterns are what guarantee that, and it is regression-tested.
HEN_FORMS: tuple[str, ...] = ("hen", "hens")

#: 395 hits in 7.3 million sentences: a per-1 000 rate would round to zero on
#: every chart. The unit is honest about how rare the word still is.
HEN_PER = 100_000.0


@lru_cache(maxsize=1)
def _hen_table() -> pd.DataFrame:
    table = pd.DataFrame({"pattern": list(HEN_FORMS)})
    table["regex"] = [compile_pattern(p) for p in table["pattern"]]
    table.attrs["alternation"] = compile_alternation(list(HEN_FORMS))
    return table


def measure_hen(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Per-speech count of sentences using ``hen``/``hens``."""
    out = ensure_sentence_spans(df, text_col=text_col).copy()
    table = _hen_table()
    spans = [find_spans(text, table, "pattern") for text in out[text_col]]
    out["spans"] = spans
    out["n"] = [len(s) for s in out[SENT_COL]]
    out["hits"] = [
        sentences_with_a_hit(sent, sp) for sent, sp in zip(out[SENT_COL], spans, strict=True)
    ]
    return out


# ------------------------------------------------------- occupational pairs


@lru_cache(maxsize=1)
def load_pairs() -> pd.DataFrame:
    """The gendered/neutral occupational pairs, validated."""
    return load_pattern_table(
        PAIRS_PATH,
        key_col="gendered",
        required_cols=["gendered", "neutral", "status", "source", "note"],
    )


def occupational_report(
    df: pd.DataFrame,
    text_col: str = "text",
    group_col: str = "party",
    year_col: str = "year",
    min_pair_mentions: int = 5,
) -> dict[str, object]:
    """Substitution ratios per pair — corpus-wide, by year, and by party.

    A report rather than a registered chart dimension: the audit showed most
    neutral forms are vanishingly rare in chamber debate, so a party-by-year
    line would be almost entirely suppressed gaps.  Publishing the honest table
    (including the pairs where nothing happened) is worth more than a chart of
    absences.

    Args:
        df:                Speeches with text, party and year.
        min_pair_mentions: A (pair, group, year) cell needs this many total
            mentions before its ratio is meaningful.

    Returns:
        ``{"pairs": [...], "by_year": {...}, "by_party": {...}}``.
    """
    pairs = load_pairs()
    prepared = df

    counts: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        gendered_rx = compile_pattern(row.gendered)
        neutral = str(row.neutral).strip().lower()
        neutral_rx = compile_pattern(neutral) if neutral and neutral != "nan" else None

        gendered_hits: list[int] = []
        neutral_hits: list[int] = []
        for text in prepared[text_col]:
            if not isinstance(text, str):
                gendered_hits.append(0)
                neutral_hits.append(0)
                continue
            gendered_hits.append(len(gendered_rx.findall(text)))
            neutral_hits.append(len(neutral_rx.findall(text)) if neutral_rx else 0)

        frame = pd.DataFrame(
            {
                "party": prepared[group_col].values,
                "year": prepared[year_col].values,
                "gendered": gendered_hits,
                "neutral": neutral_hits,
            }
        )
        total_g = int(frame["gendered"].sum())
        total_n = int(frame["neutral"].sum())

        counts.append(
            {
                "gendered": row.gendered,
                "neutral": neutral or None,
                "status": row.status,
                "note": row.note,
                "gendered_hits": total_g,
                "neutral_hits": total_n,
                "ratio": _ratio(total_n, total_g),
                "by_year": {
                    int(y): {
                        "gendered": int(sub["gendered"].sum()),
                        "neutral": int(sub["neutral"].sum()),
                        "ratio": _ratio(int(sub["neutral"].sum()), int(sub["gendered"].sum())),
                    }
                    for y, sub in frame.groupby("year", observed=True)
                    if int(sub["gendered"].sum()) + int(sub["neutral"].sum()) >= min_pair_mentions
                },
                "by_party": {
                    str(p): {
                        "gendered": int(sub["gendered"].sum()),
                        "neutral": int(sub["neutral"].sum()),
                        "ratio": _ratio(int(sub["neutral"].sum()), int(sub["gendered"].sum())),
                    }
                    for p, sub in frame.groupby("party", observed=True)
                    if int(sub["gendered"].sum()) + int(sub["neutral"].sum()) >= min_pair_mentions
                },
            }
        )

    return {"pairs": counts, "min_pair_mentions": min_pair_mentions}


def _ratio(neutral: int, gendered: int) -> float | None:
    """``neutral / (neutral + gendered)``, or ``None`` when the term never appears."""
    total = neutral + gendered
    return None if total == 0 else round(neutral / total, 4)


HEN = register(
    DimensionSpec(
        id="hen",
        label_sv="Användning av »hen«",
        unit_sv="meningar per 100 000 som använder hen/hens",
        technique="structural",
        measure_fn=measure_hen,
        status="launch",
        supports_frames=False,  # 395 hits corpus-wide; a frame split would be noise
        min_cell_speeches=8,
        per=HEN_PER,
        pattern_paths=[],  # a closed-class pronoun, not a curated list
    )
)
