"""Vi-mot-dom: what survived the audit, and what the audit killed.

Three rounds of reading *real matches*, not reasoning about patterns, cut this
module down to what the corpus actually supports.  The cuts are the substance
here, so they are documented rather than quietly reverted:

**Round 1, referring expressions are not othering.**  ``dessa människor``
(3 356 hits) and ``de här människorna`` (1 262) turned out to be overwhelmingly
*sympathetic*: "Vi har all anledning att stötta de här människorna" (S), "Vi
måste kunna möta de här människorna med behandling" (V).  ``de som kommer hit``
fails the same way: "De som kommer hit ska ges möjlighet att vara med och bygga
landet" (V).  Bare ``eliten`` is a homograph, "den internationella eliten" (L)
means world-class researchers.  **In the formal chamber register hostility lives
in the predicate, not the noun phrase.**

**Round 2, two more dimensions failed their own gate.**

- ``folk`` (11 848 hits) was specced as *people-centrism* in Mudde's sense: the
  people as one virtuous, undivided body set against a corrupt elite.  The
  sample says otherwise, "skogen har en speciell plats i hjärtat hos svenska
  folket" (V), "svenska folket tror mer på alliansregeringen" (C).  That is the
  electorate, referred to.  Charting it as people-centrism would repeat exactly
  the Round-1 mistake with a longer word list.
- ``antielit`` (254 hits) is precise enough, but **median 2 hits per non-empty
  party-year cell and 91 of 192 cells empty**.  That is not a time series; it is
  noise with a line through it.  The speech-count suppression floor cannot catch
  this because the speeches are there, the *hits* are not.

Both remain as the **halves of the construction detector** (a census needs an
in-group half and an out-group half), and neither is registered as a chart.

**What ships.**

- :func:`measure_klasskonflikt`, an out-group defined by wealth.  3 165 hits,
  real variation between parties, and it exists so the instrument can find the
  same rhetorical move aimed leftward.  It does: V is the top user by a distance.
  Homographs are excluded rather than the patterns thrown away ("de rikaste
  **länderna**" is about countries, the audit caught "Sverige är inte längre ett
  av de rikaste länderna" (M) counted as class conflict).
- :func:`vi_dom_census`, the **construction**: an in-group marker and an
  out-group marker in the *same sentence*.  This is the real "us versus them",
  and it is genuinely rare, ~51 sentences in 80 541 speeches.  Too few to chart,
  so it is published as an **exhaustive census**: every single one, not a sample,
  which is stronger than any statistic because there was nothing to cherry-pick.
  Its distribution is the project's most striking result, mostly *economic*,
  barely any origin-based, led by S and V rather than SD.  A single
  undifferentiated "us-vs-them" number would have said the opposite, loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ...config import PROJECT_ROOT
from .kernel import (
    SENT_COL,
    DimensionSpec,
    compile_alternation,
    ensure_sentence_spans,
    find_spans,
    load_pattern_table,
    register,
    sentence_index,
    sentences_with_a_hit,
)

PATTERN_PATH = PROJECT_ROOT / "data" / "lexicons" / "vi_dom_patterns.csv"

_REQUIRED_COLS = ["pattern", "slot", "subtype", "source", "exclude_next", "note"]


@lru_cache(maxsize=1)
def load_patterns() -> pd.DataFrame:
    """The vi/dom pattern table, validated and compiled (cached)."""
    return load_pattern_table(PATTERN_PATH, key_col="pattern", required_cols=_REQUIRED_COLS)


def _subset(slot: str, subtype: str | None = None) -> pd.DataFrame:
    table = load_patterns()
    rows = table[table["slot"] == slot]
    if subtype is not None:
        rows = rows[rows["subtype"] == subtype]
    rows = rows.reset_index(drop=True)
    # A slice does not inherit .attrs, and find_spans reads both off it.
    rows.attrs["alternation"] = compile_alternation(rows["pattern"].tolist())
    rows.attrs["excludes"] = {
        p: rx for p, rx in table.attrs.get("excludes", {}).items() if p in set(rows["pattern"])
    }
    return rows


def subtype_map() -> dict[str, str]:
    """``{pattern: subtype}`` for stamping receipts."""
    table = load_patterns()
    return dict(zip(table["pattern"], table["subtype"], strict=True))


def _measure(df: pd.DataFrame, table: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Per-speech ``hits`` (matching sentences), ``n`` (sentences), ``spans``."""
    out = ensure_sentence_spans(df, text_col=text_col).copy()
    spans = [
        find_spans(text, table, "pattern", exclude_col="exclude_next") for text in out[text_col]
    ]
    out["spans"] = spans
    out["n"] = [len(s) for s in out[SENT_COL]]
    out["hits"] = [
        sentences_with_a_hit(sent, sp) for sent, sp in zip(out[SENT_COL], spans, strict=True)
    ]
    return out


def measure_folk(df: pd.DataFrame) -> pd.DataFrame:
    """In-group ("folket") sentences per speech.

    **Not a registered chart dimension.**  The audit found these are usually
    just the electorate, referred to, not people-centrism in Mudde's sense (see
    the module docstring).  Kept because the census needs an in-group half.
    """
    return _measure(df, _subset("ingroup"))


def measure_antielit(df: pd.DataFrame) -> pd.DataFrame:
    """Anti-elite / anti-establishment sentences per speech.

    **Not a registered chart dimension.**  Precise, but far too sparse to plot:
    median 2 hits per non-empty party-year cell, 91 of 192 cells empty.  Kept
    because the census needs an out-group half.
    """
    return _measure(df, _subset("outgroup", "elite"))


def measure_klasskonflikt(df: pd.DataFrame) -> pd.DataFrame:
    """Sentences framing a wealth-defined out-group, per speech."""
    return _measure(df, _subset("outgroup", "economic"))


# ---------------------------------------------------------------- the census


@dataclass(frozen=True)
class Construction:
    """One sentence that actually pits an in-group against an out-group."""

    party: str
    year: int
    protocol_id: str
    speaker: str
    sentence: str
    ingroup: list[str]
    outgroup: list[str]
    subtypes: list[str]
    file_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "party": self.party,
            "year": self.year,
            "protocol_id": self.protocol_id,
            "speaker": self.speaker,
            "sentence": self.sentence,
            "ingroup": self.ingroup,
            "outgroup": self.outgroup,
            "subtypes": self.subtypes,
            "file_url": self.file_url,
        }


def vi_dom_census(df: pd.DataFrame, text_col: str = "text") -> list[Construction]:
    """Every sentence containing both an in-group and an out-group marker.

    Not a sample, the complete set.  With 51 hits across the whole 2002-2026
    corpus, an exhaustive census is both possible and far stronger than any
    statistic: a reader can check *all* of the evidence, and nobody can argue
    the examples were cherry-picked, because there was nothing to pick from.

    Args:
        df:       Speeches with ``party``, ``year``, ``text``.
        text_col: Column holding the speech text.

    Returns:
        Every construction found, in corpus order.
    """
    ingroup = _subset("ingroup")
    outgroup = _subset("outgroup")
    subtypes = subtype_map()
    prepared = ensure_sentence_spans(df, text_col=text_col)

    found: list[Construction] = []
    for row in prepared.itertuples(index=False):
        text = getattr(row, text_col)
        if not isinstance(text, str):
            continue
        sent_spans = getattr(row, SENT_COL)

        in_spans = find_spans(text, ingroup, "pattern")
        if not in_spans:
            continue
        out_spans = find_spans(text, outgroup, "pattern")
        if not out_spans:
            continue

        by_sentence: dict[int, tuple[set[str], set[str]]] = {}
        for start, _, pattern in in_spans:
            index = sentence_index(sent_spans, start)
            if index is not None:
                by_sentence.setdefault(index, (set(), set()))[0].add(pattern)
        for start, _, pattern in out_spans:
            index = sentence_index(sent_spans, start)
            if index is not None:
                by_sentence.setdefault(index, (set(), set()))[1].add(pattern)

        for index, (ins, outs) in sorted(by_sentence.items()):
            if not (ins and outs):
                continue
            start, end = sent_spans[index]
            found.append(
                Construction(
                    party=str(getattr(row, "party", "")),
                    year=int(getattr(row, "year", 0)),
                    protocol_id=str(getattr(row, "protocol_id", "")),
                    speaker=str(getattr(row, "speaker", "")),
                    sentence=" ".join(text[start:end].split()),
                    ingroup=sorted(ins),
                    outgroup=sorted(outs),
                    subtypes=sorted({subtypes[p] for p in outs}),
                    file_url=_url(getattr(row, "file_url", None)),
                )
            )
    return found


def _url(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


# ------------------------------------------------------------------ registry
#
# `folk` and `antielit` are deliberately NOT registered, they failed their own
# validation gate (see the module docstring: one measures the electorate rather
# than people-centrism, the other is too sparse to plot).  They stay as measure
# functions because the census consumes them.  Re-registering either without
# re-running scripts/validate_tone.py would put a known-bad series on the site.

KLASSKONFLIKT = register(
    DimensionSpec(
        id="klasskonflikt",
        label_sv="Klasskonflikt-ramning",
        unit_sv="meningar per 1 000 som ramar in en förmögen motpart",
        technique="lexical",
        measure_fn=measure_klasskonflikt,
        status="launch",
        supports_frames=True,
        min_cell_speeches=8,
        pattern_paths=[PATTERN_PATH],
    )
)
