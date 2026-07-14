"""Vi-mot-dom: people-centrism, anti-elitism, class conflict — and the construction.

What an audit of the real corpus forced this module to become
---------------------------------------------------------------
The first design counted "generalised out-group references" — ``dessa
människor``, ``de som kommer hit`` — and treated a hit as evidence of othering.
Sampling the actual matches killed that idea outright:

    "Vi har all anledning att stötta de här människorna."            (S)
    "De som kommer hit ska ges möjlighet att vara med och bygga landet." (V)
    "Vi måste kunna möta de här människorna med behandling."         (V)

Every one of those is *sympathetic*, and every one would have counted as
othering.  In the formal chamber register the hostility is not in the noun
phrase — it is in the predicate around it.  A word list over referring
expressions measures "talking about a group of people", which is what a welfare
debate consists of.  ``dessa människor`` and ``de här människorna`` (4 618 hits
between them) were cut for precisely this reason, and bare ``eliten`` was cut
because it also means a *top tier*: "den internationella eliten" in an L speech
is about world-class researchers.

What survives is measured three ways, none of which requires judging whether a
speaker was being hostile:

1. :func:`measure_folk` — **folkhänvisningar** (people-centrism).  How often a
   party invokes the people as one undivided body ("svenska folket", "vanligt
   folk").  Countable, valence-free, and one of the two constitutive components
   of populism in the academic definition (Mudde 2004; Rooduijn & Pauwels 2011).
2. :func:`measure_antielit` — **anti-elit-språk** (anti-elitism).  The other
   constitutive component.  Qualified elite terms only, for the homograph
   reason above.
3. :func:`measure_klasskonflikt` — **klasskonflikt-ramning**.  An out-group
   defined by wealth ("de rika", "miljardärerna").  Structurally the same
   rhetorical move as an origin-defined out-group, aimed at a different target —
   which is exactly why it is measured on the same axis rather than quietly
   omitted.

4. :func:`vi_dom_census` — the **construction** itself: an in-group marker and
   an out-group marker in the *same sentence*.  This is the real "us versus
   them", and it is genuinely rare — 51 sentences in 80 541 speeches.  Too few
   for a time series, which is why it is published as an **exhaustive census**
   (every single one, not a sample) rather than a chart.  The census is also the
   most striking result the project has: of those 51, roughly two thirds are
   *economic* and only a handful are origin-based, and the parties using the
   construction most are S and V, not SD.  A single undifferentiated "us-vs-them"
   number would have said the opposite, loudly and wrongly.
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

_REQUIRED_COLS = ["pattern", "slot", "subtype", "source", "note"]


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
    # find_spans reads the alternation off .attrs; a slice does not inherit it.
    rows.attrs["alternation"] = compile_alternation(rows["pattern"].tolist())
    return rows


def subtype_map() -> dict[str, str]:
    """``{pattern: subtype}`` for stamping receipts."""
    table = load_patterns()
    return dict(zip(table["pattern"], table["subtype"], strict=True))


def _measure(df: pd.DataFrame, table: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Per-speech ``hits`` (matching sentences), ``n`` (sentences), ``spans``."""
    out = ensure_sentence_spans(df, text_col=text_col).copy()
    spans = [find_spans(text, table, "pattern") for text in out[text_col]]
    out["spans"] = spans
    out["n"] = [len(s) for s in out[SENT_COL]]
    out["hits"] = [
        sentences_with_a_hit(sent, sp) for sent, sp in zip(out[SENT_COL], spans, strict=True)
    ]
    return out


def measure_folk(df: pd.DataFrame) -> pd.DataFrame:
    """Sentences invoking the people as one body, per speech."""
    return _measure(df, _subset("ingroup"))


def measure_antielit(df: pd.DataFrame) -> pd.DataFrame:
    """Sentences using anti-elite / anti-establishment vocabulary, per speech."""
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

    Not a sample — the complete set.  With 51 hits across the whole 2002-2026
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

FOLK = register(
    DimensionSpec(
        id="folk",
        label_sv="Folkhänvisningar",
        unit_sv="meningar per 1 000 som åberopar »folket«",
        technique="lexical",
        measure_fn=measure_folk,
        status="launch",
        supports_frames=True,
        min_cell_speeches=8,
        pattern_paths=[PATTERN_PATH],
    )
)

ANTIELIT = register(
    DimensionSpec(
        id="antielit",
        label_sv="Anti-elit-språk",
        unit_sv="meningar per 1 000 med elit-/etablissemangsretorik",
        technique="lexical",
        measure_fn=measure_antielit,
        status="launch",
        supports_frames=True,
        min_cell_speeches=8,
        pattern_paths=[PATTERN_PATH],
    )
)

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
