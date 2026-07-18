"""Validation gate for the tone dimensions.  Runs before any of this reaches a chart.

Three things a tone dimension has to survive before it is allowed on the site:

**Symmetry**, does the instrument find the pattern on both sides of the
spectrum, or only where we went looking?  For a dimension with an out-group
subtype (``vi_dom``'s economic / origin / elite), the breakdown is printed per
party.  If class-conflict framing from the left shows up at comparable intensity
to origin framing from the right, it gets shown identically.  If it genuinely
does not, that is stated on the methodology page rather than quietly omitted.
Either answer is content; only a missing answer is a problem.

**Precision**, of the sentences this dimension counts, how many really are what
it claims?  Exports a seeded, party-stratified sample for hand-labelling, and the
resulting number gets published.  This is not ceremony: the first pattern list
scored well under 50% because ``dessa människor`` mostly appears in *sympathetic*
sentences ("vi har all anledning att stötta de här människorna"), and only
reading real matches caught it.

**Correctness**, for a stylometric dimension there is nothing to hand-label; the
number is arithmetic.  It gets a worked example instead, printed so the formula
can be checked by hand.  A reader must never assume LIX has a hidden precision
score: the question does not apply to it.

Usage::

    python scripts/validate_tone.py                     # all launch dimensions
    python scripts/validate_tone.py --dimension folk    # just one
    python scripts/validate_tone.py --sample 200        # bigger audit sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from scripts._corpus import load_corpus as _load_archive
from src.maktsprak_pipeline.logger import get_logger
from src.maktsprak_pipeline.nlp.tone import inclusive, readability, vi_dom
from src.maktsprak_pipeline.nlp.tone.kernel import (
    TONE_DIMENSIONS,
    aggregate_cells,
    context_for_span,
    ensure_sentence_spans,
    hit_density,
    suppress_thin_cells,
)

logger = get_logger()

CORPUS = Path("data/parquet/speeches_full.parquet")
OUT_DIR = Path("data/reports")


def load_corpus(limit: int | None = None) -> pd.DataFrame:
    """The speech corpus with an integer ``year``.

    Goes through the shared loader rather than reading the Parquet directly,
    that is where the duplicate-id guard lives, and a tone number computed on a
    double-counted corpus would be wrong in a way nothing downstream can see.
    """
    df = _load_archive()
    df["year"] = df["protocol_date"].dt.year
    df = df[df["year"].notna() & df["text"].notna()].astype({"year": int})
    if limit:
        df = df.sample(limit, random_state=42)
    logger.info(f"Corpus: {len(df):,} speeches, {df['year'].min()}-{df['year'].max()}")
    return ensure_sentence_spans(df)


def symmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Out-group subtype by party, the check that decides whether this is fair.

    A single undifferentiated "us-vs-them" count would rank V top and SD near
    the bottom, and the naive reading of that ("V is the most populist party")
    is both absurd and indefensible.  Splitting by subtype is what makes the
    result honest: V's number is class conflict, not origin framing.
    """
    subtypes = vi_dom.subtype_map()

    rows = []
    for name, subset in (
        ("elite", vi_dom.measure_antielit(df)),
        ("economic", vi_dom.measure_klasskonflikt(df)),
    ):
        for party, spans in zip(subset["party"], subset["spans"], strict=True):
            for _, _, pattern in spans:
                rows.append({"party": party, "subtype": subtypes.get(pattern, name)})

    # Origin patterns are only meaningful inside a construction (the audit found
    # them used neutrally and sympathetically on their own), so they are counted
    # from the census rather than as bare hits.
    for construction in vi_dom.vi_dom_census(df):
        for subtype in construction.subtypes:
            if subtype == "origin":
                rows.append({"party": construction.party, "subtype": "origin (construction)"})

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return pd.crosstab(frame["party"], frame["subtype"])


def precision_sample(df: pd.DataFrame, dimension: str, size: int, seed: int = 42) -> pd.DataFrame:
    """A party-stratified sample of real matches, ready for hand-labelling.

        Stratified so no party is judged on three examples while another gets fifty
    , an unbalanced audit would produce a precision number that is really just a
        statement about whichever party happened to dominate the sample.
    """
    spec = TONE_DIMENSIONS[dimension]
    measured = spec.measure_fn(df)

    rows = []
    for record in measured.itertuples(index=False):
        for start, end, pattern in getattr(record, "spans", []) or []:
            context, hl = context_for_span(record.text, start, end)
            rows.append(
                {
                    "party": record.party,
                    "year": record.year,
                    "pattern": pattern,
                    "match": context[hl[0] : hl[1]],
                    "sentence": " ".join(context.split()),
                    "protocol_id": record.protocol_id,
                    "is_correct__fill_in": "",
                }
            )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    rng = random.Random(seed)
    per_party = max(1, size // frame["party"].nunique())
    picked = [
        sub.iloc[sorted(rng.sample(range(len(sub)), min(per_party, len(sub))))]
        for _, sub in frame.groupby("party", observed=True)
    ]
    return pd.concat(picked).reset_index(drop=True)


def report_dimension(df: pd.DataFrame, dimension: str, sample_size: int) -> dict:
    """Cells, coverage and suppression for one dimension."""
    spec = TONE_DIMENSIONS[dimension]
    measured = spec.measure_fn(df)
    cells = (
        spec.aggregate_fn(measured)
        if spec.aggregate_fn
        else aggregate_cells(measured, group_col=spec.group_col, per=spec.per, alpha=spec.alpha)
    )
    kept = suppress_thin_cells(cells, spec.min_cell_speeches, spec.min_cell_n)

    total_cells = sum(len(v) for v in cells.values())
    kept_cells = sum(len(v) for v in kept.values())
    hits = int(measured["hits"].sum()) if "hits" in measured else 0

    density = hit_density(cells)

    print(f"\n{'=' * 78}\n{dimension}  ({spec.technique}), {spec.label_sv}\n{'=' * 78}")
    print(f"  unit          : {spec.unit_sv}")
    print(f"  total hits    : {hits:,}")
    print(f"  cells         : {total_cells} -> {kept_cells} kept after suppression")

    # The check the speech-count floor cannot make: enough TEXT is not enough
    # EVIDENCE. antielit passed every floor with a median of 2 hits per cell.
    if spec.technique != "stylometric":
        verdict = ""
        if density["median_hits"] < 5 or density["empty_share"] > 0.35:
            verdict = "  <-- TOO SPARSE TO CHART: this is noise with a line through it"
        leading = (
            f", {density['leading_empty']:.0f} cells before it first appears"
            if density["leading_empty"]
            else ""
        )
        print(
            f"  density       : median {density['median_hits']:.0f} hits/cell, "
            f"{density['empty_share']:.0%} empty once attested{leading}{verdict}"
        )

    if spec.technique == "stylometric":
        excluded = readability.excluded_share(measured)
        print(f"  excluded      : {excluded:.1%} of speeches below the LIX inclusion floor")
        print("  precision     : n/a, arithmetic, not a judgement call. Worked example:")
        sample = measured[measured["included"]].iloc[0]
        print(
            f"      {sample['words']} ord / {sample['sentences']} meningar"
            f" + {sample['long_words']}*100/{sample['words']} = "
            f"LIX {readability.lix(sample['words'], sample['sentences'], sample['long_words']):.1f}"
        )

    for party in sorted(kept):
        years = kept[party]
        if not years:
            continue
        first, last = min(years), max(years)
        print(
            f"    {party:3} {first}-{last}  "
            f"{years[first].smoothed:7.2f} -> {years[last].smoothed:7.2f}"
        )

    if spec.receipt_kind == "evidentiary":
        sample = precision_sample(df, dimension, sample_size)
        if not sample.empty:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / f"precision_{dimension}.csv"
            sample.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n  -> {len(sample)} matches for hand-labelling: {path}")
            print(
                "     (fill in is_correct__fill_in with 1/0; the resulting number gets published)"
            )

    return {
        "dimension": dimension,
        "technique": spec.technique,
        "hits": hits,
        "cells": total_cells,
        "cells_kept": kept_cells,
        "density": density,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the tone dimensions.")
    parser.add_argument("--dimension", action="append", help="Dimension id (repeatable).")
    parser.add_argument("--sample", type=int, default=120, help="Precision-audit sample size.")
    parser.add_argument("--limit", type=int, default=None, help="Sub-sample the corpus (dev).")
    args = parser.parse_args()

    df = load_corpus(args.limit)
    wanted = args.dimension or [d for d, s in TONE_DIMENSIONS.items() if s.status == "launch"]

    summary = [report_dimension(df, d, args.sample) for d in wanted]

    # ------------------------------------------------------------- symmetry
    print(f"\n{'=' * 78}\nSYMMETRY, where does the instrument find the pattern?\n{'=' * 78}")
    table = symmetry(df)
    if not table.empty:
        print(table.to_string())
        print(
            "\n  Read this before shipping: if the out-group columns are lopsided in a way\n"
            "  that tracks the pattern list rather than the speech, the list is the problem."
        )

    # --------------------------------------------------------------- census
    census = vi_dom.vi_dom_census(df)
    print(
        f"\n{'=' * 78}\nVI-MOT-DOM CENSUS, {len(census)} constructions in the whole corpus\n{'=' * 78}"
    )
    if census:
        by_party = pd.Series([c.party for c in census]).value_counts()
        by_type = pd.Series([t for c in census for t in c.subtypes]).value_counts()
        print(f"  by party : {by_party.to_dict()}")
        print(f"  by type  : {by_type.to_dict()}")
        for c in census[:5]:
            print(f"    [{c.party} {c.year}] {c.sentence[:120]}")

    # ---------------------------------------------------------- occupational
    print(f"\n{'=' * 78}\nOCCUPATIONAL SUBSTITUTION, the honest null result\n{'=' * 78}")
    pairs = inclusive.occupational_report(df)
    for pair in pairs["pairs"]:
        ratio = "  n/a" if pair["ratio"] is None else f"{pair['ratio']:5.1%}"
        print(
            f"  {pair['gendered']:16} -> {str(pair['neutral'] or ', '):18} "
            f"{pair['gendered_hits']:5} vs {pair['neutral_hits']:5}   neutral share {ratio}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "dimensions": summary,
        "symmetry": table.to_dict() if not table.empty else {},
        "census": [c.as_dict() for c in census],
        "occupational": pairs,
    }
    (OUT_DIR / "tone_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'tone_validation.json'}")


if __name__ == "__main__":
    main()
