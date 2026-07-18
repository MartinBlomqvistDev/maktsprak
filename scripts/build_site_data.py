"""Precompute the static JSON the Next.js/Vercel site renders.

The site has no Python runtime, so all analysis is computed here and written as
small JSON files it can fetch statically. Reads from the local Parquet archive
(``data/parquet/speeches_full.parquet``), not Supabase, that archive is the
source of truth for analysis; run ``scripts/export_corpus.py``
first if it doesn't exist yet or is stale.

Outputs (in ``--out``):
  meta.json             corpus size, date span, per-year speech counts.
  distinctive.json      per party: its most distinctive words (Fightin' Words),
                         for the rhetorical-fingerprint word clouds.
  movers.json            top risers / fallers across the whole window (what the
                         debate moved toward and away from).
  trajectories.json     per-year relative frequency of the notable movers, for
                         line charts.
  divergence.json       mean pairwise party JS divergence per year (are the
                         parties converging or diverging?).
  year_signatures.json  per-year distinctive words, for a year-by-year scrubber.
  party_frames.json     per-party, per-year usage rate of curated issue frames
                         (crime, migration, climate, welfare, economy), for
                         "who owns this issue, and is that changing" charts.

Usage::

    python scripts/build_site_data.py [--start-date 2002-09-01] [--split-year 2020]
                                      [--out data/site]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _corpus import load_corpus

from src.maktsprak_pipeline.config import PARTY_ORDER
from src.maktsprak_pipeline.logger import get_logger
from src.maktsprak_pipeline.nlp import (
    ISSUE_FRAMES,
    add_year,
    distinctive_words,
    frame_trajectories,
    group_token_counts,
    party_divergence_by_year,
    speaker_name_stopwords,
    term_trajectories,
    top_movers,
    weighted_log_odds,
    yearly_signatures,
)
from src.maktsprak_pipeline.nlp.distinctiveness import _DISTINCT_STOPWORDS

logger = get_logger()


def _round_pairs(pairs: list[tuple[str, float]], ndigits: int = 2) -> list[dict[str, float | str]]:
    """Turn ``(word, score)`` pairs into rounded ``{"word", "z"}`` dicts."""
    return [{"word": w, "z": round(z, ndigits)} for w, z in pairs]


#: The Next.js app imports these from ``web/src/data`` (see
#: ``web/src/lib/site-data.ts``), so every payload is written to both places.
#: Keeping it a manual copy meant a regenerate could silently leave the live
#: site on stale numbers, with nothing to catch it, precompute drift is
#: invisible by construction, because the old file is still perfectly valid JSON.
WEB_DATA_DIR = Path("web/src/data")


def _write(out_dir: Path, name: str, payload: object) -> None:
    """Write one payload to *out_dir* and mirror it into the web app.

    The mirror is unconditional rather than an argument threaded through every
    call site: a call site is something a future edit can forget, and forgetting
    this one leaves the live site on stale numbers with nothing to catch it.
    """
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    path = out_dir / name
    path.write_text(body, encoding="utf-8")

    mirrored = ""
    if WEB_DATA_DIR.exists():
        (WEB_DATA_DIR / name).write_text(body, encoding="utf-8")
        mirrored = f" -> also {WEB_DATA_DIR / name}"
    logger.info(f"Wrote {path} ({path.stat().st_size // 1024 or 1} KB){mirrored}.")


def build(
    start_date: str | None, end_date: str | None, split_year: int, out_dir: Path, top_n: int = 60
) -> None:
    """Compute every site JSON payload and write it to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading corpus archive from {start_date or 'the beginning'}...")
    df = add_year(load_corpus(start_date=start_date, end_date=end_date))
    logger.info(f"Loaded {len(df)} speeches, {df['year'].min()}-{df['year'].max()}.")

    # Corpus-derived name stopwords: the curated POLITICIAN_NAME_STOPWORDS list
    # only covers names someone happened to spot-check; with ~1,600 unique
    # speakers across 2002-2026 that never keeps up (older years especially).
    # The speaker column is ground truth for who actually spoke, so derive the
    # stopword set from it directly and use that everywhere below instead of
    # each function's smaller built-in default.
    speaker_stops = speaker_name_stopwords(df["speaker"])
    stopwords = _DISTINCT_STOPWORDS | speaker_stops
    logger.info(
        f"Derived {len(speaker_stops)} name tokens from {df['speaker'].nunique()} speakers."
    )

    # --- meta ---
    per_year = df["year"].value_counts().sort_index()
    _write(
        out_dir,
        "meta.json",
        {
            "count": int(len(df)),
            "first_year": int(df["year"].min()),
            "last_year": int(df["year"].max()),
            "per_year": {int(y): int(n) for y, n in per_year.items()},
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    # --- distinctive words per party (word clouds) ---
    party_counts = group_token_counts(df, group_col="party", text_col="text", stopwords=stopwords)
    party_scores = weighted_log_odds(party_counts, min_count=10)
    distinctive = {
        p: _round_pairs(distinctive_words(party_scores, p, top_n=top_n))
        for p in PARTY_ORDER
        if p in party_scores
    }
    _write(out_dir, "distinctive.json", distinctive)

    # --- movers (what the whole chamber moved toward / away from) ---
    risers, fallers = top_movers(
        df, split_year=split_year, top_n=40, min_count=30, stopwords=stopwords
    )
    _write(
        out_dir,
        "movers.json",
        {
            "split_year": split_year,
            "risers": _round_pairs(risers),
            "fallers": _round_pairs(fallers),
        },
    )

    # --- trajectories of the notable movers ---
    # Top-15 algorithmic movers each side, plus a curated set of terms the site
    # highlights by name in its copy (e.g. "kärnkraft"), some of those are real,
    # data-backed movers that just miss a top-10/15 cutoff by rank (e.g.
    # "kärnkraft" is riser #15-20 depending on the run), not fabricated trends.
    # Without this a curated term silently renders as an all-zero flat line if
    # it falls outside the cutoff.
    curated = [
        "ukraina",
        "pandemin",
        "inflationen",
        "gaza",
        "kärnkraft",
        "alliansen",
        "landsting",
        "jobb",
        "arbetsmarknaden",
        "rut",
    ]
    notable = list(
        dict.fromkeys([w for w, _ in risers[:15]] + [w for w, _ in fallers[:15]] + curated)
    )
    traj = term_trajectories(df, notable)
    _write(
        out_dir,
        "trajectories.json",
        {
            "terms": notable,
            "series": {int(y): {t: round(v, 2) for t, v in row.items()} for y, row in traj.items()},
        },
    )

    # --- party divergence over time ---
    divergence = party_divergence_by_year(df, min_count=10)
    _write(out_dir, "divergence.json", {int(y): round(v, 4) for y, v in divergence.items()})

    # --- each year's own signature, for the year-scrubber ---
    signatures = yearly_signatures(df, top_n=14, min_count=15, stopwords=stopwords)
    _write(
        out_dir,
        "year_signatures.json",
        {int(y): _round_pairs(words, 1) for y, words in signatures.items()},
    )

    # --- per-party issue-frame trajectories ("who owns this issue") ---
    frames = frame_trajectories(df, frames=ISSUE_FRAMES)
    _write(out_dir, "party_frames.json", {"frames": list(ISSUE_FRAMES.keys()), "series": frames})

    logger.info("Site data build complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute static JSON for the MaktspråkAI site.")
    parser.add_argument("--start-date", default=None, help="ISO date; omit for the whole corpus.")
    parser.add_argument("--end-date", default=None, help="ISO date; omit for up to the present.")
    parser.add_argument(
        "--split-year", type=int, default=2020, help="Early/late boundary for movers."
    )
    parser.add_argument("--out", type=Path, default=Path("data/site"), help="Output directory.")
    args = parser.parse_args()
    build(args.start_date, args.end_date, args.split_year, args.out)


if __name__ == "__main__":
    main()
