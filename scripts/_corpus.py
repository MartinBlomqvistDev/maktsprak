"""Shared local-corpus loader for precompute/analysis scripts.

The archived corpus (``data/parquet/speeches_full.parquet``) is the source of
truth for analysis and site precompute, not Supabase: the
Vercel site only ever reads static JSON, and Supabase's free tier can't hold
the full history anyway. Loading from the local archive is also far faster and
has no network/quota dependency.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parent.parent / "data" / "parquet" / "speeches_full.parquet"
)


def load_corpus(
    start_date: str | None = None,
    end_date: str | None = None,
    archive: Path = DEFAULT_ARCHIVE,
) -> pd.DataFrame:
    """Load the archived speech corpus, optionally bounded by date.

    Args:
        start_date: Inclusive lower bound (ISO date string), or ``None``.
        end_date:   Inclusive upper bound (ISO date string), or ``None``.
        archive:    Path to the Parquet archive.

    Returns:
        DataFrame with columns ``id``, ``protocol_id``, ``protocol_date``,
        ``speaker``, ``party``, ``text``, ``file_url``, sorted by date.

    Raises:
        FileNotFoundError: If the archive doesn't exist yet (run
            ``scripts/export_corpus.py`` first).
    """
    if not archive.exists():
        raise FileNotFoundError(
            f"Corpus archive not found at {archive}. Run scripts/export_corpus.py first."
        )
    df = pd.read_parquet(archive)

    # The archive held 5 425 duplicate-id rows once (the 2002-2015 backfill
    # overlapped the range the ETL had already ingested; 2014 was 95%
    # duplicated). A duplicated speech is counted twice in every denominator
    # and inflates the confidence of every z-score built on it, and nothing
    # downstream can detect it. export_corpus.py dedups now, but this is the
    # gate every analysis passes through, so it fails loudly here rather than
    # letting a stale or hand-made archive quietly skew a published number.
    duplicated = int(df["id"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{archive.name} has {duplicated} duplicate-id rows. Analysis built on it "
            f"would double-count those speeches. Re-run scripts/export_corpus.py, which "
            f"deduplicates on export."
        )

    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    df = df.dropna(subset=["protocol_date"])
    if start_date:
        df = df[df["protocol_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["protocol_date"] <= pd.to_datetime(end_date)]
    return df.sort_values("protocol_date").reset_index(drop=True)
