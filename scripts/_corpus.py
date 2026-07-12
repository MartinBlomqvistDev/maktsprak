"""Shared local-corpus loader for precompute/analysis scripts.

The archived corpus (``data/parquet/speeches_full.parquet``) is the source of
truth for analysis and site precompute, not Supabase (see DEV_LOG #10): the
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
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    df = df.dropna(subset=["protocol_date"])
    if start_date:
        df = df[df["protocol_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["protocol_date"] <= pd.to_datetime(end_date)]
    return df.sort_values("protocol_date").reset_index(drop=True)
