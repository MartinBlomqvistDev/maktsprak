"""Speech CRUD operations against the Supabase ``speeches`` table."""

from __future__ import annotations

import io
import random
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests

from ..config import PARQUET_URL
from ..logger import get_logger
from .client import supabase, supabase_write

logger = get_logger()

# ---------------------------------------------------------------------------
# Streamlit cache — applied conditionally so the module works outside Streamlit
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    _USE_STREAMLIT: bool = get_script_run_ctx() is not None
except Exception:
    _USE_STREAMLIT = False


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def fetch_speeches_count() -> int:
    """Return the total number of rows in the ``speeches`` table.

    Returns:
        Row count as an integer.

    Raises:
        RuntimeError: If the Supabase query fails.
    """
    resp = supabase.table("speeches").select("protocol_id", count="exact").execute()
    if resp.data is None:
        raise RuntimeError(f"Supabase fetch failed: {resp}")
    return resp.count or 0


def fetch_latest_speech_date() -> str | None:
    """Return the most recent ``protocol_date`` value in the ``speeches`` table.

    Returns:
        ISO-format date string, or ``None`` if the table is empty.
    """
    resp = (
        supabase.table("speeches")
        .select("protocol_date")
        .order("protocol_date", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return resp.data[0]["protocol_date"]


if _USE_STREAMLIT:
    @st.cache_data(ttl=3600)
    def fetch_latest_speech_date_cached() -> str | None:
        """Streamlit-cached wrapper around :func:`fetch_latest_speech_date`."""
        return fetch_latest_speech_date()
else:
    fetch_latest_speech_date_cached = fetch_latest_speech_date


def fetch_random_speeches(limit: int = 5) -> list[dict[str, Any]]:
    """Return a random sample of speeches without fetching the full table.

    Uses a random page offset strategy to avoid pulling 44 k+ rows into memory.

    Args:
        limit: Number of speeches to return.

    Returns:
        List of speech dicts (all columns).
    """
    # Fetch a modest window at a random position — fast and memory-safe.
    count_resp = supabase.table("speeches").select("id", count="exact").execute()
    total = count_resp.count or 0
    if total == 0:
        return []

    window = max(limit * 10, 50)
    offset = random.randint(0, max(0, total - window))
    resp = (
        supabase.table("speeches")
        .select("*")
        .range(offset, offset + window - 1)
        .execute()
    )
    if not resp.data:
        return []
    sample = resp.data.copy()
    random.shuffle(sample)
    return sample[:limit]


def fetch_speeches_historical(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch speeches from Supabase within a date range, using pagination.

    Args:
        start_date: Inclusive lower bound (ISO date string).
        end_date:   Inclusive upper bound.  ``None`` means no upper limit.

    Returns:
        DataFrame with columns ``id``, ``text``, ``party``, ``protocol_date``.
    """
    dfs: list[pd.DataFrame] = []
    batch_size = 1000
    offset = 0

    query = supabase.table("speeches").select("id, text, party, protocol_date")
    if start_date:
        query = query.gte("protocol_date", str(start_date))
    if end_date:
        query = query.lte("protocol_date", str(end_date))
    query = query.order("protocol_date", desc=False)

    while True:
        resp = query.range(offset, offset + batch_size - 1).execute()
        if not resp.data:
            break
        dfs.append(pd.DataFrame(resp.data))
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not dfs:
        return pd.DataFrame(columns=["id", "text", "party", "protocol_date"])

    df = pd.concat(dfs, ignore_index=True)
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    return df.dropna(subset=["protocol_date"]).sort_values("protocol_date").reset_index(drop=True)


def load_historical_parquet(parquet_url: str | None = None) -> pd.DataFrame:
    """Download and parse the historical Parquet snapshot.

    Args:
        parquet_url: Direct-download URL.  Defaults to :data:`~config.PARQUET_URL`.

    Returns:
        DataFrame with columns ``id``, ``text``, ``party``, ``protocol_date``.
    """
    url = parquet_url or PARQUET_URL
    if not url:
        logger.warning("No Parquet URL configured — returning empty DataFrame.")
        return pd.DataFrame(columns=["id", "text", "party", "protocol_date"])

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pq.read_table(io.BytesIO(resp.content)).to_pandas()
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    logger.info(f"Loaded {len(df):,} rows from Parquet snapshot.")
    return df


def fetch_combined_speeches(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Merge the historical Parquet snapshot with live Supabase rows.

    Deduplicates on ``id`` and sorts by ``protocol_date``.

    Args:
        start_date: Inclusive lower bound for filtering.
        end_date:   Inclusive upper bound.

    Returns:
        Combined, deduplicated DataFrame.
    """
    df_hist = load_historical_parquet()
    df_hist = df_hist[df_hist["protocol_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df_hist = df_hist[df_hist["protocol_date"] <= pd.to_datetime(end_date)]

    if df_hist.empty:
        cutoff = start_date
    else:
        cutoff = (df_hist["protocol_date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df_new = fetch_speeches_historical(start_date=cutoff, end_date=end_date)
    df_combined = (
        pd.concat([df_hist, df_new], ignore_index=True)
        .drop_duplicates(subset=["id"])
        .sort_values("protocol_date")
        .reset_index(drop=True)
    )
    logger.info(
        f"Combined speeches — historic: {len(df_hist):,}, new: {len(df_new):,}, "
        f"total: {len(df_combined):,}"
    )
    return df_combined


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def insert_speech(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Upsert a single speech record into the ``speeches`` table.

    Conflicts on the primary key (``id``) are resolved by updating the row.

    Args:
        row: Speech dict with keys matching the table schema.

    Returns:
        The upserted row(s) returned by Supabase, or ``None`` on a no-data response.

    Raises:
        RuntimeError: If Supabase returns a ``None`` data payload (hard failure).
    """
    resp = supabase_write.table("speeches").upsert(row).execute()
    if resp.data is None:
        raise RuntimeError(f"Supabase upsert failed for id={row.get('id')}: {resp}")
    if resp.data:
        logger.debug(f"Speech upserted: {row.get('id')}")
        return resp.data
    logger.warning(f"Upsert returned no data for id={row.get('id')}")
    return None


def delete_speeches_invalid_parties(valid_parties: frozenset[str]) -> int:
    """Delete speeches whose ``party`` value is not in ``valid_parties``.

    Args:
        valid_parties: The canonical set of accepted party abbreviations.

    Returns:
        Number of rows deleted.
    """
    resp = (
        supabase_write.table("speeches")
        .delete()
        .not_("party", "in", list(valid_parties))
        .execute()
    )
    count = len(resp.data) if resp.data else 0
    if count:
        logger.info(f"Deleted {count} speeches with invalid party labels.")
    return count
