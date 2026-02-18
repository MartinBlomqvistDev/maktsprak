"""Tweet CRUD operations against the Supabase ``tweets`` table."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..logger import get_logger
from .client import supabase, supabase_write

logger = get_logger()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def fetch_tweets_count_since(since_date: str) -> int:
    """Return the number of tweets created on or after ``since_date``.

    Args:
        since_date: ISO-format datetime string (e.g. ``"2025-02-01T00:00:00Z"``).

    Returns:
        Tweet count as an integer.
    """
    resp = (
        supabase.table("tweets")
        .select("tweet_id", count="exact")
        .gte("created_at", since_date)
        .execute()
    )
    return resp.count or 0


def fetch_all_tweets() -> pd.DataFrame:
    """Fetch every tweet from Supabase, using pagination.

    Returns:
        DataFrame with columns ``tweet_id``, ``username``, ``text``, ``created_at``.
    """
    dfs: list[pd.DataFrame] = []
    batch_size = 1000
    offset = 0

    while True:
        resp = (
            supabase.table("tweets")
            .select("tweet_id, username, text, created_at")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if not resp.data:
            break
        dfs.append(pd.DataFrame(resp.data))
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not dfs:
        return pd.DataFrame(columns=["tweet_id", "username", "text", "created_at"])
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def insert_tweet(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Insert a tweet, skipping rows that already exist in the table.

    Uses an upsert with ``on_conflict`` on ``tweet_id`` so a single round-trip
    replaces the previous SELECT-then-INSERT pattern.

    Args:
        row: Tweet dict with keys matching the ``tweets`` table schema.

    Returns:
        The inserted/updated row(s), or ``None`` when nothing changed.

    Raises:
        RuntimeError: If Supabase returns a ``None`` data payload.
    """
    resp = (
        supabase_write.table("tweets")
        .upsert(row, on_conflict="tweet_id", ignore_duplicates=True)
        .execute()
    )
    if resp.data is None:
        raise RuntimeError(f"Supabase upsert tweet failed: {resp}")
    if resp.data:
        logger.debug(f"Tweet upserted: {row.get('tweet_id')}")
    else:
        logger.debug(f"Tweet already exists, skipped: {row.get('tweet_id')}")
    return resp.data or None
