"""Load phase, persist transformed records to Supabase."""

from __future__ import annotations

from typing import Any

from ..db import insert_tweet
from ..logger import get_logger

logger = get_logger()


def load_tweets(tweets: list[dict[str, Any]]) -> None:
    """Upsert a list of tweet records into Supabase.

    Args:
        tweets: Output of :func:`~.extract.extract_all_tweets`.
    """
    for row in tweets:
        insert_tweet(row)
    logger.info(f"{len(tweets)} tweet record(s) processed (duplicates skipped).")
