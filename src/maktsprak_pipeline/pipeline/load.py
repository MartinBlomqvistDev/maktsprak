"""Load phase — persist transformed records to Supabase."""

from __future__ import annotations

from typing import Any

from ..db import insert_speech, insert_tweet
from ..logger import get_logger

logger = get_logger()


def load_riksdag(speeches: list[dict[str, Any]]) -> None:
    """Upsert a list of speech records into Supabase.

    Args:
        speeches: Output of :func:`~.transform.transform_riksdag` or
            :func:`~.transform._process_doc`.
    """
    for row in speeches:
        insert_speech(row)
    logger.info(f"{len(speeches)} speech record(s) upserted.")


def load_tweets(tweets: list[dict[str, Any]]) -> None:
    """Upsert a list of tweet records into Supabase.

    Args:
        tweets: Output of :func:`~.extract.extract_all_tweets`.
    """
    for row in tweets:
        insert_tweet(row)
    logger.info(f"{len(tweets)} tweet record(s) processed (duplicates skipped).")
