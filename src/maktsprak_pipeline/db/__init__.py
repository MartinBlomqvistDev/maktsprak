"""Database sub-package for MaktspråkAI.

Re-exports the public API so callers can do::

    from src.maktsprak_pipeline.db import fetch_speeches_count, insert_speech
"""

from .client import supabase, supabase_write
from .speeches import (
    delete_speeches_by_protocol,
    delete_speeches_invalid_parties,
    fetch_combined_speeches,
    fetch_latest_speech_date,
    fetch_latest_speech_date_cached,
    fetch_random_speeches,
    fetch_speeches_count,
    fetch_speeches_historical,
    insert_speech,
    insert_speeches,
    load_historical_parquet,
)
from .tweets import fetch_all_tweets, fetch_tweets_count_since, insert_tweet

__all__ = [
    "supabase",
    "supabase_write",
    # speeches
    "fetch_speeches_count",
    "fetch_latest_speech_date",
    "fetch_latest_speech_date_cached",
    "fetch_random_speeches",
    "fetch_speeches_historical",
    "fetch_combined_speeches",
    "load_historical_parquet",
    "insert_speech",
    "insert_speeches",
    "delete_speeches_by_protocol",
    "delete_speeches_invalid_parties",
    # tweets
    "fetch_all_tweets",
    "fetch_tweets_count_since",
    "insert_tweet",
]
