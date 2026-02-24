# =========================================================
# File: src/maktsprak_pipeline/db.py
# Purpose: Database module for MaktsprakAI (Supabase REST API + external Parquet)
# Dependencies:
#   - supabase-py
#   - streamlit (optional, fallback to .env)
#   - logger
#   - tenacity
#   - pandas, pyarrow
# =========================================================

import os
import io
import random
from datetime import datetime
from tenacity import retry, wait_fixed, stop_after_attempt

import pandas as pd
import pyarrow.parquet as pq
import requests

from supabase import create_client
from .logger import get_logger
from .config import PARQUET_URL

logger = get_logger()

# =========================================================
# Initialize Supabase client (Streamlit Secrets or .env)
# =========================================================

from dotenv import load_dotenv
load_dotenv()

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    import streamlit as st
    USE_STREAMLIT = get_script_run_ctx() is not None
except Exception:
    USE_STREAMLIT = False

if USE_STREAMLIT:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", SUPABASE_KEY)
else:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", SUPABASE_KEY)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in .env or Streamlit Secrets.")

# Read client: anon key (respects RLS)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# Write client: service key (bypasses RLS) — falls back to anon key if not set
supabase_write = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# =========================================================
# Read functions
# =========================================================

def fetch_speeches_count():
    """Returns the total number of rows in the 'speeches' table."""
    resp = supabase.table("speeches").select("protocol_id", count="exact").execute()
    if resp.data is None:
        raise Exception(f"Supabase fetch failed: {resp}")
    return resp.count

def fetch_latest_speech_date():
    """Returns the most recent protocol_date from the 'speeches' table."""
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

if USE_STREAMLIT:
    @st.cache_data(ttl=3600)
    def fetch_latest_speech_date_cached():
        return fetch_latest_speech_date()
else:
    fetch_latest_speech_date_cached = fetch_latest_speech_date

def fetch_random_speeches(limit: int = 5):
    """Fetches random speeches."""
    resp = supabase.table("speeches").select("*").execute()
    if not resp.data:
        return []
    data = resp.data
    random.shuffle(data)
    return data[:limit]

def fetch_speeches_historical(start_date="2015-01-01", end_date=None):
    """Fetches all speeches from Supabase between start_date and end_date."""
    dfs = []
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
        if resp.data:
            dfs.append(pd.DataFrame(resp.data))
            if len(resp.data) < batch_size:
                break
            offset += batch_size
        else:
            break

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
        df = df.dropna(subset=["protocol_date"])
        df = df.sort_values("protocol_date").reset_index(drop=True)
        return df

    return pd.DataFrame(columns=["id", "text", "party", "protocol_date"])

def fetch_all_tweets():
    """Fetches all tweets from Supabase for training purposes."""
    dfs = []
    batch_size = 1000
    offset = 0

    while True:
        resp = (
            supabase.table("tweets")
            .select("tweet_id, username, text, created_at")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if resp.data:
            dfs.append(pd.DataFrame(resp.data))
            if len(resp.data) < batch_size:
                break
            offset += batch_size
        else:
            break

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["tweet_id", "username", "text", "created_at"])

def fetch_tweets_count_since(since_date: str) -> int:
    """Returns the number of tweets created on or after since_date."""
    resp = (
        supabase.table("tweets")
        .select("tweet_id", count="exact")
        .gte("created_at", since_date)
        .execute()
    )
    return resp.count or 0

# =========================================================
# Load historical Parquet from external URL
# =========================================================

def load_historical_parquet(parquet_url: str):
    """
    Reads Parquet with historical data from an external URL (e.g. Google Drive).
    Expects columns: id, text, party, protocol_date
    """
    if not parquet_url:
        logger.warning("No Parquet URL defined, returning empty DataFrame")
        return pd.DataFrame(columns=["id", "text", "party", "protocol_date"])

    r = requests.get(parquet_url)
    r.raise_for_status()
    table = pq.read_table(io.BytesIO(r.content))
    df = table.to_pandas()
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    return df

# =========================================================
# Combine historical Parquet + new Supabase data
# =========================================================

def fetch_combined_speeches(start_date="2015-01-01", end_date=None):
    df_historic = load_historical_parquet(PARQUET_URL)
    df_historic = df_historic[df_historic["protocol_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df_historic = df_historic[df_historic["protocol_date"] <= pd.to_datetime(end_date)]

    last_date = df_historic["protocol_date"].max() + pd.Timedelta(days=1) if not df_historic.empty else start_date
    df_new = fetch_speeches_historical(start_date=last_date, end_date=end_date)

    df_combined = pd.concat([df_historic, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id"])
    df_combined = df_combined.sort_values("protocol_date").reset_index(drop=True)

    logger.info(f"Historic: {len(df_historic)}, New: {len(df_new)}, Total: {len(df_combined)}")
    return df_combined

# =========================================================
# Write functions
# =========================================================

def insert_speech(row: dict):
    """Inserts or updates a speech in the 'speeches' table. Conflicts on protocol_id."""
    resp = supabase_write.table("speeches").upsert(row).execute()

    if resp.data is None:
        logger.error(f"Supabase upsert failed for row: {row}")
        raise Exception(f"Supabase upsert failed: {resp}")

    if resp.data:
        logger.info(f"Speech upserted in 'speeches': {row.get('id', 'no-id')}")
        return resp.data
    else:
        logger.warning(f"Upsert returned no data for {row.get('id')}")
        return None

def insert_tweet(row: dict):
    """Inserts a new tweet, skipping duplicates based on tweet_id."""
    if "tweet_id" in row:
        existing = supabase.table("tweets").select("tweet_id").eq("tweet_id", row["tweet_id"]).execute()
        if existing.data:
            logger.warning(f"Tweet already exists: tweet_id {row['tweet_id']} – skipping.")
            return existing.data

    resp = supabase_write.table("tweets").insert(row).execute()
    if resp.data is None:
        raise Exception(f"Supabase insert tweet failed: {resp}")
    logger.info("New tweet inserted into 'tweets'")
    return resp.data

# =========================================================
# Maintenance functions
# =========================================================

def delete_speeches_invalid_parties(valid_parties: set) -> int:
    """Deletes speeches whose party is not in valid_parties. Returns count deleted."""
    resp = (
        supabase_write.table("speeches")
        .delete()
        .not_("party", "in", list(valid_parties))
        .execute()
    )
    return len(resp.data) if resp.data else 0

def create_all_tables():
    logger.warning(
        "Tables are managed directly in Supabase. "
        "Use the Supabase Dashboard or SQL scripts for schema changes."
    )
