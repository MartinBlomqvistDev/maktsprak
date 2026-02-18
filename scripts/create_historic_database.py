"""Dump the entire ``speeches`` table to a local Parquet snapshot.

Run this whenever you want to refresh the historical baseline that the
Streamlit app loads via ``PARQUET_URL``.  After running:

1. Upload the generated ``.parquet`` file to Google Drive (or any public host).
2. Get a direct-download URL and set ``PARQUET_URL=<url>`` in ``.env`` /
   Streamlit Secrets.

Usage::

    python scripts/create_historic_database.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when the script is run directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.maktsprak_pipeline.db.client import _get_read_client
from src.maktsprak_pipeline.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"


def fetch_all_speeches() -> pd.DataFrame:
    """Paginate through the ``speeches`` table and return the full dataset.

    Returns:
        DataFrame with all columns, sorted by ``protocol_date``.

    Raises:
        RuntimeError: If no rows are returned from Supabase.
    """
    client = _get_read_client()
    batch_size = 1000
    offset = 0
    dfs: list[pd.DataFrame] = []

    query = client.table("speeches").select("*").order("protocol_date", desc=False)

    while True:
        resp = query.range(offset, offset + batch_size - 1).execute()
        if not resp.data:
            break
        dfs.append(pd.DataFrame(resp.data))
        total_so_far = sum(len(d) for d in dfs)
        logger.info(f"Fetched batch of {len(resp.data)} rows (running total: {total_so_far:,})")
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not dfs:
        raise RuntimeError("No data returned from Supabase — is SUPABASE_URL/KEY set?")

    df = pd.concat(dfs, ignore_index=True)
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    return df


def write_parquet(df: pd.DataFrame) -> Path:
    """Write *df* to a snappy-compressed Parquet file dated today.

    Args:
        df: DataFrame to serialise.

    Returns:
        Path to the written file.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = _OUT_DIR / f"speeches_historic_{date.today().strftime('%Y%m%d')}.parquet"
    pq.write_table(pa.Table.from_pandas(df), filename, compression="snappy")
    logger.info(f"Parquet written: {filename} ({len(df):,} rows)")
    return filename


if __name__ == "__main__":
    logger.info("Fetching all speeches from Supabase…")
    df = fetch_all_speeches()
    logger.info(f"Total rows fetched: {len(df):,}")

    out = write_parquet(df)

    logger.info("Done.")
    logger.info(f"Upload '{out}' to Google Drive, then add to .env:")
    logger.info("  PARQUET_URL=<your-direct-download-url>")
