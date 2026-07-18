"""Re-parse cached Riksdag protocols and replace the ``speeches`` table.

The PDF parser was rewritten to fix two-column interleaving, running-header
bleed, ``replik`` reply headers and party-less-speaker mis-attribution.  Rows
ingested before that fix are incomplete and mislabelled, so the whole table
needs re-parsing.

Because the new parser groups speakers differently (and therefore assigns
different ``id`` values), a plain re-run of ``backfill_speeches.py`` would leave
stale rows behind.  This script instead replaces each protocol atomically:
delete the protocol's existing rows, then batch-insert the freshly-parsed ones.

Usage::

    # Safe dry run, parse everything, write nothing, report counts:
    python scripts/reindex_speeches.py --dry-run

    # Validate end-to-end on a single protocol (writes to DB):
    python scripts/reindex_speeches.py --limit 1

    # Full re-index from the start of the dataset:
    python scripts/reindex_speeches.py --from-date 2014-01-01

A timestamped, full-column backup of the current table is written to
``data/parquet/`` before any writes (skip with ``--skip-backup``).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
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
from tqdm import tqdm

from src.maktsprak_pipeline.db import (
    delete_speeches_by_protocol,
    insert_speeches,
)
from src.maktsprak_pipeline.db.client import _get_read_client
from src.maktsprak_pipeline.logger import get_logger
from src.maktsprak_pipeline.pipeline.orchestrate import fetch_protocol_docs
from src.maktsprak_pipeline.pipeline.transform import _process_doc

logger = get_logger()

_BACKUP_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"


def backup_speeches() -> Path:
    """Dump the current ``speeches`` table (all columns) to a timestamped Parquet.

    Returns:
        Path to the written backup file.

    Raises:
        RuntimeError: If the table is empty (nothing to back up).
    """
    client = _get_read_client()
    batch_size = 1000
    offset = 0
    frames: list[pd.DataFrame] = []

    query = client.table("speeches").select("*").order("id", desc=False)
    while True:
        resp = query.range(offset, offset + batch_size - 1).execute()
        if not resp.data:
            break
        frames.append(pd.DataFrame(resp.data))
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not frames:
        raise RuntimeError("No rows to back up, is the table already empty / creds set?")

    df = pd.concat(frames, ignore_index=True)
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _BACKUP_DIR / f"speeches_backup_{stamp}.parquet"
    pq.write_table(pa.Table.from_pandas(df), out, compression="snappy")
    logger.info(f"Backup written: {out} ({len(df):,} rows)")
    return out


def reindex(from_date: str, dry_run: bool, limit: int | None) -> None:
    """Re-parse protocols and replace their rows in the ``speeches`` table.

    Args:
        from_date: ISO start date for the API listing window.
        dry_run:   If ``True``, parse and report but never write to Supabase.
        limit:     Process at most this many protocols (``None`` = all).
    """
    to_date = date.today().strftime("%Y-%m-%d")
    logger.info(f"Re-index window: {from_date} -> {to_date} (dry_run={dry_run}, limit={limit})")

    docs = fetch_protocol_docs(from_date, to_date)
    if limit is not None:
        docs = docs[:limit]
    logger.info(f"Processing {len(docs)} protocol(s)…")

    protocols_with_speeches = 0
    total_deleted = 0
    total_inserted = 0

    for doc in tqdm(docs, desc="Re-indexing protocols"):
        protocol_id = doc.findtext("dok_id")
        records = _process_doc(doc)
        if records:
            protocols_with_speeches += 1

        if dry_run:
            total_inserted += len(records)
            continue

        if protocol_id:
            total_deleted += delete_speeches_by_protocol(protocol_id)
        total_inserted += insert_speeches(records)

    verb = "would insert" if dry_run else f"inserted (deleted {total_deleted} old)"
    logger.info(
        f"Re-index complete: {total_inserted} speeches {verb} "
        f"across {protocols_with_speeches}/{len(docs)} protocols with speeches."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-parse and replace the speeches table.")
    parser.add_argument(
        "--from-date", default="2014-01-01", help="ISO start date (default: 2014-01-01)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and report; write nothing.")
    parser.add_argument("--skip-backup", action="store_true", help="Do not back up before writing.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N protocols.")
    args = parser.parse_args()

    if not args.dry_run and not args.skip_backup:
        backup_speeches()

    reindex(from_date=args.from_date, dry_run=args.dry_run, limit=args.limit)
