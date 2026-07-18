"""Export the full speech corpus from Supabase to a compressed Parquet archive.

The Vercel site never queries the database (it serves precomputed static JSON), so
holding ~80k raw speeches in a 0.5 GB hosted Postgres is the wrong shape for this
data.  This script pulls the whole corpus into a single Parquet file, which becomes
the source of truth for analysis and precompute: free, no quota, and faster to read.

Reads in small keyset pages (by primary key), which stays under Supabase's statement
timeout even while the project is throttled for exceeding its size cap.

**Deduplicates on ``id``.**  The table itself holds duplicates: the 2002-2015
historical backfill overlapped the range the weekly ETL had already ingested, so
5 425 speeches exist twice (2014 is 95% duplicated, 2002 76%, 2015 75%).  The
copies are byte-identical, so keeping the first is lossless.  This matters
beyond tidiness, a duplicated speech is counted twice in every rate denominator
and inflates the confidence of every z-score built on it.  Since this file is
the source of truth for analysis, the dedup belongs here, at the boundary.

Usage::

    python scripts/export_corpus.py [--out data/parquet/speeches_full.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.maktsprak_pipeline.db.client import supabase
from src.maktsprak_pipeline.logger import get_logger

logger = get_logger()

COLUMNS = "id, protocol_id, protocol_date, speaker, party, text, file_url"
BATCH = 200  # small pages: survives the tightened statement timeout


def export(out_path: Path) -> int:
    """Keyset-paginate the whole table and write it to *out_path* as Parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    last_id = ""
    total = 0
    page = 0

    while True:
        query = supabase.table("speeches").select(COLUMNS).order("id").limit(BATCH)
        if last_id:
            query = query.gt("id", last_id)
        resp = query.execute()
        rows = resp.data or []
        if not rows:
            break

        frames.append(pd.DataFrame(rows))
        total += len(rows)
        last_id = rows[-1]["id"]
        page += 1
        if page % 25 == 0:
            logger.info(f"Exported {total} rows...")
        if len(rows) < BATCH:
            break

    if not frames:
        logger.error("No rows exported, aborting without writing a file.")
        return 0

    df = pd.concat(frames, ignore_index=True)
    df = deduplicate(df)
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    df = df.sort_values("protocol_date").reset_index(drop=True)
    df.to_parquet(out_path, compression="zstd", index=False)

    size_mb = out_path.stat().st_size / 1_000_000
    years = f"{df['protocol_date'].min():%Y-%m-%d} .. {df['protocol_date'].max():%Y-%m-%d}"
    logger.info(f"Wrote {len(df)} rows to {out_path} ({size_mb:.1f} MB, {years}).")
    logger.info(f"Rows per party: {df['party'].value_counts().to_dict()}")
    return len(df)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows repeating an ``id`` already seen, reporting what went.

    See the module docstring: the backfill overlapped the ETL's range, so the
    table holds the same speech twice.  Reported per year rather than as a bare
    total, because the shape of the overlap is the evidence for *why* they are
    there, a flat 7% would mean something very different from "2014 is 95%
    duplicated".
    """
    duplicated = df["id"].duplicated()
    if not duplicated.any():
        logger.info("No duplicate ids.")
        return df

    dropped = df[duplicated]
    years = pd.to_datetime(dropped["protocol_date"], errors="coerce").dt.year
    logger.warning(
        f"Dropping {len(dropped)} duplicate-id rows ({len(dropped) / len(df):.1%} of the "
        f"export), the backfill overlapped the ETL's range. Per year: "
        f"{years.value_counts().sort_index().to_dict()}"
    )
    return df[~duplicated].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the speech corpus to Parquet.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/parquet/speeches_full.parquet"),
        help="Destination Parquet file.",
    )
    args = parser.parse_args()
    n = export(args.out)
    print(f"EXPORT_DONE rows={n}")


if __name__ == "__main__":
    main()
