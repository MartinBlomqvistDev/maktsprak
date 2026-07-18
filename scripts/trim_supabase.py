"""Trim the Supabase table back under the free-tier cap after archiving to Parquet.

The full corpus lives in ``data/parquet/speeches_full.parquet`` (source of truth
for analysis/precompute).  Supabase only needs to hold recent speeches as the
weekly-ETL landing zone.  This deletes every speech older than ``--cutoff`` by
primary key (indexed, fast, survives the tightened statement timeout), taking the
project back under 0.5 GB.

Safety: refuses to run unless the Parquet archive exists and actually contains the
ids being deleted, so nothing is removed that isn't already backed up.  Deletes by
id are idempotent, so a re-run after interruption is safe.

Usage::

    python scripts/trim_supabase.py [--cutoff 2015-06-01] [--archive data/parquet/speeches_full.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.maktsprak_pipeline.db.client import supabase_write
from src.maktsprak_pipeline.logger import get_logger

logger = get_logger()
DELETE_BATCH = 100


def trim(cutoff: str, archive: Path) -> None:
    if not archive.exists():
        logger.error(f"Archive {archive} not found, refusing to delete anything.")
        sys.exit(1)

    df = pd.read_parquet(archive, columns=["id", "protocol_date"])
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    doomed = df.loc[df["protocol_date"] < pd.to_datetime(cutoff), "id"].tolist()
    logger.info(
        f"Archive holds {len(df)} rows; {len(doomed)} are older than {cutoff} and "
        f"backed up, so safe to delete from Supabase."
    )
    if not doomed:
        logger.info("Nothing to delete.")
        return

    deleted = 0
    for i in range(0, len(doomed), DELETE_BATCH):
        batch = doomed[i : i + DELETE_BATCH]
        supabase_write.table("speeches").delete().in_("id", batch).execute()
        deleted += len(batch)
        if (i // DELETE_BATCH) % 20 == 0:
            logger.info(f"Deleted {deleted}/{len(doomed)}...")

    logger.info(f"Trim complete: removed {deleted} rows older than {cutoff} from Supabase.")
    print(f"TRIM_DONE deleted={deleted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim Supabase back under the free-tier cap.")
    parser.add_argument(
        "--cutoff", default="2015-06-01", help="Delete speeches strictly before this date."
    )
    parser.add_argument("--archive", type=Path, default=Path("data/parquet/speeches_full.parquet"))
    args = parser.parse_args()
    trim(args.cutoff, args.archive)


if __name__ == "__main__":
    main()
