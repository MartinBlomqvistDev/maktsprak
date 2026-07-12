"""Back-fill Riksdag speeches over a date window.

Usage::

    python scripts/backfill_speeches.py [from_date] [to_date]

``from_date`` / ``to_date`` are ISO date strings (e.g. ``2002-09-01``).
``from_date`` defaults to the start of the 2025/26 parliamentary session;
``to_date`` defaults to today.  Bound ``to_date`` to avoid re-processing an
already-ingested range (e.g. backfill only the 2002 -> 2015 gap).

The script paginates through the Riksdag API, downloads each protocol PDF,
extracts speech segments, and batch-upserts them to Supabase.  Every upsert is
keyed on the row id, so a re-run (or a restart after interruption) is safe:
already-cached PDFs are reused and existing rows are simply overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when the script is run directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.maktsprak_pipeline.pipeline.orchestrate import run_historical_backfill

if __name__ == "__main__":
    _from_date = sys.argv[1] if len(sys.argv) > 1 else "2025-09-15"
    _to_date = sys.argv[2] if len(sys.argv) > 2 else None
    run_historical_backfill(_from_date, _to_date)
