"""Back-fill Riksdag speeches from a given start date to today.

Usage::

    python scripts/backfill_speeches.py [from_date]

``from_date`` is an ISO date string (e.g. ``2025-09-15``).  Defaults to the
start of the 2025/26 parliamentary session if omitted.

The script paginates through the Riksdag API, downloads each protocol PDF,
extracts speech segments, and upserts them to Supabase.  Already-cached PDFs
are reused so a re-run is safe and fast.
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
    run_historical_backfill(_from_date)
