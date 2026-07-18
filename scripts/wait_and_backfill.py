"""Wait for the Riksdag host to recover, then run the 2002->2015 backfill.

data.riksdagen.se rate-limits/blocks the IP after heavy use.  This poller checks
connectivity gently (one request every 10 minutes, never hammering, which would
only prolong a block) and launches the historical backfill the moment the host
responds.  If the backfill aborts on a transient host failure it goes back to
waiting and tries again.  The backfill itself is idempotent, so restarts are safe.

Usage::

    python scripts/wait_and_backfill.py [from_date] [to_date]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv()

from src.maktsprak_pipeline.pipeline.orchestrate import run_historical_backfill

FROM_DATE = sys.argv[1] if len(sys.argv) > 1 else "2002-09-01"
TO_DATE = sys.argv[2] if len(sys.argv) > 2 else "2015-06-30"

# A tiny, cheap query used purely as a reachability probe.
CHECK_URL = (
    "https://data.riksdagen.se/dokumentlista/"
    "?sok=&doktyp=prot&from=2024-02-01&tom=2024-02-03&utformat=xml&p=1"
)
POLL_SECONDS = 600  # 10 minutes between gentle checks
MAX_WAITS = 30  # give up after ~5 hours of the host refusing


def _reachable() -> bool:
    try:
        return requests.get(CHECK_URL, timeout=30).status_code == 200
    except requests.exceptions.RequestException:
        return False


def main() -> None:
    for attempt in range(1, MAX_WAITS + 1):
        stamp = time.strftime("%Y-%m-%d %H:%M")
        if _reachable():
            print(
                f"[{stamp}] Riksdag reachable, starting backfill {FROM_DATE} -> {TO_DATE}.",
                flush=True,
            )
            try:
                run_historical_backfill(FROM_DATE, TO_DATE)
                print(f"[{time.strftime('%Y-%m-%d %H:%M')}] Backfill finished.", flush=True)
                return
            except Exception as exc:  # noqa: BLE001, host flaked mid-run; wait and retry
                print(
                    f"[{stamp}] Backfill aborted ({exc}); will retry after {POLL_SECONDS // 60} min.",
                    flush=True,
                )
        else:
            print(
                f"[{stamp}] Riksdag still refusing (check {attempt}/{MAX_WAITS}); waiting {POLL_SECONDS // 60} min.",
                flush=True,
            )
        time.sleep(POLL_SECONDS)

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M')}] Gave up: host refused for the whole window.",
        flush=True,
    )


if __name__ == "__main__":
    main()
