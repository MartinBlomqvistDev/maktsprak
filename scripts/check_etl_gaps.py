"""Check that the weekly ETL has stored every recent protocol with party speeches.

Run daily by ``.github/workflows/etl-gap-check.yml``.  The ETL runs once a week
on one computer, and a run that fails there shows only in its Task Scheduler.
This check compares outcomes instead: every protocol from the last 30 days that
the Riksdag published more than eight days ago, with at least one speech by a
party member, must have rows in ``speeches``.  Eight days is the week between
runs plus one day.  The check exits non-zero when a protocol is missing, and
GitHub emails about the failed run.  Going protocol by protocol, it also finds
holes behind newer protocols, such as HE0913 and HE0914.

Standard library only, so the workflow installs nothing.  Reads the project
URL and the read-only anon key from ``SUPABASE_URL`` and ``SUPABASE_ANON_KEY``.

Usage::

    python scripts/check_etl_gaps.py [--lookback-days 30] [--grace-days 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

#: A copy of ``VALID_PARTIES`` in ``src/maktsprak_pipeline/config.py``, so that
#: the check needs no install.  A test keeps the two equal.
VALID_PARTIES = frozenset({"C", "KD", "L", "M", "MP", "S", "SD", "V"})

RIKSDAG = "https://data.riksdagen.se"
LOOKBACK_DAYS = 30
GRACE_DAYS = 8
#: The speech list has no paging, so it is fetched newest first in batches
#: that double from the first size up to the largest.
FIRST_SPEECH_BATCH = 2000
MAX_SPEECH_BATCH = 32000


def _get(url: str, headers: dict[str, str] | None = None, attempts: int = 3) -> bytes:
    """GET *url* and return the body, trying up to *attempts* times."""
    attempt = 1
    while True:
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as exc:  # URLError and HTTPError are both OSErrors
            if attempt >= attempts:
                raise RuntimeError(f"GET failed after {attempts} attempts: {url}") from exc
            time.sleep(10 * attempt)
            attempt += 1


def count_party_speeches(items: list[dict], start: date) -> dict[str, tuple[date, int]]:
    """Map each protocol dated *start* or later to its date and its party speeches.

    Args:
        items: Speeches from the Riksdag's ``anforandelista``.
        start: The first protocol date to count.

    Returns:
        ``{dok_id: (protocol date, speeches by party members)}``.  Protocols
        with no speech by a party member, such as those with only the
        Speaker's, are left out.
    """
    counts: dict[str, tuple[date, int]] = {}
    for item in items:
        day = date.fromisoformat(item["dok_datum"][:10])
        if day < start or (item.get("parti") or "").upper() not in VALID_PARTIES:
            continue
        _, seen = counts.get(item["dok_id"], (day, 0))
        counts[item["dok_id"]] = (day, seen + 1)
    return counts


def party_speeches(start: date) -> dict[str, tuple[date, int]]:
    """Fetch the speeches back to *start* and count them per protocol.

    The list comes newest first.  Every protocol from *start* on is complete
    in it only once the oldest speech returned is older than *start*, so the
    batch doubles until it is.

    Raises:
        RuntimeError: If even the largest batch does not reach back to *start*.
    """
    size = FIRST_SPEECH_BATCH
    while True:
        query = urllib.parse.urlencode(
            {
                "rm": "",
                "anftyp": "",
                "d": "",
                "ts": "",
                "parti": "",
                "iid": "",
                "sz": size,
                "utformat": "json",
            }
        )
        payload = json.loads(_get(f"{RIKSDAG}/anforandelista/?{query}"))
        items = (payload.get("anforandelista") or {}).get("anforande") or []
        if isinstance(items, dict):  # a single speech comes back as an object
            items = [items]
        if items and min(item["dok_datum"][:10] for item in items) < start.isoformat():
            return count_party_speeches(items, start)
        if size >= MAX_SPEECH_BATCH:
            raise RuntimeError(f"The {size} newest speeches do not reach back to {start}.")
        size *= 2


def published_protocols(start: date, today: date) -> dict[str, date]:
    """Map each protocol dated from *start* to *today* to the day it was published."""
    published: dict[str, date] = {}
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "sok": "",
                "doktyp": "prot",
                "from": start.isoformat(),
                "tom": today.isoformat(),
                "utformat": "xml",
                "sort": "datum",
                "sortorder": "asc",
                "p": page,
            }
        )
        root = ET.fromstring(_get(f"{RIKSDAG}/dokumentlista/?{query}"))
        docs = root.findall(".//dokument")
        if not docs:
            return published
        for doc in docs:
            day = doc.findtext("publicerad") or doc.findtext("datum")
            if day:
                published[doc.findtext("dok_id") or ""] = date.fromisoformat(day[:10])
        if len(published) >= int(root.get("traffar", 0)):
            return published
        page += 1


def overdue(
    counts: dict[str, tuple[date, int]],
    published: dict[str, date],
    today: date,
    grace_days: int = GRACE_DAYS,
) -> list[str]:
    """Return the protocols with party speeches published more than *grace_days* ago.

    A protocol missing from the listing is dated by its speeches instead.
    """
    return sorted(
        protocol
        for protocol, (day, _) in counts.items()
        if (today - published.get(protocol, day)).days > grace_days
    )


def stored(protocols: list[str], url: str, key: str) -> set[str]:
    """Return the protocols among *protocols* that have rows in ``speeches``."""
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    found: set[str] = set()
    for protocol in protocols:
        query = urllib.parse.urlencode(
            {"select": "protocol_id", "protocol_id": f"eq.{protocol}", "limit": 1}
        )
        if json.loads(_get(f"{url}/rest/v1/speeches?{query}", headers)):
            found.add(protocol)
    return found


def main() -> int:
    """Run the check, print a table and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--grace-days", type=int, default=GRACE_DAYS)
    args = parser.parse_args()

    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_ANON_KEY"]
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=args.lookback_days)

    counts = party_speeches(start)
    published = published_protocols(start, today)
    due = overdue(counts, published, today, args.grace_days)
    have = stored(due, url, key)
    missing = [protocol for protocol in due if protocol not in have]

    rows = [
        f"| {p} | {published.get(p, counts[p][0])} | {counts[p][1]} | "
        f"{'yes' if p in have else 'MISSING'} |"
        for p in due
    ]
    report = "\n".join(
        [
            f"{len(due)} protocols with party speeches, dated {start} or later and published "
            f"more than {args.grace_days} days ago. Missing from speeches: {len(missing)}.",
            "",
            "| Protocol | Published | Party speeches | In speeches |",
            "|---|---|---|---|",
            *rows,
        ]
    )
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")

    if not missing:
        return 0
    days = sorted(counts[p][0] for p in missing)
    print(
        f"::error::speeches lacks {len(missing)} protocol(s) with party speeches: "
        f"{', '.join(missing)}. Backfill them with: "
        f"python scripts/backfill_speeches.py {days[0]} {days[-1]}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
