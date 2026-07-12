"""High-level orchestration — the public ``run_etl`` and ``run_historical_backfill`` functions."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from ..config import RIKSDAG_BASE_URL
from ..db import insert_speeches
from ..logger import get_logger
from .extract import extract_all_tweets, extract_riksdag_protocols
from .load import load_riksdag, load_tweets
from .transform import _process_doc, transform_riksdag

logger = get_logger()


def run_etl(lookback_days: int = 7) -> None:
    """Run the full incremental ETL pipeline.

    Fetches Riksdag protocols from the last *lookback_days* days, parses the
    PDFs, upserts speeches to Supabase, then collects recent tweets.

    Args:
        lookback_days: How many days back to fetch protocols (default: 7).
    """
    logger.info("=" * 55)
    logger.info("MaktspråkAI ETL — starting incremental run")
    logger.info("=" * 55)

    xml_file = extract_riksdag_protocols(lookback_days=lookback_days)
    speeches = transform_riksdag(xml_file)
    if speeches:
        load_riksdag(speeches)
    else:
        logger.info("No new speeches to load.")

    tweets = extract_all_tweets()
    if tweets:
        load_tweets(tweets)
    else:
        logger.info("No new tweets to load.")

    logger.info("=" * 55)
    logger.info("ETL run complete.")
    logger.info("=" * 55)


def fetch_protocol_docs(from_date: str, to_date: str | None = None) -> list[ET.Element]:
    """Paginate the Riksdag API and return every protocol ``<dokument>`` element.

    Args:
        from_date: ISO date string for the start of the window (e.g. ``"2015-06-01"``).
        to_date:   ISO date string for the end of the window.  Defaults to today.

    Returns:
        All ``<dokument>`` elements in the window, ascending by date.
    """
    from datetime import datetime

    to_date = to_date or datetime.today().strftime("%Y-%m-%d")
    all_docs: list[ET.Element] = []
    page = 1

    while True:
        url = (
            f"{RIKSDAG_BASE_URL}"
            f"?sok=&doktyp=prot&from={from_date}&tom={to_date}"
            f"&utformat=xml&sort=datum&sortorder=asc&p={page}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        docs = root.findall(".//dokument")
        if not docs:
            break

        all_docs.extend(docs)
        total = int(root.get("traffar", 0))
        logger.info(f"Page {page}: {len(docs)} protocols (total so far: {len(all_docs)}/{total})")

        if total and len(all_docs) >= total:
            break
        page += 1
        time.sleep(1)

    return all_docs


def run_historical_backfill(from_date: str, to_date: str | None = None) -> None:
    """Back-fill Riksdag protocols over a date window.

    Paginates the Riksdag API, downloads and parses each protocol PDF, then
    **batch-upserts** each protocol's speeches to Supabase (one request per
    protocol instead of one per row).  Every upsert is keyed on the row ``id``,
    so a re-run is idempotent and the job can be safely restarted after an
    interruption.  A failure on any single protocol is logged and skipped so a
    long unattended run cannot be aborted by one bad PDF.

    Args:
        from_date: ISO date string for the start of the window (e.g. ``"2002-09-01"``).
        to_date:   ISO date string for the end of the window.  Defaults to today.
            Bound this to avoid re-processing an already-ingested range.
    """
    from datetime import datetime

    to_date = to_date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Starting historical backfill: {from_date} -> {to_date}")

    all_docs = fetch_protocol_docs(from_date, to_date)
    logger.info(f"Fetched {len(all_docs)} protocols. Starting PDF extraction...")

    total_speeches = 0
    failures = 0
    for i, doc in enumerate(tqdm(all_docs, desc="Backfilling protocols"), start=1):
        try:
            speeches = _process_doc(doc)
            if speeches:
                inserted = insert_speeches(speeches)
                total_speeches += inserted
        except Exception as exc:  # noqa: BLE001 — one bad PDF must not abort the run
            failures += 1
            logger.warning(f"Backfill skipped a protocol ({exc}).")
        time.sleep(0.3)  # be polite to the Riksdag file host; cuts connection resets
        if i % 50 == 0:
            logger.info(
                f"Progress: {i}/{len(all_docs)} protocols, "
                f"{total_speeches} speeches upserted, {failures} skipped."
            )

    logger.info(
        f"Historical backfill complete: {total_speeches} speeches from "
        f"{len(all_docs)} protocols ({failures} skipped)."
    )
