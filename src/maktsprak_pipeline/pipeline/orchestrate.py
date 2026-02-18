"""High-level orchestration — the public ``run_etl`` and ``run_historical_backfill`` functions."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from ..config import RIKSDAG_BASE_URL
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


def run_historical_backfill(from_date: str) -> None:
    """Back-fill all Riksdag protocols from *from_date* to today.

    Paginates through the Riksdag API (20 results per page), downloads and
    parses each protocol PDF, then upserts every speech to Supabase.

    Args:
        from_date: ISO date string for the start of the backfill window
            (e.g. ``"2025-09-15"``).
    """
    from datetime import datetime

    to_date = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Starting historical backfill: {from_date} → {to_date}")

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

    logger.info(f"Fetched {len(all_docs)} protocols. Starting PDF extraction…")

    total_speeches = 0
    for doc in tqdm(all_docs, desc="Backfilling protocols"):
        speeches = _process_doc(doc)
        if speeches:
            load_riksdag(speeches)
            total_speeches += len(speeches)

    logger.info(
        f"Historical backfill complete: {total_speeches} speeches from {len(all_docs)} protocols."
    )
