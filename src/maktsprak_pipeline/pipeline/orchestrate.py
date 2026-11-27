"""High-level orchestration, the public ``run_etl`` and ``run_historical_backfill`` functions."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import httpx
from tqdm import tqdm

from ..config import TWEETS_ENABLED, WATERMARK_MARGIN_DAYS
from ..db import fetch_latest_speech_date, insert_speeches
from ..logger import get_logger
from .extract import call_with_backoff, extract_all_tweets, fetch_protocol_docs
from .load import load_tweets
from .transform import _process_doc

logger = get_logger()


def listing_start(margin_days: int = WATERMARK_MARGIN_DAYS, today: date | None = None) -> date:
    """Return the first protocol date the incremental run lists.

    The watermark is the newest protocol date already in ``speeches``, and the
    run lists from there, minus *margin_days*.  This replaces a fixed window of
    seven days back from today.  With that window, a week whose run failed was
    never listed again.  Now the next run lists everything since the last
    stored protocol, however many runs failed in between, and the margin gives
    a protocol that failed while newer ones loaded another chance.  The upsert
    is keyed on ``id``, so the overlap rewrites rows rather than duplicating
    them.

    Args:
        margin_days: Days before the watermark to start listing.
        today:       The day the run lists up to; defaults to today.  Used only
            when ``speeches`` is empty.

    Returns:
        The first protocol date to list.
    """
    latest = call_with_backoff(
        fetch_latest_speech_date,
        retry_on=(httpx.TransportError,),
        what="Reading the newest protocol date in speeches",
    )
    if latest is None:
        start = (today or date.today()) - timedelta(days=margin_days)
        logger.warning(f"speeches is empty, listing from {start} ({margin_days} days back).")
        return start

    watermark = date.fromisoformat(str(latest)[:10])
    start = watermark - timedelta(days=margin_days)
    logger.info(f"Watermark: the newest protocol in speeches is from {watermark}.")
    return start


def _ingest(docs: list[ET.Element]) -> tuple[int, list[str]]:
    """Parse and upsert each protocol in turn, one request per protocol.

    A failure on one protocol is logged and the rest still load, so one bad
    PDF cannot cost a whole run.

    Args:
        docs: ``<dokument>`` elements from :func:`~.extract.fetch_protocol_docs`.

    Returns:
        Tuple of ``(speeches_upserted, failed_protocol_ids)``.
    """
    upserted = 0
    failed: list[str] = []
    for i, doc in enumerate(tqdm(docs, desc="Protocols"), start=1):
        protocol_id = doc.findtext("dok_id") or "unknown"
        try:
            upserted += insert_speeches(_process_doc(doc))
        except Exception as exc:  # noqa: BLE001, one bad PDF must not abort the run
            failed.append(protocol_id)
            logger.error(f"Protocol {protocol_id} failed ({exc}).")
        time.sleep(0.3)  # be polite to the Riksdag file host; cuts connection resets
        if i % 50 == 0:
            logger.info(
                f"Progress: {i}/{len(docs)} protocols, "
                f"{upserted} speeches upserted, {len(failed)} failed."
            )
    return upserted, failed


def run_etl(margin_days: int = WATERMARK_MARGIN_DAYS) -> None:
    """Run the full incremental ETL pipeline.

    Lists the Riksdag protocols from :func:`listing_start` to today, parses
    each PDF and upserts its speeches to Supabase.  Then it collects recent
    tweets, if :data:`~config.TWEETS_ENABLED` is set.

    Args:
        margin_days: Days before the watermark to start listing.

    Raises:
        RuntimeError: If any protocol failed.  The others are loaded first.
            The exception makes the process exit non-zero, which the Task
            Scheduler shows as a failed run.
    """
    logger.info("=" * 55)
    logger.info("MaktspråkAI ETL, starting incremental run")
    logger.info("=" * 55)

    start = listing_start(margin_days)
    docs = fetch_protocol_docs(start.isoformat())
    logger.info(f"Found {len(docs)} protocols from {start} to today.")
    upserted, failed = _ingest(docs)
    logger.info(f"{upserted} speech record(s) upserted from {len(docs) - len(failed)} protocols.")

    if not TWEETS_ENABLED:
        logger.info("The X step is off; set TWEETS_ENABLED=true to run it.")
    elif tweets := extract_all_tweets():
        load_tweets(tweets)
    else:
        logger.info("No new tweets to load.")

    if failed:
        raise RuntimeError(f"{len(failed)} of {len(docs)} protocols failed: {', '.join(failed)}.")

    logger.info("=" * 55)
    logger.info("ETL run complete.")
    logger.info("=" * 55)


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
    to_date = to_date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Starting historical backfill: {from_date} -> {to_date}")

    all_docs = fetch_protocol_docs(from_date, to_date)
    logger.info(f"Fetched {len(all_docs)} protocols. Starting PDF extraction...")

    total_speeches, failed = _ingest(all_docs)

    logger.info(
        f"Historical backfill complete: {total_speeches} speeches from "
        f"{len(all_docs)} protocols ({len(failed)} skipped"
        f"{': ' + ', '.join(failed) if failed else ''})."
    )
