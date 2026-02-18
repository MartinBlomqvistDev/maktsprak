"""Extraction phase — pull raw data from the Riksdag API and Twitter/X API."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..config import (
    MAX_TWEETS_PER_PARTY,
    MONTHLY_TWEET_LIMIT,
    PARTY_LEADERS_IDS,
    RATE_LIMIT_WAIT_SECONDS,
    RAW_DATA_PATH,
    RIKSDAG_BASE_URL,
    X_BASE_URL,
    X_BEARER_TOKEN,
)
from ..db import fetch_tweets_count_since
from ..logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def fetch_with_retry(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
) -> requests.Response | None:
    """GET a URL, retrying on HTTP 429 with a fixed back-off.

    Args:
        url:         Fully-formed request URL.
        headers:     HTTP headers (e.g. Authorization).
        params:      Query-string parameters.
        max_retries: Total attempts before giving up.

    Returns:
        The successful :class:`requests.Response`, or ``None`` on failure.
    """
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            logger.warning(
                f"Rate limit hit on attempt {attempt + 1}/{max_retries} — "
                f"waiting {RATE_LIMIT_WAIT_SECONDS}s."
            )
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
            continue
        logger.error(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
        return None

    logger.error(f"Gave up after {max_retries} attempts: {url}")
    return None


# ---------------------------------------------------------------------------
# Riksdag extraction
# ---------------------------------------------------------------------------

def extract_riksdag_protocols(
    lookback_days: int = 7,
    max_back: int = 90,
) -> str | None:
    """Fetch the Riksdag protocol listing for the last *n* days and save it to disk.

    Steps back one week at a time if no documents are found, up to *max_back*
    days.  This handles parliamentary recesses without requiring manual
    configuration.

    Args:
        lookback_days: Starting window size (days back from today).
        max_back:      Maximum days to look back before aborting.

    Returns:
        Absolute path to the saved XML file, or ``None`` if no protocols found.
    """
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    today = datetime.today()

    while lookback_days <= max_back:
        start_date = today - timedelta(days=lookback_days)
        url = (
            f"{RIKSDAG_BASE_URL}"
            f"?sok=&doktyp=prot"
            f"&from={start_date.strftime('%Y-%m-%d')}"
            f"&tom={today.strftime('%Y-%m-%d')}"
            f"&utformat=xml&sort=datum&sortorder=desc"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        doc_count = len(root.findall(".//dokument"))

        if doc_count > 0:
            logger.info(
                f"Found {doc_count} protocols from {start_date.date()} to {today.date()}."
            )
            filename = (
                RAW_DATA_PATH
                / f"protocols_{start_date.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}.xml"
            )
            filename.write_bytes(resp.content)
            logger.info(f"Protocol listing saved: {filename}")
            return str(filename)

        logger.info("No protocols found — stepping back one week.")
        lookback_days += 7

    logger.warning(f"No protocols found within {max_back} days. Aborting.")
    return None


# ---------------------------------------------------------------------------
# Twitter / X extraction
# ---------------------------------------------------------------------------

def extract_all_tweets() -> list[dict[str, Any]]:
    """Fetch recent tweets for each party leader, respecting monthly limits.

    Checks how many tweets have already been collected this calendar month and
    fetches only as many as remain under :data:`~config.MONTHLY_TWEET_LIMIT`.

    Returns:
        List of tweet dicts ready for :func:`~.load.load_tweets`.
    """
    if not X_BEARER_TOKEN:
        logger.warning("X_BEARER_TOKEN not set — skipping tweet extraction.")
        return []

    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=7)

    first_of_month = datetime(end_time.year, end_time.month, 1, tzinfo=timezone.utc)
    already_fetched = fetch_tweets_count_since(first_of_month.isoformat())
    tweets_remaining = max(0, MONTHLY_TWEET_LIMIT - already_fetched)

    if tweets_remaining == 0:
        logger.info("Monthly tweet limit reached — skipping fetch.")
        return []

    tweets_per_party: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for party, user_ids in PARTY_LEADERS_IDS.items():
        if tweets_remaining <= 0:
            break
        max_for_party = min(MAX_TWEETS_PER_PARTY, tweets_remaining)

        for user_id in user_ids:
            if len(tweets_per_party[party]) >= max_for_party:
                break

            params: dict[str, Any] = {
                "max_results": 5,
                "start_time": start_time.isoformat().replace("+00:00", "Z"),
                "end_time": end_time.isoformat().replace("+00:00", "Z"),
                "tweet.fields": "id,text,created_at,lang",
            }
            url = f"{X_BASE_URL}/{user_id}/tweets"
            resp = fetch_with_retry(url, headers, params)
            if not resp:
                continue

            raw = sorted(
                resp.json().get("data", []),
                key=lambda x: x["created_at"],
                reverse=True,
            )
            for tweet in raw:
                if len(tweets_per_party[party]) >= max_for_party:
                    break
                tweets_per_party[party].append(
                    {
                        "tweet_id": tweet["id"],
                        "created_at": tweet["created_at"],
                        "username": user_id,
                        "lang": tweet.get("lang", "NA").upper(),
                        "text": tweet["text"].strip(),
                        "url": f"https://x.com/i/web/status/{tweet['id']}",
                    }
                )

        party_count = len(tweets_per_party[party])
        tweets_remaining -= party_count
        logger.info(
            f"{party}: fetched {party_count} tweet(s). "
            f"{tweets_remaining} remaining this month."
        )

    all_tweets = [t for tweets in tweets_per_party.values() for t in tweets]
    logger.info(f"Total tweets fetched: {len(all_tweets)}")
    return all_tweets
