"""Extraction phase, pull raw data from the Riksdag API and Twitter/X API."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any, TypeVar

import requests

from ..config import (
    MAX_TWEETS_PER_PARTY,
    MONTHLY_TWEET_LIMIT,
    PARTY_LEADERS_IDS,
    RATE_LIMIT_WAIT_SECONDS,
    RAW_DATA_PATH,
    RETRY_ATTEMPTS,
    RETRY_FIRST_WAIT_SECONDS,
    RIKSDAG_BASE_URL,
    X_BASE_URL,
    X_BEARER_TOKEN,
)
from ..db import fetch_tweets_count_since
from ..logger import get_logger

logger = get_logger()

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def call_with_backoff(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    what: str,
    attempts: int = RETRY_ATTEMPTS,
    first_wait: float = RETRY_FIRST_WAIT_SECONDS,
) -> T:
    """Call *fn*, retrying the given errors with a wait that doubles each time.

    Any other error propagates at once, so a bug is not retried as if it were
    the network.

    Args:
        fn:         The call to make, without arguments.
        retry_on:   Exception types that count as transient.
        what:       What the call does, for the log and the final error.
        attempts:   Total attempts before giving up.
        first_wait: Seconds before the second attempt.

    Returns:
        Whatever *fn* returns.

    Raises:
        RuntimeError: If every attempt fails, chained to the last error.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except retry_on as exc:
            if attempt >= attempts:
                raise RuntimeError(f"{what} failed after {attempts} attempts") from exc
            wait = first_wait * 2 ** (attempt - 1)
            logger.warning(
                f"{what} failed on attempt {attempt}/{attempts} ({exc}), retrying in {wait:g} s."
            )
            time.sleep(wait)
            attempt += 1


def get_with_backoff(url: str, timeout: float = 60) -> requests.Response:
    """GET *url*, retrying failed connections and error statuses.

    The Riksdag host drops connections now and then (``RemoteDisconnected``,
    ``ConnectionResetError``), and scheduled runs have died on a name lookup
    that failed (``getaddrinfo``).  Without retries one such failure ended the
    whole run.

    Args:
        url:     The URL to fetch.
        timeout: Per-attempt timeout in seconds.

    Returns:
        The successful :class:`requests.Response`.

    Raises:
        RuntimeError: If every attempt fails.
    """

    def attempt() -> requests.Response:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    return call_with_backoff(
        attempt, retry_on=(requests.exceptions.RequestException,), what=f"GET {url}"
    )


# ---------------------------------------------------------------------------
# Twitter / X HTTP helper
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
                f"Rate limit hit on attempt {attempt + 1}/{max_retries}, "
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


def fetch_protocol_docs(from_date: str, to_date: str | None = None) -> list[ET.Element]:
    """List every protocol dated within a window, oldest first.

    Pages through the Riksdag's document list, 20 protocols a page, retrying
    each page.  Every page is saved under ``data/raw`` as it arrives, because
    ``scripts/rebuild_corpus.py`` takes each protocol's date from those files:
    the PDFs carry no machine-readable date.

    Args:
        from_date: ISO date of the first protocol day (e.g. ``"2026-11-11"``).
        to_date:   ISO date of the last.  Defaults to today.

    Returns:
        All ``<dokument>`` elements in the window, ascending by date.
    """
    to_date = to_date or date.today().isoformat()
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    stem = f"protocols_{from_date.replace('-', '')}_{to_date.replace('-', '')}"
    all_docs: list[ET.Element] = []
    page = 1

    while True:
        url = (
            f"{RIKSDAG_BASE_URL}"
            f"?sok=&doktyp=prot&from={from_date}&tom={to_date}"
            f"&utformat=xml&sort=datum&sortorder=asc&p={page}"
        )
        resp = get_with_backoff(url, timeout=30)
        root = ET.fromstring(resp.content)

        docs = root.findall(".//dokument")
        if not docs:
            break

        saved = RAW_DATA_PATH / (f"{stem}.xml" if page == 1 else f"{stem}_p{page}.xml")
        saved.write_bytes(resp.content)
        all_docs.extend(docs)
        total = int(root.get("traffar", 0))
        logger.info(
            f"Page {page}: {len(docs)} protocols (total so far: {len(all_docs)}/{total}), "
            f"saved to {saved.name}."
        )

        if total and len(all_docs) >= total:
            break
        page += 1
        time.sleep(1)

    return all_docs


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
        logger.warning("X_BEARER_TOKEN not set, skipping tweet extraction.")
        return []

    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=7)

    first_of_month = datetime(end_time.year, end_time.month, 1, tzinfo=timezone.utc)
    already_fetched = fetch_tweets_count_since(first_of_month.isoformat())
    tweets_remaining = max(0, MONTHLY_TWEET_LIMIT - already_fetched)

    if tweets_remaining == 0:
        logger.info("Monthly tweet limit reached, skipping fetch.")
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
            f"{party}: fetched {party_count} tweet(s). {tweets_remaining} remaining this month."
        )

    all_tweets = [t for tweets in tweets_per_party.values() for t in tweets]
    logger.info(f"Total tweets fetched: {len(all_tweets)}")
    return all_tweets
