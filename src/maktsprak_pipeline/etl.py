# =========================================================
# File: src/maktsprak_pipeline/etl.py
# Purpose: ETL pipeline for MaktsprakAI
#   - Riksdag (heavy PDF/protocol parsing)
#   - Tweets (small batches, last 7 days)
# Dependencies:
#   - pandas, pdfplumber, requests, tqdm, xml.etree.ElementTree
# =========================================================

import os
import re
import requests
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict
from pathlib import Path

import pdfplumber
import xml.etree.ElementTree as ET
from tqdm import tqdm

from .db import insert_speech, insert_tweet, fetch_tweets_count_since, delete_speeches_invalid_parties
from .config import RAW_DATA_PATH, X_API_KEY
from .logger import get_logger

logger = get_logger()

# =========================================================
# Configuration
# =========================================================

PARTY_LEADERS_IDS = {
    "S": ["1587012835409788928"],
    "M": ["747426555417198592"],
    "V": ["282532238"],
    "L": ["455193032"],
    "KD": ["1407151866"],
    "C": ["232799403"],
    "MP": ["41214271", "370900852"],
    "SD": ["95972673"]
}

VALID_PARTIES = set(PARTY_LEADERS_IDS.keys())
MAX_TWEETS_PER_PARTY = 2
MONTHLY_TWEET_LIMIT = 100

# =========================================================
# Helpers
# =========================================================

def fetch_with_retry(url: str, headers: Dict, params: Dict = None, max_retries: int = 5):
    """Fetches data from an API with retry on rate limit (429)."""
    wait_time = 900  # 15 minutes
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
        logger.error(f"API error {resp.status_code}: {resp.text}")
        return None
    logger.error(f"Failed after {max_retries} attempts: {url}")
    return None

# =========================================================
# Riksdag: Extract → Transform → Load
# =========================================================

def extract_riksdag_protocols(lookback_days: int = 7, max_back: int = 90) -> str:
    today = datetime.today()
    found_docs = False
    start_date = today
    while not found_docs:
        start_date = today - timedelta(days=lookback_days)
        api_url = (
            "https://data.riksdagen.se/dokumentlista/"
            "?sok=&doktyp=prot"
            f"&from={start_date.strftime('%Y-%m-%d')}&tom={today.strftime('%Y-%m-%d')}"
            "&utformat=xml&sort=datum&sortorder=desc"
        )
        resp = requests.get(api_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        doc_count = len(root.findall(".//dokument"))

        if doc_count > 0:
            found_docs = True
            logger.info(f"Found {doc_count} protocols from {start_date.date()} to {today.date()}")
        else:
            logger.info("No protocols found, stepping back 1 week")
            lookback_days += 7
            if lookback_days > max_back:
                logger.warning("No protocols in 3 months, aborting")
                return None

        filename = os.path.join(
            RAW_DATA_PATH,
            f"protocols_{start_date.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}.xml"
        )
        with open(filename, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved protocol list: {filename}")
        return filename

_regex_speech = re.compile(r"Anf\.\s+\d+\s+(.*?)\s+\(([A-ZÅÄÖ]{1,2})\):(.*?)(?=Anf\.|\Z)", re.S)

def _process_doc(doc) -> List[Dict]:
    """Process a single riksdag protocol XML element into a list of speech dicts."""
    data = []
    protocol_id = doc.findtext("dok_id")
    protocol_date = doc.findtext("datum")
    file_url = doc.findtext("filbilaga/fil/url")
    if not file_url:
        return []

    pdf_path = Path(RAW_DATA_PATH) / f"{protocol_id}.pdf"
    if not pdf_path.exists():
        resp_pdf = requests.get(file_url)
        resp_pdf.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(resp_pdf.content)

    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2) or ""
                page_text = re.sub(r'-\n', '', page_text)
                page_text = re.sub(r'\n', ' ', page_text)
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'\s{2,}', ' ', page_text)
                text += page_text + " "

        grouped = defaultdict(list)
        for speaker, party, speech_text in _regex_speech.findall(text):
            if party in VALID_PARTIES:
                speech_text_cleaned = speech_text.strip()
                if speech_text_cleaned:
                    grouped[(speaker.strip(), party)].append(speech_text_cleaned)

        for idx, ((speaker, party), speeches) in enumerate(grouped.items(), start=1):
            speech_id = f"{protocol_id}_{idx}"
            combined_text = "\n\n".join(speeches)
            data.append({
                "id": speech_id,
                "protocol_id": protocol_id,
                "protocol_date": protocol_date,
                "speaker": speaker,
                "party": party,
                "text": combined_text,
                "file_url": file_url
            })

    except Exception as e:
        logger.warning(f"Failed to read PDF {protocol_id}: {e}")

    return data

def transform_riksdag(xml_file: str) -> List[Dict]:
    if not xml_file:
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    documents = root.findall(".//dokument")
    for doc in tqdm(documents, desc="Processing protocols"):
        data.extend(_process_doc(doc))

    logger.info(f"Total speeches transformed: {len(data)}")
    return data

def load_riksdag(data: List[Dict]):
    for row in data:
        insert_speech({
            "id": row["id"],
            "protocol_id": row["protocol_id"],
            "protocol_date": row["protocol_date"],
            "speaker": row["speaker"],
            "party": row["party"],
            "text": row["text"],
            "file_url": row["file_url"]
        })
    logger.info(f"{len(data)} speeches loaded to DB")

# =========================================================
# Tweets: Extract → Load
# =========================================================

def extract_all_tweets() -> List[Dict]:
    headers = {"Authorization": f"Bearer {X_API_KEY}"}
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=7)

    first_of_month = datetime(end_time.year, end_time.month, 1, tzinfo=timezone.utc)
    already_fetched = fetch_tweets_count_since(first_of_month.isoformat())

    tweets_remaining = max(0, MONTHLY_TWEET_LIMIT - already_fetched)
    if tweets_remaining == 0:
        logger.info("Monthly tweet limit reached, skipping fetch")
        return []

    tweets_per_party = defaultdict(list)
    for party in tqdm(PARTY_LEADERS_IDS.keys(), desc="Fetching tweets by party"):
        if tweets_remaining <= 0:
            break
        user_ids = PARTY_LEADERS_IDS[party]
        max_for_party = min(MAX_TWEETS_PER_PARTY, tweets_remaining)

        for user_id in user_ids:
            if len(tweets_per_party[party]) >= max_for_party:
                break
            params = {
                "max_results": 5,
                "start_time": start_time.isoformat().replace("+00:00", "Z"),
                "end_time": end_time.isoformat().replace("+00:00", "Z"),
                "tweet.fields": "id,text,created_at,lang"
            }
            url = f"https://api.twitter.com/2/users/{user_id}/tweets"
            resp = fetch_with_retry(url, headers, params)
            if not resp:
                continue
            tweets_data = sorted(resp.json().get("data", []), key=lambda x: x["created_at"], reverse=True)
            for t in tweets_data:
                if len(tweets_per_party[party]) >= max_for_party:
                    break
                tweets_per_party[party].append({
                    "tweet_id": t["id"],
                    "created_at": t["created_at"],
                    "username": user_id,
                    "lang": t.get("lang", "NA").upper(),
                    "text": t["text"].strip(),
                    "url": f"https://x.com/i/web/status/{t['id']}"
                })

        tweets_remaining -= len(tweets_per_party[party])
        logger.info(f"{len(tweets_per_party[party])} tweets fetched for {party}, {tweets_remaining} remaining this month")

    all_tweets = [t for tweets in tweets_per_party.values() for t in tweets]
    logger.info(f"Total tweets fetched: {len(all_tweets)}")
    return all_tweets

def load_tweets(data: List[Dict]):
    for row in data:
        insert_tweet(row)
    logger.info(f"{len(data)} tweets loaded to DB (duplicates skipped)")

# =========================================================
# Run ETL
# =========================================================

def run_etl():
    logger.info("===== Starting MaktsprakAI ETL =====")

    xml_file = extract_riksdag_protocols()
    riksdag_data = transform_riksdag(xml_file)
    if riksdag_data:
        load_riksdag(riksdag_data)

    tweet_data = extract_all_tweets()
    if tweet_data:
        load_tweets(tweet_data)

    logger.info("===== ETL complete =====")

# =========================================================
# Historical backfill
# =========================================================

def run_historical_backfill(from_date: str):
    """Fetches all Riksdag protocols from from_date to today and upserts speeches to DB."""
    to_date = datetime.today().strftime('%Y-%m-%d')
    logger.info(f"Starting historical backfill: {from_date} → {to_date}")

    all_docs = []
    page = 1
    while True:
        api_url = (
            "https://data.riksdagen.se/dokumentlista/"
            f"?sok=&doktyp=prot&from={from_date}&tom={to_date}"
            f"&utformat=xml&sort=datum&sortorder=asc&p={page}"
        )
        resp = requests.get(api_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        docs = root.findall(".//dokument")
        if not docs:
            break
        all_docs.extend(docs)
        total = int(root.get("traffar", 0))
        logger.info(f"Page {page}: {len(docs)} protocols (collected {len(all_docs)}/{total})")
        if total and len(all_docs) >= total:
            break
        page += 1
        time.sleep(1)

    logger.info(f"Fetched {len(all_docs)} protocols total. Starting PDF extraction...")
    total_speeches = 0
    for doc in tqdm(all_docs, desc="Backfilling protocols"):
        speeches = _process_doc(doc)
        if speeches:
            load_riksdag(speeches)
            total_speeches += len(speeches)

    logger.info(f"Historical backfill complete: {total_speeches} speeches from {len(all_docs)} protocols")

# =========================================================
# Maintenance: Remove speeches with invalid party labels
# =========================================================

def clean_invalid_parties():
    try:
        deleted = delete_speeches_invalid_parties(VALID_PARTIES)
        if deleted > 0:
            logger.info(f"Safety clean: Deleted {deleted} speeches with invalid party from DB.")
    except Exception as e:
        logger.warning(f"Party cleanup failed: {e}")

# =========================================================
# Run manually
# =========================================================

if __name__ == "__main__":
    try:
        run_etl()
        print("ETL completed successfully!")
    except Exception as e:
        print(f"ETL failed: {e}")
        logger.exception("ETL aborted due to error")
