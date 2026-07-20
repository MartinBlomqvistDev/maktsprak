"""Configuration module for the MaktspråkAI pipeline.

All constants and environment-variable bindings live here. Import from this
module instead of calling ``os.getenv`` directly in other modules.  This
creates a single source of truth that is trivial to audit and test.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (ignored when real environment variables are already set, e.g.
# on Cloud Run or in a CI environment).
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed"

#: Curated political tone lexicon (version-controlled in the repo, not fetched
#: from Hugging Face Hub).  Columns: ``ord``, ``kategori``, ``vikt``.
LEXICON_PATH: Path = PROJECT_ROOT / "data" / "lexicons" / "politisk_ton_lexikon.csv"

# ---------------------------------------------------------------------------
# Supabase credentials
# ---------------------------------------------------------------------------
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")

# ---------------------------------------------------------------------------
# PostgreSQL direct-connection (legacy / psycopg2 fallback)
# ---------------------------------------------------------------------------
DB_HOST: str | None = os.getenv("DB_HOST")
DB_NAME: str | None = os.getenv("DB_NAME")
DB_USER: str | None = os.getenv("DB_USER")
DB_PASSWORD: str | None = os.getenv("DB_PASSWORD")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))

# ---------------------------------------------------------------------------
# External data
# ---------------------------------------------------------------------------
#: Google Drive (or other) URL for the historical Parquet snapshot.
PARQUET_URL: str | None = os.getenv("PARQUET_URL")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Riksdag API
# ---------------------------------------------------------------------------
RIKSDAG_BASE_URL: str = "https://data.riksdagen.se/dokumentlista/"

# ---------------------------------------------------------------------------
# Twitter / X API
# ---------------------------------------------------------------------------
X_BASE_URL: str = "https://api.twitter.com/2/users"
#: Bearer token sourced from either env key name variant.
X_BEARER_TOKEN: str | None = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")

# ---------------------------------------------------------------------------
# Swedish parliamentary parties
# Single canonical definition, imported by the ETL, the model, and the site build.
# ---------------------------------------------------------------------------
#: All valid Riksdag party abbreviations.
VALID_PARTIES: frozenset[str] = frozenset({"C", "KD", "L", "M", "MP", "S", "SD", "V"})

#: Historical party-abbreviation renames, applied at ingestion so the same party
#: carries one label across time.  Folkpartiet (``FP``) became Liberalerna
#: (``L``) in 2015; without this, pre-2015 backfilled speeches would be dropped
#: (``FP`` is not in :data:`VALID_PARTIES`) or split into a phantom second party.
PARTY_RENAMES: dict[str, str] = {"FP": "L"}

#: Left-to-right display order for charts and tables.
PARTY_ORDER: list[str] = ["V", "MP", "S", "C", "L", "KD", "M", "SD"]

#: Twitter / X user IDs for each party's current leader account(s).
#: Update these when party leadership changes.
PARTY_LEADERS_IDS: dict[str, list[str]] = {
    "S": ["1587012835409788928"],
    "M": ["747426555417198592"],
    "V": ["282532238"],
    "L": ["455193032"],
    "KD": ["1407151866"],
    "C": ["232799403"],
    "MP": ["41214271", "370900852"],
    "SD": ["95972673"],
}

# ---------------------------------------------------------------------------
# Tweet-fetch limits
# ---------------------------------------------------------------------------
#: Maximum tweets stored per party per ETL run.
MAX_TWEETS_PER_PARTY: int = 2
#: Soft cap on total tweets fetched per calendar month.
MONTHLY_TWEET_LIMIT: int = 100
#: Seconds to wait when the Twitter API returns HTTP 429.
RATE_LIMIT_WAIT_SECONDS: int = 900

# ---------------------------------------------------------------------------
# Inference model
# ---------------------------------------------------------------------------
#: Hugging Face repository for the fine-tuned party classifier.
MODEL_NAME_OR_PATH: str = os.getenv(
    "MODEL_NAME_OR_PATH", "MartinBlomqvist/maktsprak_classifier_clean"
)

# ---------------------------------------------------------------------------
# Training hyperparameters (used by scripts/train_party_model_db.py)
# ---------------------------------------------------------------------------
TRAIN_BASE_MODEL: str = "KB/bert-base-swedish-cased"
TRAIN_MODEL_DIR: Path = PROJECT_ROOT / "data" / "models" / "party_classifier"
TRAIN_BATCH_SIZE: int = 16
TRAIN_MAX_EPOCHS: int = 20
TRAIN_MAX_LENGTH: int = 512
TRAIN_LEARNING_RATE: float = 3e-5
TRAIN_WEIGHT_DECAY: float = 0.01
TRAIN_EARLY_STOPPING_PATIENCE: int = 5
TRAIN_LABEL_SMOOTHING: float = 0.05
TRAIN_GRAD_CLIP_NORM: float = 1.0
TRAIN_DROPOUT_PROB: float = 0.2
TRAIN_MAX_LR: float = 5e-5
