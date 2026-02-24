"""
One-time script to backfill all Riksdag speeches from a given date.
Usage: python scripts/backfill_speeches.py [from_date]
Default from_date: 2025-09-15
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.maktsprak_pipeline.etl import run_historical_backfill

if __name__ == "__main__":
    from_date = sys.argv[1] if len(sys.argv) > 1 else "2025-09-15"
    run_historical_backfill(from_date)
