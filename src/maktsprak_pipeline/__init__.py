"""MaktspråkAI pipeline package.

Top-level re-exports for the most common entry points::

    from src.maktsprak_pipeline import run_etl, run_historical_backfill
"""

from .pipeline.orchestrate import run_etl, run_historical_backfill

__all__ = ["run_etl", "run_historical_backfill"]
