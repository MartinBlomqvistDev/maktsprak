"""Entry point for the MaktspråkAI ETL pipeline.

Run via the Windows scheduled task::

    python -m src.maktsprak_pipeline.main

Or via the installed console script (after ``pip install -e .``)::

    maktsprak-etl
"""

from __future__ import annotations

from .logger import get_logger
from .pipeline.orchestrate import run_etl

logger = get_logger()


def main() -> None:
    """Run the incremental ETL pipeline, logging any unhandled exceptions."""
    logger.info("MaktspråkAI ETL — starting.")
    try:
        run_etl()
    except Exception:
        logger.exception("ETL run failed with an unhandled exception.")
        raise
    else:
        logger.info("MaktspråkAI ETL — finished successfully.")


if __name__ == "__main__":
    main()
