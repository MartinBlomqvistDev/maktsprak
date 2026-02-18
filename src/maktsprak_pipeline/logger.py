"""Logging configuration for the MaktspråkAI pipeline.

All modules should call ``get_logger()`` to obtain the shared Loguru logger
instance.  The log directory and handlers are initialised on first call so
that importing this module never creates files or directories as a side effect.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger as _logger

from .config import LOG_LEVEL, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_configured: bool = False


def get_logger():  # -> loguru.Logger
    """Return the configured Loguru logger, initialising handlers on first call.

    The logger writes to:
    - ``logs/etl_<UTC-timestamp>.log`` — rotated at 10 MB, retained for 30 days.
    - ``stdout`` — useful during interactive development and CI runs.

    Returns:
        The module-level Loguru ``logger`` singleton.
    """
    global _configured
    if _configured:
        return _logger

    log_dir: Path = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"etl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

    _logger.remove()
    _logger.add(
        log_file,
        level=LOG_LEVEL,
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,  # thread-safe async writes
    )
    _logger.add(sys.stdout, level=LOG_LEVEL, colorize=True)

    _configured = True
    return _logger
