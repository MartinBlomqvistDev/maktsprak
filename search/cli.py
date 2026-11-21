"""Command-line plumbing shared by the numbered pipeline scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from search.config import DATA_ROOT_ENV, Settings, load_settings


def add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--data-root`` option every pipeline script takes."""
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Where the gitignored bulk data lives. Defaults to "
            f"{DATA_ROOT_ENV}, then to the repository root."
        ),
    )


def settings_from(args: argparse.Namespace) -> Settings:
    """Load the settings with the data root the command line asked for."""
    return load_settings(data_root=args.data_root)


def configure_logging() -> None:
    """Send progress messages to the terminal, one plain line each."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
