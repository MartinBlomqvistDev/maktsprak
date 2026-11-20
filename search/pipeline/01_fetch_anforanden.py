"""Step 01: fetch the Riksdag's speech datasets and record them in the manifest.

One zip per parliamentary session goes to the raw-data directory. The manifest,
which is tracked, records the size, the SHA-256 and the fetch date of each.

Usage::

    python search/pipeline/01_fetch_anforanden.py
    python search/pipeline/01_fetch_anforanden.py --session 2026/27 --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.fetch import (
    Manifest,
    fetch_datasets,
    load_manifest,
    sessions_between,
    write_manifest,
)

logger = logging.getLogger("fetch_anforanden")

MANIFEST_NAME = "anforanden_manifest.json"
TIMEOUT_SECONDS = 120.0


async def run(settings: Settings, sessions: list[str], force: bool) -> Manifest:
    """Fetch *sessions* and write the updated manifest."""
    manifest_path = settings.tracked.reports / MANIFEST_NAME
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        manifest = await fetch_datasets(
            sessions,
            url_template=settings.source.dataset_url,
            raw_dir=settings.data.raw_anforanden,
            known=load_manifest(manifest_path),
            client=client,
            force=force,
        )
    write_manifest(manifest, manifest_path)
    return manifest


def main() -> None:
    """Parse the command line and fetch the datasets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Fetch only this session, for example 2026/27. Repeat for several.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch again even when the file is already on disk.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    sessions = args.session or sessions_between(
        settings.source.first_session, settings.source.last_session
    )
    manifest = asyncio.run(run(settings, sessions, args.force))

    total_mb = sum(entry.bytes for entry in manifest.files) / 1e6
    logger.info(
        "%d files in the manifest, %.1f MB, under %s",
        len(manifest.files),
        total_mb,
        settings.data.raw_anforanden,
    )


if __name__ == "__main__":
    main()
