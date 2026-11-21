"""Step 02: build the speech corpus from the fetched datasets.

Reads the dataset of every session from ``source.first_session`` to
``source.last_session`` in the settings, after checking that the manifest
holds them all and that each file matches its recorded hash. The speeches of
the sessions from ``corpus.first_session`` are written to
``anforanden.parquet`` in the search-data directory. The report, which is
tracked, accounts for every record that was left out, for every paragraph set
aside by the rule that set it aside, and for the speeches kept with a known
defect of the source.

Usage::

    python search/pipeline/02_build_anforanden.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.corpus.build import BUILD_REPORT_NAME, build_corpus
from search.corpus.fetch import MANIFEST_NAME, load_manifest, sessions_between
from search.corpus.store import SPEECHES_NAME, write_report

logger = logging.getLogger("build_anforanden")


def main() -> None:
    """Parse the command line and build the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    manifest = load_manifest(settings.tracked.reports / MANIFEST_NAME)
    if not manifest.files:
        raise SystemExit("The manifest is empty. Run 01_fetch_anforanden.py first.")

    destination = settings.data.search / SPEECHES_NAME
    report = build_corpus(
        manifest,
        settings.data.raw_anforanden,
        destination,
        sessions=sessions_between(settings.source.first_session, settings.source.last_session),
        first_session=settings.corpus.first_session,
    )
    write_report(report, settings.tracked.reports / BUILD_REPORT_NAME)

    logger.info("%d source records, %d speeches kept", report.source_records, report.speeches)
    for reason, count in report.excluded.items():
        logger.info("  left out, %s: %d", reason, count)
    for rule, count in report.set_aside.items():
        logger.info("  paragraphs set aside, %s: %d", rule, count)
    for flag, keys in report.flagged.items():
        logger.info("  speeches kept with a known defect of the source, %s: %d", flag, len(keys))
    logger.info("Speeches written to %s", destination)


if __name__ == "__main__":
    main()
