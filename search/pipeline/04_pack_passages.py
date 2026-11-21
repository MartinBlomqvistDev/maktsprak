"""Step 04: pack each speech's paragraphs into passages.

Reads ``anforanden.parquet`` and cuts every speech into runs of whole
consecutive paragraphs of nearly equal size, towards ``passages.target_words``
in ``search/config.yaml``. Writes ``passages.parquet`` to the search-data
directory. The report, which is tracked, counts the passages and records the
check that they hold exactly the paragraphs and words of the corpus. The script
stops without writing anything if that check fails.

Usage::

    python search/pipeline/04_pack_passages.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.corpus.build import BUILD_REPORT_NAME, BuildReport
from search.corpus.passages import PASSAGES_REPORT_NAME, CorpusTotals, build_passages
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report

logger = logging.getLogger("pack_passages")


def main() -> None:
    """Parse the command line and pack the corpus into passages."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    speeches_path = settings.data.search / SPEECHES_NAME
    if not speeches_path.is_file():
        raise SystemExit(f"{speeches_path} does not exist. Run 02_build_anforanden.py first.")
    build_report_path = settings.tracked.reports / BUILD_REPORT_NAME
    if not build_report_path.is_file():
        raise SystemExit(f"{build_report_path} does not exist. Run 02_build_anforanden.py first.")
    build = BuildReport.model_validate_json(build_report_path.read_text(encoding="utf-8"))

    destination = settings.data.search / PASSAGES_NAME
    report = build_passages(
        speeches_path,
        destination,
        settings.passages.target_words,
        settings.passages.trial_targets,
        CorpusTotals(speeches=build.speeches, paragraphs=build.paragraphs, words=build.words),
        build.inputs_sha256,
    )
    write_report(report, settings.tracked.reports / PASSAGES_REPORT_NAME)

    sizes = report.words_per_passage
    logger.info(
        "%d speeches packed into %d passages of %.1f words at the median (target %d)",
        report.speeches,
        report.passages,
        sizes.median,
        report.target_words,
    )
    logger.info(
        "The passages hold %d paragraphs and %d words, as the corpus does",
        report.cover.in_passages.paragraphs,
        report.cover.in_passages.words,
    )
    for trial in report.trial_targets:
        logger.info(
            "  a target of %d would give %d passages, median %.1f words",
            trial.target_words,
            trial.passages,
            trial.median_words,
        )
    logger.info("Passages written to %s", destination)


if __name__ == "__main__":
    main()
