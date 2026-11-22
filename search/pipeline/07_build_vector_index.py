"""Step 07: build the exact vector index from a whole run of step 06.

Reads one candidate's run over one input set, checks every shard against its
hash and the run against its input set, and writes the vectors as one matrix
with the passages' positions and a manifest. The index is then checked once
against the passages file it points into. The tracked report keeps the run's
manifest and the index's, so the provenance of every vector stays in git: the
model, its revision, the prompts, the texts that were cut and what the run cost.

Usage::

    python search/pipeline/07_build_vector_index.py --candidate bekko-a25m --inputs subset-200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.store import write_report
from search.corpus.table import load_passage_table
from search.index.embed import RunManifest, read_inputs, read_run_manifest
from search.index.vector import (
    VectorIndexManifest,
    build_vector_index,
    index_path,
    inputs_path,
    load_vector_index,
    passages_path,
    run_path,
)

logger = logging.getLogger("vector_index")

REPORTS_DIR = "vector_runs"


class VectorRunReport(BaseModel):
    """A run of step 06 and the index step 07 built from it.

    Attributes:
        run: The run's manifest.
        index: The index's manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: RunManifest
    index: VectorIndexManifest


def report_path(settings: Settings, inputs: str, candidate: str) -> Path:
    """The tracked report of one candidate's run and index over one input set."""
    return settings.tracked.reports / REPORTS_DIR / f"{inputs}--{candidate}.json"


def build(settings: Settings, inputs_name: str, candidate: str) -> VectorRunReport:
    """Build one candidate's index over one input set, check it, and write its report.

    Raises:
        KeyError: If the settings name no such candidate.
        ValueError: If the run is not whole or does not match its inputs or the
            passages file.
    """
    settings.vectors.candidate(candidate)
    search = settings.data.search
    run = run_path(search, inputs_name, candidate)
    inputs = inputs_path(search, inputs_name)
    out = index_path(search, inputs_name, candidate)
    manifest = build_vector_index(run, inputs, out)

    target = read_inputs(inputs).manifest.target_words
    passages = passages_path(search, target, settings.passages.target_words)
    load_vector_index(out, load_passage_table(passages).passage_ids)
    report = VectorRunReport(run=read_run_manifest(run), index=manifest)
    write_report(report, report_path(settings, inputs_name, candidate))
    logger.info(
        "%s over %s: %d passages of %d dimensions in %s",
        candidate,
        inputs_name,
        manifest.rows,
        manifest.dimension,
        out,
    )
    return report


def main() -> None:
    """Parse the command line and build the index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--inputs", required=True, help="The input set, for example subset-200.")
    args = parser.parse_args()

    configure_logging()
    build(settings_from(args), args.inputs, args.candidate)


if __name__ == "__main__":
    main()
