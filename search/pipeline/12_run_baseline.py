"""Step 12: tune BM25 on the development split, then run it on every question set there.

The question sets are scored only once their hashes stand in a committed
addendum to the pre-registration. Before anything is scored, the script checks
that every set file still matches its hash in the manifest, and that every hash
appears in ``docs/search/PREREGISTRATION.md`` as committed at ``HEAD``; it stops
if either check fails. Only the development split is run: the test split is not
opened in this phase.

The tuning tries every analyzer with every pair of ``k1`` and ``b`` in
``search/config.yaml`` on the natural questions, and picks one configuration by
the rule in :mod:`search.eval.baseline`. Every set with gold speeches is then
run once with it. The reports, which are tracked, hold the whole grid, the
choice, each set's metrics with 95% bootstrap intervals, by period and by
party, and every question's result.

Usage::

    python search/pipeline/12_run_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import REPO_ROOT, Settings
from search.corpus.store import PASSAGES_NAME, write_report
from search.corpus.table import load_passage_table
from search.eval.assemble import MANIFEST_NAME, QuestionsManifest
from search.eval.baseline import (
    GridPoint,
    SetResult,
    choose,
    run_set,
    speech_years,
    summarize_set,
    tune,
)
from search.eval.questions import read_question_set, split_of
from search.index.analyzer import ANALYZERS
from search.index.lexical import Bm25Params, LexicalIndex, lexical_dir

logger = logging.getLogger("baseline")

PREREGISTRATION = "docs/search/PREREGISTRATION.md"
TUNING_REPORT = "bm25_tuning_dev.json"
BASELINE_REPORT = "baseline_dev.json"
QUESTIONS_REPORT = "baseline_dev_questions.jsonl"

#: The sets with gold speeches, which retrieval is scored on.
SCORED_SETS = ("natural", "keyword", "interpellation", "filtered", "stance")

#: The set the tuning is scored on.
TUNED_ON = "natural"


class TuningReport(BaseModel):
    """The tuning of BM25 on the development split.

    Attributes:
        tuned_on: The set the configurations were scored on.
        questions: How many questions that set has in the development split.
        grid: Every configuration tried, and its scores.
        chosen: The configuration the rule picked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tuned_on: str
    questions: int
    grid: tuple[GridPoint, ...]
    chosen: GridPoint


class BaselineReport(BaseModel):
    """BM25's run on the development split.

    Attributes:
        split: The split run.
        sets_sha256: The hash of each question set file scored.
        analyzer: The chosen analyzer.
        k1: The chosen ``k1``.
        b: The chosen ``b``.
        resamples: The bootstrap resamples behind each interval.
        bootstrap_seed: The seed of the resamples.
        sets: Each set's result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    sets_sha256: dict[str, str]
    analyzer: str
    k1: float
    b: float
    resamples: int
    bootstrap_seed: int
    sets: dict[str, SetResult]


def committed_preregistration() -> str:
    """The pre-registration as committed at ``HEAD``."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{PREREGISTRATION}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8")


def check_registered(settings: Settings) -> QuestionsManifest:
    """Stop unless every set matches its hash and every hash is in the committed pre-registration.

    Returns:
        The manifest of the question sets.
    """
    path = settings.tracked.questions / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"{path} does not exist. Assemble the question sets first.")
    manifest = QuestionsManifest.model_validate_json(path.read_text(encoding="utf-8"))
    registered = committed_preregistration()
    for name, record in manifest.sets.items():
        actual = hashlib.sha256((settings.tracked.questions / record.file).read_bytes()).hexdigest()
        if actual != record.sha256:
            raise SystemExit(f"{record.file} no longer matches its hash in the manifest.")
        if record.sha256 not in registered:
            raise SystemExit(
                f"The hash of {name} is not in the committed {PREREGISTRATION}. "
                "Commit the addendum that records it before anything is scored."
            )
    return manifest


def main() -> None:
    """Parse the command line, tune BM25 and run the baseline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    manifest = check_registered(settings)
    sets = {
        name: split_of(read_question_set(settings.tracked.questions / f"{name}.jsonl"), "dev")
        for name in SCORED_SETS
    }
    table = load_passage_table(settings.data.search / PASSAGES_NAME)
    indexes = {
        analyzer.name: LexicalIndex.load(lexical_dir(settings.data.search, analyzer))[0]
        for analyzer in ANALYZERS
    }

    baseline = settings.baseline
    grid = tune(indexes, table, sets[TUNED_ON], baseline.k1, baseline.b)
    chosen = choose(grid)
    write_report(
        TuningReport(
            tuned_on=TUNED_ON, questions=len(sets[TUNED_ON]), grid=tuple(grid), chosen=chosen
        ),
        settings.tracked.reports / TUNING_REPORT,
    )
    logger.info(
        "Chosen: %s, k1 %.2f, b %.2f; recall@10 %.4f, MRR@10 %.4f on %d natural questions",
        chosen.analyzer,
        chosen.k1,
        chosen.b,
        chosen.recall_10,
        chosen.mrr_10,
        len(sets[TUNED_ON]),
    )

    params = Bm25Params(k1=chosen.k1, b=chosen.b)
    years = speech_years(table)
    lines = []
    results = {}
    for name, questions in sets.items():
        per_question = run_set(indexes[chosen.analyzer], table, questions, params, years)
        lines.extend(result.model_dump_json() for result in per_question)
        results[name] = summarize_set(per_question, baseline.bootstrap_seed, baseline.resamples)
        overall = results[name].overall
        logger.info(
            "%-14s %4d questions: recall@10 %.3f [%.3f, %.3f], MRR@10 %.3f",
            name,
            overall.questions,
            overall.means["recall@10"],
            *overall.intervals["recall@10"],
            overall.means["mrr@10"],
        )
    report = BaselineReport(
        split="dev",
        sets_sha256={name: manifest.sets[name].sha256 for name in SCORED_SETS},
        analyzer=chosen.analyzer,
        k1=chosen.k1,
        b=chosen.b,
        resamples=baseline.resamples,
        bootstrap_seed=baseline.bootstrap_seed,
        sets=results,
    )
    write_report(report, settings.tracked.reports / BASELINE_REPORT)
    (settings.tracked.reports / QUESTIONS_REPORT).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
