"""Step 06: embed passages and questions, in shards that a stopped run goes on from.

Three commands:

- ``export`` writes an input set: the passages' texts and the development
  split's questions, with a manifest of their hashes. ``--set subset`` draws the
  subset of step A and records it in the tracked ``vector_subset.json``;
  ``--set full`` takes every passage, packed towards ``--target-words``. A
  passage size other than the corpus's own is packed first, by the same rule.
- ``embed`` embeds an input set with one candidate from the settings. A run
  that stops is started again with the same command and goes on from the first
  shard it lacks. A run through the API stops before it spends more than its cap.
- ``bundle`` zips an input set with the code that embeds it, so that a machine
  with a GPU can run ``embed`` and send the run back.

The test split is never read here.

Usage::

    python search/pipeline/06_embed_passages.py export --set subset
    python search/pipeline/06_embed_passages.py embed --candidate bekko-a25m --inputs subset-200
    python search/pipeline/06_embed_passages.py bundle --inputs subset-200 --out stepA.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import REPO_ROOT, Settings
from search.corpus.passages import CorpusTotals, build_passages
from search.corpus.store import SPEECHES_NAME, write_report
from search.corpus.table import load_passage_table
from search.eval.questions import read_question_set, split_of
from search.index.embed import (
    RunManifest,
    build_embedder,
    embed_inputs,
    ids_sha256,
    load_openrouter_key,
    read_inputs,
    write_inputs,
)
from search.index.lexical import file_sha256, passage_texts
from search.index.subset import draw_subset
from search.index.vector import inputs_path, passages_path, run_path

logger = logging.getLogger("embed")

#: The sets whose questions have gold speeches; their development questions are embedded.
GOLD_SETS = ("natural", "keyword", "interpellation", "filtered", "stance")

SUBSET_REPORT = "vector_subset.json"
BUILD_REPORT = "anforanden_build.json"


def passages_file(settings: Settings, target_words: int) -> Path:
    """The passages file of one passage size."""
    return passages_path(settings.data.search, target_words, settings.passages.target_words)


class SubsetReport(BaseModel):
    """The subset of step A, recorded before any vector is scored.

    Attributes:
        seed: The seed of the draw.
        size: The passages the subset was filled up to.
        passages: The passages it holds.
        passage_ids_sha256: The hash of their ids, in order.
        gold_speeches: The gold speeches of the development split it holds.
        gold_passages: Their passages.
        drawn_speeches: The speeches the draw added.
        drawn_passages: Their passages.
        passages_file_sha256: The hash of the passages file it was drawn from.
        questions_sha256: The hash of each question set the gold came from.
        by_period: The passages of each period.
        by_party: The passages of each party.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    size: int
    passages: int
    passage_ids_sha256: str
    gold_speeches: int
    gold_passages: int
    drawn_speeches: int
    drawn_passages: int
    passages_file_sha256: str
    questions_sha256: dict[str, str]
    by_period: dict[str, int]
    by_party: dict[str, int]


def development_questions(settings: Settings) -> tuple[list, dict[str, str]]:
    """The development questions of the gold sets, and the hash of each set's file."""
    questions, hashes = [], {}
    for name in GOLD_SETS:
        path = settings.tracked.questions / f"{name}.jsonl"
        hashes[name] = file_sha256(path)
        questions.extend(split_of(read_question_set(path), "dev"))
    return questions, hashes


def ensure_passages(settings: Settings, target_words: int) -> Path:
    """The passages file of a size, packed first when it does not exist yet."""
    path = passages_file(settings, target_words)
    if path.is_file():
        return path
    build = json.loads((settings.tracked.reports / BUILD_REPORT).read_text(encoding="utf-8"))
    logger.info("Packing the corpus towards %d words into %s", target_words, path)
    report = build_passages(
        settings.data.search / SPEECHES_NAME,
        path,
        target_words,
        (),
        CorpusTotals(
            speeches=build["speeches"], paragraphs=build["paragraphs"], words=build["words"]
        ),
        build["inputs_sha256"],
    )
    write_report(report, settings.tracked.reports / f"passages_{target_words}.json")
    return path


def export(settings: Settings, which: str, target_words: int) -> None:
    """Write an input set: the subset of step A, or every passage of one size."""
    if which == "subset" and target_words != settings.passages.target_words:
        raise SystemExit("The subset of step A is drawn from the corpus's own passages.")
    path = ensure_passages(settings, target_words)
    table = load_passage_table(path)
    questions, hashes = development_questions(settings)
    if which == "subset":
        gold = {speech for question in questions for speech in question.gold}
        draw = draw_subset(table, gold, settings.vectors.subset_size, settings.vectors.subset_seed)
        positions = [int(position) for position in draw.positions]
        chosen = [table.passage_ids[position] for position in positions]
        periods: dict[str, int] = {}
        parties: dict[str, int] = {}
        for position in positions:
            period = table.period_of_year(int(table.years[position]))
            periods[period] = periods.get(period, 0) + 1
            party = str(table.parties[position])
            parties[party] = parties.get(party, 0) + 1
        write_report(
            SubsetReport(
                seed=settings.vectors.subset_seed,
                size=settings.vectors.subset_size,
                passages=len(positions),
                passage_ids_sha256=ids_sha256(chosen),
                gold_speeches=draw.gold_speeches,
                gold_passages=draw.gold_passages,
                drawn_speeches=draw.drawn_speeches,
                drawn_passages=draw.drawn_passages,
                passages_file_sha256=file_sha256(path),
                questions_sha256=hashes,
                by_period=dict(sorted(periods.items())),
                by_party=dict(sorted(parties.items())),
            ),
            settings.tracked.reports / SUBSET_REPORT,
        )
        logger.info(
            "Subset: %d passages, %d of them from %d gold speeches",
            len(positions),
            draw.gold_passages,
            draw.gold_speeches,
        )
    else:
        positions = list(range(len(table)))
    wanted = set(positions)
    texts = {}
    for position, (passage, text) in enumerate(
        passage_texts(settings.data.search / SPEECHES_NAME, path)
    ):
        if position in wanted:
            if passage != table.passage_ids[position]:
                raise SystemExit(f"{path} and its table disagree at position {position}.")
            texts[position] = text
    name = f"{which}-{target_words}"
    manifest = write_inputs(
        inputs_path(settings.data.search, name),
        name,
        target_words,
        [(position, table.passage_ids[position], texts[position]) for position in positions],
        [(question.question_id, question.question_set, question.text) for question in questions],
        {
            SPEECHES_NAME: file_sha256(settings.data.search / SPEECHES_NAME),
            path.name: file_sha256(path),
            **{f"{set_name}.jsonl": digest for set_name, digest in hashes.items()},
        },
    )
    logger.info(
        "Input set %s: %d passages and %d questions in %s",
        name,
        manifest.passages,
        manifest.queries,
        inputs_path(settings.data.search, name),
    )


def _report_progress(started: float, done_before: int) -> Callable[[RunManifest], None]:
    """A callback that logs each shard with the rate so far and the time left."""

    def report(manifest: RunManifest) -> None:
        done = sum(record.rows for record in manifest.shards)
        elapsed = time.monotonic() - started
        rate = (done - done_before) / elapsed if elapsed > 0 else 0.0
        left = manifest.inputs.passages - done
        hours = left / rate / 3600 if rate > 0 else float("nan")
        logger.info(
            "Shard %d of %d: %d passages, %.1f a second, %.1f hours left, %.4f USD spent",
            len(manifest.shards),
            manifest.expected_shards,
            done,
            rate,
            hours,
            manifest.usage.cost_usd,
        )

    return report


def embed(settings: Settings, args: argparse.Namespace) -> None:
    """Embed an input set with one candidate, going on from where an earlier run stopped."""
    candidate = settings.vectors.candidate(args.candidate)
    inputs = read_inputs(inputs_path(settings.data.search, args.inputs))
    key = None
    if candidate.kind == "openrouter":
        key = load_openrouter_key(args.env_file or settings.data.search.parents[1] / ".env")
    embedder = build_embedder(
        candidate,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
        key=key,
        cap_usd=args.max_cost_usd or settings.vectors.run_cap_usd,
        batch_inputs=settings.vectors.api_batch_inputs,
        concurrency=settings.vectors.api_concurrency,
    )
    directory = run_path(settings.data.search, args.inputs, args.candidate)
    started = time.monotonic()
    before = 0
    manifest_file = directory / "manifest.json"
    if manifest_file.is_file():
        before = sum(
            record["rows"]
            for record in json.loads(manifest_file.read_text(encoding="utf-8"))["shards"]
        )
    manifest = embed_inputs(
        inputs,
        embedder,
        directory,
        settings.vectors.shard_rows,
        max_shards=args.max_shards,
        progress=_report_progress(started, before),
    )
    logger.info(
        "%s: %d of %d shards, questions %s, %d texts cut, %.4f USD; run in %s",
        args.candidate,
        len(manifest.shards),
        manifest.expected_shards,
        "written" if manifest.queries else "not yet written",
        manifest.usage.truncated,
        manifest.usage.cost_usd,
        directory,
    )


def bundle_readme(inputs: str, folder: str, zip_name: str, candidates: Sequence[str]) -> str:
    """The instructions a bundle carries, for a Mac and for Colab."""
    embed = f"python search/pipeline/06_embed_passages.py embed --data-root . --inputs {inputs}"
    lines = [
        f"Step 06 of Fråga protokollet: embed the input set {inputs} on a machine with a GPU.",
        "",
        "On a Mac, in Terminal, from the folder that holds the zip:",
        "",
        "    curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"    unzip -q {zip_name} -d {folder}",
        f"    cd {folder}",
        "    uv venv --python 3.11",
        "    source .venv/bin/activate",
        "    uv pip install -r requirements-embed.txt",
        *(f"    {embed} --candidate {name} --device mps" for name in candidates),
        "",
        "On Colab, in a GPU runtime, upload the zip and run these cells:",
        "",
        f"    !unzip -q {zip_name} -d {folder}",
        f"    %cd {folder}",
        "    !pip install -q -r requirements-embed.txt",
        *(f"    !{embed} --candidate {name} --device cuda" for name in candidates),
        "",
        "A run that stops is started again with the same command, and goes on from",
        "the first shard it lacks. When a run is done, zip its folder and send it back:",
        f"data/search/vectors/runs/{inputs}/<candidate>/",
        "",
    ]
    return "\n".join(lines)


def bundle(settings: Settings, args: argparse.Namespace) -> None:
    """Zip an input set with the code that embeds it."""
    source = inputs_path(settings.data.search, args.inputs)
    read_inputs(source)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "search", "requirements-embed.txt"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    out = args.out.resolve()
    folder = out.stem
    files: list[tuple[Path, str]] = []
    for path in sorted((REPO_ROOT / "search").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append((path, path.relative_to(REPO_ROOT).as_posix()))
    files.append((REPO_ROOT / "requirements-embed.txt", "requirements-embed.txt"))
    for path in sorted(source.iterdir()):
        files.append((path, f"data/search/vectors/inputs/{args.inputs}/{path.name}"))
    record = {
        "inputs": args.inputs,
        "commit": commit,
        "uncommitted_changes": dirty,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("bundle.json", json.dumps(record, indent=2) + "\n")
        local = [c.name for c in settings.vectors.candidates if c.kind == "sentence_transformers"]
        archive.writestr("README.txt", bundle_readme(args.inputs, folder, out.name, local))
    logger.info(
        "Bundle %s: %d files, from commit %s%s",
        out,
        len(files) + 2,
        commit[:12],
        ", with uncommitted changes" if dirty else "",
    )


def main() -> None:
    """Parse the command line and run the command it names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    exporting = commands.add_parser("export", help="Write an input set.")
    exporting.add_argument("--set", choices=("subset", "full"), required=True)
    exporting.add_argument("--target-words", type=int, default=None)

    embedding = commands.add_parser("embed", help="Embed an input set with one candidate.")
    embedding.add_argument("--candidate", required=True)
    embedding.add_argument("--inputs", required=True, help="The input set, for example subset-200.")
    embedding.add_argument("--device", default=None, help="cpu, mps or cuda; chosen by default.")
    embedding.add_argument("--batch-size", type=int, default=32)
    embedding.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    embedding.add_argument("--max-shards", type=int, default=None, help="Stop after this many.")
    embedding.add_argument("--max-cost-usd", type=float, default=None)
    embedding.add_argument("--env-file", type=Path, default=None)

    bundling = commands.add_parser("bundle", help="Zip an input set with the code.")
    bundling.add_argument("--inputs", required=True)
    bundling.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    configure_logging()
    settings = settings_from(args)
    if args.command == "export":
        export(settings, args.set, args.target_words or settings.passages.target_words)
    elif args.command == "embed":
        embed(settings, args)
    else:
        bundle(settings, args)


if __name__ == "__main__":
    main()
