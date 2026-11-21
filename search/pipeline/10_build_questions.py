"""Step 10: write the question sets with a language model, and try the prompts first.

Commands:

- ``pilot`` tries the prompts on passages of their own draw, apart from the
  real one, one request per passage, and stops at the pilot's budget. Each
  pilot run gets its own directory, so a changed prompt is tried afresh.
- ``generate`` writes one of the at most two pre-registered generations of the
  question sets, in message batches at half the list price. It needs
  ``--confirm``, since it is the generation the evaluation uses.
- ``assemble`` builds every question set from a generation's logs and the
  corpus, writes them as JSON Lines with their hashes, and writes the manifest.
- ``review`` draws the 50 questions of the review and writes the form.
- ``tally`` checks the answers from the form and records the review.

``pilot`` and ``generate`` read the stemmed BM25 index for the rare-word check,
so step 05 comes first. Everything the model is asked and answers is logged
under ``questions_work/`` in the search-data directory.

Usage::

    python search/pipeline/10_build_questions.py pilot
    python search/pipeline/10_build_questions.py generate --generation 1 --confirm
    python search/pipeline/10_build_questions.py assemble --generation 1
    python search/pipeline/10_build_questions.py review --generation 1
    python search/pipeline/10_build_questions.py tally --generation 1 --answers svar.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report
from search.corpus.table import load_passage_table
from search.eval.assemble import (
    MANIFEST_NAME,
    interpellation_questions,
    manifest_of,
    passage_questions,
    unanswerable_questions,
    write_sets,
)
from search.eval.generation import (
    Attempt,
    RareWords,
    load_passage_texts,
    load_prompt,
    read_log,
    rejudge,
    simulate,
)
from search.eval.interpellations import draw_sessions, read_debates, sessions_by_period
from search.eval.llm import complete_many, load_key, run_batch
from search.eval.questions import QUESTION_SETS, read_question_set, set_file_name
from search.eval.review import ReviewItem, draw_review, review_form, tally
from search.eval.runner import (
    ATTEMPTS_NAME,
    PENDING_NAME,
    PRIOR_REQUEST_USD,
    UNANSWERABLE_NAME,
    RunContext,
    run_unanswerable,
    run_waves,
    speaker_names,
)
from search.eval.sampling import build_frame, draw_slots
from search.eval.unanswerable import KINDS, UnanswerableRequest, accepted
from search.index.analyzer import STEMMING
from search.index.lexical import LexicalIndex, lexical_dir
from search.models import session_label, session_start_year

logger = logging.getLogger("build_questions")

WORK_DIR_NAME = "questions_work"
PROMPT_NAMES = ("natural", "stance", *KINDS)
REVIEW_ITEMS_NAME = "review_items.json"


def _context(settings: Settings, directory: Path, seed: int) -> RunContext:
    """Load what a run needs: the passages, the frame, the stemmed index and the prompts."""
    search_dir = settings.data.search
    table = load_passage_table(search_dir / PASSAGES_NAME)
    index, _ = LexicalIndex.load(lexical_dir(search_dir, STEMMING))
    return RunContext(
        directory=directory,
        table=table,
        frame=build_frame(table, seed),
        speeches_path=search_dir / SPEECHES_NAME,
        rare_words=RareWords(index, settings.questions.rare_share),
        prompts={name: load_prompt(name) for name in PROMPT_NAMES},
        generation=settings.generation,
    )


def _pilot_dirs(work_dir: Path) -> list[Path]:
    """The directories of earlier pilot runs, in order."""
    return sorted(
        (path for path in work_dir.glob("pilot-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[1]),
    )


def _spent(directory: Path) -> float:
    """What a run has cost so far."""
    attempts = read_log(directory / ATTEMPTS_NAME)
    requests = read_log(directory / UNANSWERABLE_NAME, UnanswerableRequest)
    return sum(entry.cost_usd for entry in [*attempts, *requests])


def _write_pilot_review(context: RunContext, directory: Path, settings: Settings) -> None:
    """Write what the pilot produced as Markdown, for reading."""
    attempts: list[Attempt] = read_log(directory / ATTEMPTS_NAME)
    texts = load_passage_texts(context.speeches_path, {a.passage_id for a in attempts})
    lines = [f"# Pilot {directory.name}", ""]
    outcomes = Counter(attempt.outcome for attempt in attempts)
    lines.append(f"Outcomes: {dict(outcomes)}. Cost: {_spent(directory):.4f} USD.")
    lines.append("")
    for attempt in attempts:
        text = texts[attempt.passage_id]
        banned = context.rare_words.banned(text)
        answer = json.loads(attempt.response) if attempt.response else {}
        lines += [
            f"## {attempt.custom_id} ({attempt.passage_id}) {attempt.outcome}",
            "",
            *(f"> {paragraph}" for paragraph in text.split("\n\n")),
            "",
            f"Banned: {', '.join(banned) or '(none)'}",
            "",
            f"Answer: `{json.dumps(answer, ensure_ascii=False)}`",
            "",
        ]
        if attempt.problem:
            lines += [f"Problem: {attempt.problem} {list(attempt.reused) or ''}", ""]
    requests = read_log(directory / UNANSWERABLE_NAME, UnanswerableRequest)
    names = speaker_names(context.speeches_path)
    for kind in KINDS:
        kept, turned_away = accepted(kind, requests, 10_000, names)
        lines += [f"## Unanswerable: {kind}", "", f"Turned away: {turned_away}", ""]
        lines += [f"- {text}" for text, _ in kept]
        lines.append("")
    path = directory / "review.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    logger.info("Pilot written out to %s", path)


def pilot(settings: Settings) -> None:
    """Try the prompts on the pilot's own passages, within the pilot's budget."""
    work_dir = settings.data.search / WORK_DIR_NAME
    earlier = _pilot_dirs(work_dir)
    spent_before = sum(_spent(path) for path in earlier)
    budget = settings.questions.pilot_budget_usd
    directory = work_dir / f"pilot-{len(earlier) + 1}"
    directory.mkdir(parents=True, exist_ok=False)
    context = _context(settings, directory, settings.questions.pilot_seed)
    slots = draw_slots(context.frame, settings.questions.pilot_sizes, settings.questions.pilot_seed)
    logger.info("Pilot run %s; earlier pilot runs spent %.4f USD", directory.name, spent_before)

    run_waves(
        context,
        slots,
        complete_many,
        excluded=set(),
        max_passages=1,
        budget_usd=budget,
        spent_before=spent_before,
    )
    spent = spent_before + _spent(directory)
    asked = dict.fromkeys(KINDS, settings.questions.pilot_unanswerable_asked)
    if spent + len(asked) * PRIOR_REQUEST_USD <= budget:
        run_unanswerable(context, asked, complete_many)
    else:
        logger.info("No unanswerable requests: the budget would not hold them")
    _write_pilot_review(context, directory, settings)
    logger.info("The pilot has spent %.4f USD of %.2f", spent_before + _spent(directory), budget)


def generate(settings: Settings, generation: int) -> None:
    """Write one generation of the question sets in message batches."""
    work_dir = settings.data.search / WORK_DIR_NAME
    directory = work_dir / f"generation-{generation}"
    directory.mkdir(parents=True, exist_ok=True)
    context = _context(settings, directory, settings.questions.seed)
    slots = draw_slots(context.frame, settings.questions.sizes, settings.questions.seed)
    pilot_passages = _pilot_passages(context, work_dir)
    pending_path = directory / PENDING_NAME

    def send(requests):
        pending = (
            pending_path.read_text(encoding="utf-8").strip() if pending_path.is_file() else None
        )
        return run_batch(
            requests,
            on_submitted=lambda batch_id: pending_path.write_text(batch_id, encoding="utf-8"),
            batch_id=pending or None,
        )

    log = run_waves(
        context,
        slots,
        send,
        excluded=pilot_passages,
        max_passages=settings.questions.max_passages_per_slot,
    )
    requests = run_unanswerable(context, settings.questions.unanswerable_asked, send)
    pending_path.unlink(missing_ok=True)
    filled = sum(attempt.outcome == "filled" for attempt in log)
    logger.info(
        "%d of %d slots filled; %d unanswerable requests; %.4f USD in all",
        filled,
        len(slots),
        len(requests),
        _spent(directory),
    )


def _pilot_passages(context: RunContext, work_dir: Path) -> set[int]:
    """The passages the pilot asked about, which no slot of a generation may take."""
    return {
        context.table.position(attempt.passage_id)
        for path in _pilot_dirs(work_dir)
        for attempt in read_log(path / ATTEMPTS_NAME)
    }


def _generation_dir(settings: Settings, generation: int) -> Path:
    """The directory of one generation's logs."""
    return settings.data.search / WORK_DIR_NAME / f"generation-{generation}"


def interpellation_sessions(settings: Settings) -> list[str]:
    """Draw the sessions of the interpellation set: whole sessions within one period each."""
    last_complete = session_label(session_start_year(settings.source.last_session) - 1)
    periods = sessions_by_period(settings.corpus.first_session, last_complete)
    return draw_sessions(
        periods, settings.questions.interpellation_sessions_per_period, settings.questions.seed
    )


def assemble(settings: Settings, generation: int) -> None:
    """Build every question set from a generation's logs and the corpus, and record them.

    The log's answers are judged again with the checks as they stand, and the
    waves are run again against the log, so a check corrected after the run
    counts as if it had held from the start. Assembly stops if that calls for a
    request the log holds no answer for.
    """
    directory = _generation_dir(settings, generation)
    context = _context(settings, directory, settings.questions.seed)
    slots = draw_slots(context.frame, settings.questions.sizes, settings.questions.seed)
    made = read_log(directory / ATTEMPTS_NAME)
    texts = load_passage_texts(context.speeches_path, {attempt.passage_id for attempt in made})
    judged = rejudge(made, texts, context.rare_words)
    attempts, missing = simulate(
        slots,
        judged,
        context.frame,
        context.table,
        _pilot_passages(context, settings.data.search / WORK_DIR_NAME),
        settings.questions.max_passages_per_slot,
    )
    if missing:
        names = ", ".join(item.slot.slot_id for item in missing)
        raise SystemExit(f"The log holds no answer for {len(missing)} requests: {names}.")
    requests = read_log(directory / UNANSWERABLE_NAME, UnanswerableRequest)
    sets = dict(passage_questions(slots, attempts, context.table, generation))
    sets["unanswerable"], turned_away = unanswerable_questions(
        requests,
        settings.questions.unanswerable,
        speaker_names(context.speeches_path),
        generation,
        settings.questions.seed,
    )
    sessions = interpellation_sessions(settings)
    debates, left_out = read_debates(context.speeches_path, sessions)
    sets["interpellation"] = interpellation_questions(debates, settings.questions.seed)
    records = write_sets(settings.tracked.questions, sets)
    manifest = manifest_of(
        settings=settings,
        generation=generation,
        prompts={name: prompt.sha256 for name, prompt in context.prompts.items()},
        rare_below=context.rare_words.below,
        band_edges=context.frame.band_edges,
        slots=slots,
        made=made,
        judged=judged,
        attempts=attempts,
        requests=requests,
        turned_away=turned_away,
        sessions=sessions,
        left_out=left_out,
        records=records,
    )
    write_report(manifest, settings.tracked.questions / MANIFEST_NAME)
    for name, record in records.items():
        logger.info(
            "%-14s %4d questions (%d dev, %d test) %s",
            name,
            record.questions,
            record.dev,
            record.test,
            record.sha256[:16],
        )


def review(settings: Settings, generation: int) -> None:
    """Draw the review's questions and write the form beside the generation's logs."""
    sets = {
        name: read_question_set(settings.tracked.questions / set_file_name(name))
        for name in QUESTION_SETS
    }
    passage_ids = {
        question.passage_id
        for questions in sets.values()
        for question in questions
        if question.passage_id
    }
    texts = load_passage_texts(settings.data.search / SPEECHES_NAME, passage_ids)
    items = draw_review(sets, texts, settings.questions.review_seed)
    directory = _generation_dir(settings, generation)
    (directory / REVIEW_ITEMS_NAME).write_text(
        json.dumps([item.model_dump() for item in items], ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    form = directory / f"granskning-generation-{generation}.html"
    form.write_text(
        review_form(items, f"granskning-generation-{generation}"), encoding="utf-8", newline="\n"
    )
    logger.info("The review form is %s", form)


def record_review(settings: Settings, generation: int, answers: Path) -> None:
    """Check the review's answers and record the tally among the tracked reports."""
    directory = _generation_dir(settings, generation)
    raw_items = json.loads((directory / REVIEW_ITEMS_NAME).read_text(encoding="utf-8"))
    items = [ReviewItem.model_validate(item) for item in raw_items]
    record = tally(
        items,
        answers.read_text(encoding="utf-8"),
        generation,
        datetime.now(timezone.utc).date(),
    )
    write_report(record, settings.tracked.reports / f"review_generation_{generation}.json")
    logger.info(
        "%d of %d approved (%.0f%%); the threshold is %.0f%%: %s",
        record.approved,
        len(record.verdicts),
        100 * record.share,
        100 * record.threshold,
        "passed" if record.passed else "not passed",
    )


def main() -> None:
    """Parse the command line and run the command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="The .env file that holds ANTHROPIC_API_KEY. Defaults to .env in the data root.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pilot", help="Try the prompts on the pilot's passages.")
    real = commands.add_parser("generate", help="Write one generation of the question sets.")
    real.add_argument("--generation", type=int, choices=(1, 2), required=True)
    real.add_argument(
        "--confirm",
        action="store_true",
        help="Required: this writes a pre-registered generation.",
    )
    for name, text in (
        ("assemble", "Build and record the question sets of a generation."),
        ("review", "Draw the review's questions and write the form."),
        ("tally", "Record the answers from the review form."),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--generation", type=int, choices=(1, 2), required=True)
        if name == "tally":
            command.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    if args.command == "assemble":
        assemble(settings, args.generation)
        return
    if args.command == "review":
        review(settings, args.generation)
        return
    if args.command == "tally":
        record_review(settings, args.generation, args.answers)
        return
    env_file = args.env_file or settings.data.search.parents[1] / ".env"
    if args.command == "generate" and not args.confirm:
        raise SystemExit("A generation is pre-registered and limited to two; pass --confirm.")
    load_key(env_file)
    if args.command == "pilot":
        pilot(settings)
    else:
        generate(settings, args.generation)


if __name__ == "__main__":
    main()
