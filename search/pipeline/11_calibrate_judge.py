"""Step 11: calibrate the judge against Martin's own grades on 100 question-passage pairs.

Commands:

- ``pairs`` draws 25 of the pilot's questions, pairs each with the passage it
  was written from and the three passages BM25 ranks next, and writes the form
  for Martin. The pairs are recorded among the tracked reports.
- ``run`` has the judge grade the same pairs. It needs ``--confirm``, since it
  calls the API. Martin's grades are not read, so the judge cannot see them.
- ``agreement`` reads Martin's answers from the form and compares them with
  the judge's, as the pre-registration's rule requires.

Usage::

    python search/pipeline/11_calibrate_judge.py pairs
    python search/pipeline/11_calibrate_judge.py run --confirm
    python search/pipeline/11_calibrate_judge.py agreement --answers kalibrering.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report
from search.corpus.table import load_passage_table
from search.eval.generation import Attempt, load_passage_texts, load_prompt, read_log
from search.eval.judge import (
    Agreement,
    JudgePair,
    JudgeVerdict,
    PilotQuestion,
    agreement,
    calibration_form,
    calibration_pairs,
    judge_request,
    read_grade,
)
from search.eval.llm import complete_many, cost_usd, load_key
from search.eval.runner import ATTEMPTS_NAME, now
from search.index.analyzer import ANALYZERS
from search.index.lexical import LexicalIndex, lexical_dir

logger = logging.getLogger("calibrate_judge")

JUDGE_DIR_NAME = "judge"
PAIRS_REPORT = "judge_calibration_pairs.json"
VERDICTS_REPORT = "judge_calibration_verdicts.json"
AGREEMENT_REPORT = "judge_agreement.json"
FORM_NAME = "kalibrering-av-domaren"


class PairsRecord(BaseModel):
    """The calibration pairs, as recorded.

    Attributes:
        pilot_run: The pilot run the questions come from.
        seed: The seed of the draw.
        analyzer: The analyzer of the index that ranked the other passages.
        pairs: The pairs, in the order the form shows them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pilot_run: str
    seed: int
    analyzer: str
    pairs: tuple[JudgePair, ...]


class VerdictsRecord(BaseModel):
    """The judge's grades of the calibration pairs.

    Attributes:
        model: The judge's model.
        effort: Its effort setting.
        prompt_sha256: The hash of its prompt file.
        cost_usd: What the run cost.
        verdicts: One per pair, in the order of the pairs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    effort: str
    prompt_sha256: str
    cost_usd: float
    verdicts: tuple[JudgeVerdict, ...]


class AgreementRecord(BaseModel):
    """The judge's agreement with Martin, with every pair's two grades.

    Attributes:
        agreement: The statistics and the decision.
        grades: Martin's grade and the judge's for each pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agreement: Agreement
    grades: dict[str, tuple[int, int | None]]


def _latest_pilot(settings: Settings) -> Path:
    """The directory of the last pilot run, whose prompts the generation uses."""
    work = settings.data.search / "questions_work"
    runs = sorted(
        (path for path in work.glob("pilot-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[1]),
    )
    if not runs:
        raise SystemExit(f"No pilot run in {work}. Run 10_build_questions.py pilot first.")
    return runs[-1]


def pilot_questions(directory: Path) -> list[PilotQuestion]:
    """The questions a pilot run wrote, one per filled slot."""
    attempts: list[Attempt] = read_log(directory / ATTEMPTS_NAME)
    questions = []
    for attempt in attempts:
        if attempt.outcome != "filled" or attempt.response is None:
            continue
        questions.append(
            PilotQuestion(
                question_id=f"pilot-{attempt.slot_id}",
                text=json.loads(attempt.response)["question"],
                passage_id=attempt.passage_id,
            )
        )
    return questions


def make_pairs(settings: Settings) -> None:
    """Draw the calibration pairs and write Martin's form."""
    pilot = _latest_pilot(settings)
    analyzer = ANALYZERS[0]
    index, _ = LexicalIndex.load(lexical_dir(settings.data.search, analyzer))
    table = load_passage_table(settings.data.search / PASSAGES_NAME)
    pairs = calibration_pairs(pilot_questions(pilot), index, table, settings.judge.seed)
    record = PairsRecord(
        pilot_run=pilot.name, seed=settings.judge.seed, analyzer=analyzer.name, pairs=tuple(pairs)
    )
    write_report(record, settings.tracked.reports / PAIRS_REPORT)
    texts = load_passage_texts(
        settings.data.search / SPEECHES_NAME, {pair.passage_id for pair in pairs}
    )
    directory = settings.data.search / JUDGE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    form = directory / f"{FORM_NAME}.html"
    form.write_text(calibration_form(pairs, texts, FORM_NAME), encoding="utf-8", newline="\n")
    logger.info("%d pairs from %s; the form is %s", len(pairs), pilot.name, form)


def _pairs(settings: Settings) -> list[JudgePair]:
    """The recorded calibration pairs."""
    path = settings.tracked.reports / PAIRS_REPORT
    if not path.is_file():
        raise SystemExit(f"{path} does not exist. Run the pairs command first.")
    return list(PairsRecord.model_validate_json(path.read_text(encoding="utf-8")).pairs)


def run_judge(settings: Settings) -> None:
    """Have the judge grade the calibration pairs, and record its grades."""
    pairs = _pairs(settings)
    prompt = load_prompt("judge")
    texts = load_passage_texts(
        settings.data.search / SPEECHES_NAME, {pair.passage_id for pair in pairs}
    )
    judge = settings.judge
    requests = {
        pair.pair_id: judge_request(
            prompt,
            pair.question,
            texts[pair.passage_id],
            judge.model,
            judge.effort,
            judge.max_tokens,
        )
        for pair in pairs
    }
    completions = complete_many(requests)
    prices = settings.generation.prices[judge.model]
    verdicts = []
    for pair in pairs:
        completion = completions[pair.pair_id]
        grade, reason = read_grade(completion)
        verdicts.append(
            JudgeVerdict(
                pair_id=pair.pair_id,
                grade=grade,
                reason=reason,
                model=completion.model,
                prompt_sha256=prompt.sha256,
                stop_reason=completion.stop_reason,
                error=completion.error,
                usage=completion.usage,
                cost_usd=cost_usd(completion.usage, prices),
                created=now(),
            )
        )
    record = VerdictsRecord(
        model=judge.model,
        effort=judge.effort,
        prompt_sha256=prompt.sha256,
        cost_usd=round(sum(verdict.cost_usd for verdict in verdicts), 4),
        verdicts=tuple(verdicts),
    )
    write_report(record, settings.tracked.reports / VERDICTS_REPORT)
    graded = sum(verdict.grade is not None for verdict in verdicts)
    logger.info("The judge graded %d of %d pairs for %.4f USD", graded, len(pairs), record.cost_usd)


def compare(settings: Settings, answers: Path) -> None:
    """Compare Martin's grades with the judge's and record the agreement."""
    pairs = _pairs(settings)
    path = settings.tracked.reports / VERDICTS_REPORT
    if not path.is_file():
        raise SystemExit(f"{path} does not exist. Run the judge first.")
    judged = VerdictsRecord.model_validate_json(path.read_text(encoding="utf-8"))
    raw = json.loads(answers.read_text(encoding="utf-8"))
    martin = {entry["question_id"]: entry["grade"] for entry in raw["answers"]}
    missing = [pair.pair_id for pair in pairs if martin.get(pair.pair_id) not in (0, 1, 2)]
    if missing:
        raise SystemExit(f"These pairs have no grade from Martin: {missing}")
    judge_grades = {verdict.pair_id: verdict.grade for verdict in judged.verdicts}
    result = agreement({pair.pair_id: martin[pair.pair_id] for pair in pairs}, judge_grades)
    record = AgreementRecord(
        agreement=result,
        grades={pair.pair_id: (martin[pair.pair_id], judge_grades[pair.pair_id]) for pair in pairs},
    )
    write_report(record, settings.tracked.reports / AGREEMENT_REPORT)
    logger.info(
        "kappa %.3f against %.2f: the judge %s; linear kappa %.3f, exact %.0f%%",
        result.kappa,
        result.threshold,
        "is accepted" if result.accepted else "is not accepted",
        result.linear_kappa,
        100 * result.exact,
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
    commands.add_parser("pairs", help="Draw the pairs and write the form.")
    run = commands.add_parser("run", help="Have the judge grade the pairs.")
    run.add_argument("--confirm", action="store_true", help="Required: this calls the API.")
    compared = commands.add_parser("agreement", help="Compare Martin's grades with the judge's.")
    compared.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    if args.command == "pairs":
        make_pairs(settings)
    elif args.command == "run":
        if not args.confirm:
            raise SystemExit("The judge's run calls the API; pass --confirm.")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_judge(settings)
    else:
        compare(settings, args.answers)


if __name__ == "__main__":
    main()
