"""Building the question sets from a generation's logs and the corpus, and recording them.

The generated sets come from the filled slots of one generation. The
unanswerable set comes from the same generation's unanswerable requests, and
the interpellation set from the corpus alone. Each set is written as JSON Lines
and hashed; the manifest records the hashes and everything that decided the
sets, for the addendum to the pre-registration.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from search.config import Settings, UnanswerableKind
from search.corpus.table import PassageTable
from search.eval.generation import STANCE_DIRECTIONS, Attempt
from search.eval.interpellations import Debate, split_debates
from search.eval.questions import (
    QUESTION_SETS,
    EvalQuestion,
    Provenance,
    QuestionFilter,
    QuestionSet,
    set_file_name,
    write_question_set,
)
from search.eval.sampling import Slot, split_pattern
from search.eval.unanswerable import KINDS, UnanswerableRequest, accepted

MANIFEST_NAME = "questions_manifest.json"


class SetRecord(BaseModel):
    """One question set as written.

    Attributes:
        file: The file's name.
        questions: How many questions it holds.
        dev: How many of them are in the development split.
        test: How many are in the test split.
        sha256: The file's hash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    questions: int
    dev: int
    test: int
    sha256: str


class QuestionsManifest(BaseModel):
    """What decided the question sets, and the hash of each.

    Attributes:
        generation: Which of the at most two generations the sets come from.
        seed: The seed of the draw and the split.
        model: The model that wrote the generated sets.
        effort: Its effort setting.
        prompts: The hash of each prompt file used, by name.
        rare_share: A stem held by fewer than this share of the passages is rare.
        rare_below: The same bound as a count of passages.
        length_band_edges: The passage lengths, in words, that cut the length bands.
        slots: How many slots each set had.
        filled: How many of them were filled.
        requests_made: How many passage requests the generation sent.
        attempts: How many of the requests that count ended in each outcome,
            under the checks as they stand.
        outcomes_changed: How many logged answers the checks as they stand
            judge otherwise than the checks of the run did.
        passages_changed: How many slots got another passage than the run gave them.
        passages_tried: How many distinct passages the requests that count asked about.
        unanswerable_turned_away: How many unanswerable questions failed each check, by kind.
        interpellation_sessions: The sessions the interpellation set was drawn from.
        interpellation_left_out: Speeches of those sessions left out, by reason.
        cost_usd: What the generation cost.
        sets: Each set as written, by name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    generation: int
    seed: int
    model: str
    effort: str
    prompts: dict[str, str]
    rare_share: float
    rare_below: float
    length_band_edges: tuple[int, ...]
    slots: dict[str, int]
    filled: dict[str, int]
    requests_made: int
    attempts: dict[str, int]
    outcomes_changed: int
    passages_changed: int
    passages_tried: int
    unanswerable_turned_away: dict[str, dict[str, int]]
    interpellation_sessions: tuple[str, ...]
    interpellation_left_out: dict[str, int]
    cost_usd: float
    sets: dict[str, SetRecord]


def _number(slot_id: str) -> str:
    """The number part of a slot id: ``natural-0017`` gives ``0017``."""
    return slot_id.rpartition("-")[2]


def passage_questions(
    slots: Sequence[Slot],
    attempts: Sequence[Attempt],
    table: PassageTable,
    generation: int,
) -> dict[QuestionSet, list[EvalQuestion]]:
    """Turn the filled slots into the natural, keyword, filtered and stance questions.

    Args:
        slots: The slots of the generation.
        attempts: Its log.
        table: The passages.
        generation: Which generation it is.

    Returns:
        The questions, by set.
    """
    filled = {attempt.slot_id: attempt for attempt in attempts if attempt.outcome == "filled"}
    sets: dict[QuestionSet, list[EvalQuestion]] = {
        name: [] for name in ("natural", "keyword", "filtered", "stance")
    }
    for slot in slots:
        attempt = filled.get(slot.slot_id)
        if attempt is None or attempt.response is None:
            continue
        answer = json.loads(attempt.response)
        position = table.position(attempt.passage_id)
        speech = table.anf_ids[position]
        provenance = Provenance(
            model=attempt.model or "",
            prompt_sha256=attempt.prompt_sha256,
            created=attempt.created.date(),
            generation=generation,
        )
        common = {
            "split": slot.split,
            "gold": (speech,),
            "passage_id": attempt.passage_id,
            "provenance": provenance,
        }
        if slot.question_set == "stance":
            sets["stance"].append(
                EvalQuestion(
                    question_id=slot.slot_id,
                    question_set="stance",
                    text=answer["question"],
                    kind=STANCE_DIRECTIONS[answer["direction"]],
                    **common,
                )
            )
            continue
        if slot.question_set == "filtered":
            if slot.filter_kind == "party":
                question_filter = QuestionFilter(party=str(table.parties[position]))
            else:
                question_filter = QuestionFilter(
                    period=table.period_of_year(int(table.years[position]))
                )
            sets["filtered"].append(
                EvalQuestion(
                    question_id=slot.slot_id,
                    question_set="filtered",
                    text=answer["question"],
                    filter=question_filter,
                    **common,
                )
            )
            continue
        sets["natural"].append(
            EvalQuestion(
                question_id=slot.slot_id, question_set="natural", text=answer["question"], **common
            )
        )
        sets["keyword"].append(
            EvalQuestion(
                question_id=f"keyword-{_number(slot.slot_id)}",
                question_set="keyword",
                text=" ".join(answer["keywords"]),
                **common,
            )
        )
    return sets


def unanswerable_questions(
    requests: Sequence[UnanswerableRequest],
    keep: Mapping[UnanswerableKind, int],
    speaker_names: Sequence[str],
    generation: int,
    seed: int,
) -> tuple[list[EvalQuestion], dict[str, dict[str, int]]]:
    """Keep the unanswerable questions that pass the checks, and split each kind two in five.

    Args:
        requests: The generation's unanswerable requests.
        keep: How many of each kind to keep.
        speaker_names: Every speaker in the corpus as published, lower-case.
        generation: Which generation it is.
        seed: Decides the split pattern's offset.

    Returns:
        The questions, and how many were turned away for each reason, by kind.
    """
    offset = random.Random(seed).randrange(5)
    questions: list[EvalQuestion] = []
    turned_away: dict[str, dict[str, int]] = {}
    for kind in KINDS:
        kept, refused = accepted(kind, requests, keep[kind], speaker_names)
        turned_away[kind] = refused
        for (text, request), split in zip(kept, split_pattern(len(kept), offset), strict=True):
            questions.append(
                EvalQuestion(
                    question_id=f"unanswerable-{len(questions) + 1:04d}",
                    question_set="unanswerable",
                    split=split,
                    text=text,
                    gold=(),
                    kind=kind,
                    provenance=Provenance(
                        model=request.model or "",
                        prompt_sha256=request.prompt_sha256,
                        created=request.created.date(),
                        generation=generation,
                    ),
                )
            )
    return questions, turned_away


def interpellation_questions(debates: Sequence[Debate], seed: int) -> list[EvalQuestion]:
    """Turn the debates into questions, split two in five within each session.

    Args:
        debates: The debates, sorted by session and heading.
        seed: Decides the split.

    Returns:
        The questions.
    """
    splits = split_debates(debates, seed)
    return [
        EvalQuestion(
            question_id=f"interpellation-{number:04d}",
            question_set="interpellation",
            split=split,
            text=debate.topic,
            gold=debate.speeches,
            session=debate.session,
        )
        for number, (debate, split) in enumerate(zip(debates, splits, strict=True), start=1)
    ]


def write_sets(
    directory: Path, sets: Mapping[QuestionSet, Sequence[EvalQuestion]]
) -> dict[str, SetRecord]:
    """Write every set and record its file, counts and hash.

    Args:
        directory: Where the files go.
        sets: The questions, by set; every set of :data:`QUESTION_SETS` must be present.

    Returns:
        The record of each set, by name.

    Raises:
        ValueError: If a set is missing.
    """
    missing = [name for name in QUESTION_SETS if name not in sets]
    if missing:
        raise ValueError(f"These sets are missing: {missing}.")
    records = {}
    for name in QUESTION_SETS:
        questions = sets[name]
        file = set_file_name(name)
        splits = Counter(question.split for question in questions)
        records[name] = SetRecord(
            file=file,
            questions=len(questions),
            dev=splits["dev"],
            test=splits["test"],
            sha256=write_question_set(directory / file, questions),
        )
    return records


def manifest_of(
    *,
    settings: Settings,
    generation: int,
    prompts: Mapping[str, str],
    rare_below: float,
    band_edges: Sequence[int],
    slots: Sequence[Slot],
    made: Sequence[Attempt],
    judged: Sequence[Attempt],
    attempts: Sequence[Attempt],
    requests: Sequence[UnanswerableRequest],
    turned_away: Mapping[str, Mapping[str, int]],
    sessions: Sequence[str],
    left_out: Mapping[str, int],
    records: Mapping[str, SetRecord],
) -> QuestionsManifest:
    """Gather the manifest of a generation's question sets.

    Args:
        made: The log as the run wrote it.
        judged: The same log, judged again with the checks as they stand.
        attempts: The requests that count, from :func:`search.eval.generation.simulate`.

    The other arguments are recorded as given.
    """
    filled_slots = {attempt.slot_id for attempt in attempts if attempt.outcome == "filled"}
    slot_counts = Counter(slot.question_set for slot in slots)
    filled_counts = Counter(slot.question_set for slot in slots if slot.slot_id in filled_slots)
    run_gave = {
        attempt.slot_id: attempt.passage_id for attempt in made if attempt.outcome == "filled"
    }
    now_gives = {
        attempt.slot_id: attempt.passage_id for attempt in attempts if attempt.outcome == "filled"
    }
    return QuestionsManifest(
        generation=generation,
        seed=settings.questions.seed,
        model=settings.generation.model,
        effort=settings.generation.effort,
        prompts=dict(sorted(prompts.items())),
        rare_share=settings.questions.rare_share,
        rare_below=rare_below,
        length_band_edges=tuple(band_edges),
        slots=dict(sorted(slot_counts.items())),
        filled=dict(sorted(filled_counts.items())),
        requests_made=len(made),
        attempts=dict(sorted(Counter(attempt.outcome for attempt in attempts).items())),
        outcomes_changed=sum(
            before.outcome != after.outcome for before, after in zip(made, judged, strict=True)
        ),
        passages_changed=sum(
            run_gave.get(slot_id) != passage_id for slot_id, passage_id in now_gives.items()
        ),
        passages_tried=len({attempt.passage_id for attempt in attempts}),
        unanswerable_turned_away={kind: dict(found) for kind, found in turned_away.items()},
        interpellation_sessions=tuple(sessions),
        interpellation_left_out=dict(sorted(left_out.items())),
        cost_usd=round(sum(entry.cost_usd for entry in [*made, *requests]), 4),
        sets=dict(records),
    )
