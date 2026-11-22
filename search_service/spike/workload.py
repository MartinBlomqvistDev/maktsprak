"""The questions the measurements send, and the filters they are sent under.

Every question comes from the development split of the question sets; nothing
from the test split is read into a workload. A filtered question keeps its own
filter. Every other question gets one drawn with the weights in ``spike.yaml``:
a party, a period, one speaker, or a party and a period together. A speaker is
drawn by drawing a passage, so speakers are drawn as often as they hold
passages, and a party and a period are drawn again until they allow a passage.
Each question's synthetic vector has a seed derived from its id.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.corpus.passages import PERIOD_FIRST_YEARS
from search.eval.questions import EvalQuestion, read_question_set, set_file_name, split_of
from search.parties import PARTY_ORDER
from search_service.spike.contracts import SearchFilter
from search_service.spike.inputs import PassageColumns, day_number, day_of
from search_service.spike.settings import FilterKind, WorkloadSettings

QueryFilterKind = Literal["own", "party", "period", "speaker", "party_period", "none"]


class WorkloadQuery(BaseModel):
    """One question as it is sent.

    Attributes:
        query_id: The question's id.
        question_set: The set it comes from.
        text: The question.
        filter: The filter it is searched under.
        filter_kind: Where the filter came from: the question's own, a drawn
            kind, or none.
        vector_seed: The seed of its synthetic vector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    question_set: str
    text: str
    filter: SearchFilter
    filter_kind: QueryFilterKind
    vector_seed: int


class Workload(BaseModel):
    """Everything one store is sent, in order.

    Attributes:
        warmup: Sent first and not counted.
        queries: The measured searches.
        party_view: The questions of the view of the eight parties, without filters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    warmup: tuple[WorkloadQuery, ...]
    queries: tuple[WorkloadQuery, ...]
    party_view: tuple[WorkloadQuery, ...]


def vector_seed(question_id: str, seed: int) -> int:
    """The seed of a question's synthetic vector, from its id and the workload's seed."""
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def period_filter(first_year: int, latest_year: int) -> SearchFilter:
    """The filter of the period that starts in *first_year*, as the breakdowns cut the periods.

    Args:
        first_year: One of the periods' first years.
        latest_year: The latest year among the speeches, which ends the last period.

    Returns:
        A filter from 1 January of the first year to 31 December of the last.
    """
    later = [year for year in PERIOD_FIRST_YEARS if year > first_year]
    last = min(later) - 1 if later else latest_year
    return SearchFilter(start=date(first_year, 1, 1), end=date(last, 12, 31))


def own_filter(question: EvalQuestion) -> SearchFilter:
    """The filter of a filtered question: its party, or its period such as ``2010-2017``.

    Raises:
        ValueError: If the question has no filter.
    """
    if question.filter is None:
        raise ValueError(f"{question.question_id} has no filter of its own.")
    if question.filter.party is not None:
        return SearchFilter(party=question.filter.party)
    first, last = (int(year) for year in (question.filter.period or "").split("-"))
    return SearchFilter(start=date(first, 1, 1), end=date(last, 12, 31))


class FilterDraw:
    """Draws filters of each kind over the passages' fields."""

    def __init__(self, columns: PassageColumns, rng: random.Random) -> None:
        self.columns = columns
        self.rng = rng
        self.parties = np.array(columns.parties)
        self.latest_year = day_of(int(columns.days.max())).year

    def allows_any(self, search_filter: SearchFilter) -> bool:
        """Whether at least one passage passes *search_filter*."""
        allowed = np.ones(len(self.columns), dtype=bool)
        if search_filter.party is not None:
            allowed &= self.parties == search_filter.party
        if search_filter.start is not None:
            allowed &= self.columns.days >= day_number(search_filter.start)
        if search_filter.end is not None:
            allowed &= self.columns.days <= day_number(search_filter.end)
        return bool(allowed.any())

    def period(self) -> SearchFilter:
        """A period, each equally likely."""
        return period_filter(self.rng.choice(PERIOD_FIRST_YEARS), self.latest_year)

    def draw(self, kind: FilterKind) -> SearchFilter:
        """A filter of one kind.

        Raises:
            ValueError: If no passage names a speaker, so no speaker can be drawn.
        """
        if kind == "party":
            return SearchFilter(party=self.rng.choice(PARTY_ORDER))
        if kind == "period":
            return self.period()
        if kind == "speaker":
            if all(person is None for person in self.columns.person_ids):
                raise ValueError("No passage names a speaker.")
            while True:
                person = self.columns.person_ids[self.rng.randrange(len(self.columns))]
                if person is not None:
                    return SearchFilter(person_id=person)
        while True:
            period = self.period()
            combined = SearchFilter(
                party=self.rng.choice(PARTY_ORDER), start=period.start, end=period.end
            )
            if self.allows_any(combined):
                return combined


def _query(
    question: EvalQuestion, search_filter: SearchFilter, kind: QueryFilterKind, seed: int
) -> WorkloadQuery:
    return WorkloadQuery(
        query_id=question.question_id,
        question_set=question.question_set,
        text=question.text,
        filter=search_filter,
        filter_kind=kind,
        vector_seed=vector_seed(question.question_id, seed),
    )


def _dev_questions(questions_dir: Path, question_set: str) -> list[EvalQuestion]:
    questions = split_of(read_question_set(questions_dir / set_file_name(question_set)), "dev")
    if any(question.split != "dev" for question in questions):
        raise ValueError("A workload holds questions of the development split only.")
    return questions


def build_workload(
    settings: WorkloadSettings, questions_dir: Path, columns: PassageColumns
) -> Workload:
    """Build the workload: the same questions, filters and order for every store.

    Args:
        settings: The workload's settings.
        questions_dir: The directory of the question sets.
        columns: The passages' fields, to draw speakers from and to check filters against.

    Returns:
        The workload.
    """
    rng = random.Random(settings.seed)
    draw = FilterDraw(columns, rng)
    kinds = list(settings.filters)
    weights = [settings.filters[kind] for kind in kinds]

    def drawn(question: EvalQuestion) -> WorkloadQuery:
        kind = rng.choices(kinds, weights)[0]
        return _query(question, draw.draw(kind), kind, settings.seed)

    queries = []
    for question_set in settings.sets:
        for question in _dev_questions(questions_dir, question_set):
            if question.filter is not None:
                queries.append(_query(question, own_filter(question), "own", settings.seed))
            else:
                queries.append(drawn(question))
    rng.shuffle(queries)

    warmup_pool = _dev_questions(questions_dir, settings.warmup_set)
    warmup = [
        drawn(q) for q in rng.sample(warmup_pool, min(settings.warmup_queries, len(warmup_pool)))
    ]

    natural = _dev_questions(questions_dir, "natural")
    chosen = rng.sample(natural, min(settings.party_view_questions, len(natural)))
    party_view = [_query(question, SearchFilter(), "none", settings.seed) for question in chosen]
    return Workload(warmup=tuple(warmup), queries=tuple(queries), party_view=tuple(party_view))
