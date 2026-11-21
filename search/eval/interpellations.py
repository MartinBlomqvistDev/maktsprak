"""The interpellation set: each debate's interpellation title as a question, written by no model.

An interpellation debate is held under a heading such as "Svar på interpellation
2024/25:734 om arbetsvillkor inom gigekonomin". The heading names the
interpellation by its session and number, so it identifies the debate even
where the debate is continued later in the sitting, under "(forts. från 10 §)",
or after a break, under a trailing "(forts.)". The Riksdag's ``rel_dok_id`` does
not: in 2014/15 and 2015/16 several debates of one day share one value, and
some debates have none.

The question is the title of the interpellation: the heading's text after its
first " om ", with a capital first letter. Its gold speeches are every speech of
the debate in the corpus. A heading without " om " is left out and counted.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from search.corpus.passages import PERIOD_FIRST_YEARS
from search.eval.sampling import split_pattern
from search.models import session_start_year

ACTIVITY = "interpellationsdebatt"
ANSWER_OPENING = "Svar på interpellation"
CONTINUED_OPENING = "(forts"
CONTINUED_ENDING = " (forts.)"
TOPIC_MARKER = " om "


def normalise_heading(heading: str | None) -> str | None:
    """Strip the marks of a continued debate from a heading.

    ``"(forts. från 10 §) Svar på interpellation 2013/14:434 om stöd till
    kvinnojourer"`` and ``"Svar på interpellation 2013/14:434 om stöd till
    kvinnojourer (forts.)"`` both give the heading without the marks.

    Args:
        heading: The debate's heading as published, or ``None``.

    Returns:
        The heading without the marks, or ``None`` when there is none.
    """
    if heading is None:
        return None
    text = heading.strip()
    if text.startswith(CONTINUED_OPENING):
        _, closed, rest = text.partition(") ")
        text = rest if closed else text
    return text.removesuffix(CONTINUED_ENDING).strip() or None


def topic_of(heading: str) -> str | None:
    """Take the interpellation's title from a normalised heading.

    Args:
        heading: A heading as :func:`normalise_heading` returns it.

    Returns:
        The text after the first " om ", with a capital first letter, or
        ``None`` when the heading does not answer an interpellation or has no " om ".
    """
    if not heading.startswith(ANSWER_OPENING):
        return None
    _, marker, topic = heading.partition(TOPIC_MARKER)
    topic = topic.strip()
    if not marker or not topic:
        return None
    return topic[0].upper() + topic[1:]


@dataclass(frozen=True)
class Debate:
    """One interpellation debate.

    Attributes:
        heading: The normalised heading, which identifies the debate.
        session: The session the debate was held in.
        topic: The interpellation's title, the question.
        speeches: Its speeches in the corpus, in corpus order.
    """

    heading: str
    session: str
    topic: str
    speeches: tuple[str, ...]


def read_debates(speeches_path: Path, sessions: Sequence[str]) -> tuple[list[Debate], Counter]:
    """Gather the interpellation debates of some sessions.

    Args:
        speeches_path: ``anforanden.parquet``.
        sessions: The sessions to read.

    Returns:
        The debates, sorted by session and heading, and how many speeches were
        left out, by reason.
    """
    table = pq.read_table(
        speeches_path,
        columns=["anf_id", "rm", "title", "activity"],
        filters=(pc.field("activity") == ACTIVITY) & pc.field("rm").isin(list(sessions)),
    )
    grouped: dict[str, list[tuple[str, str]]] = {}
    left_out: Counter = Counter()
    for anf_id, session, title in zip(
        table.column("anf_id").to_pylist(),
        table.column("rm").to_pylist(),
        table.column("title").to_pylist(),
        strict=True,
    ):
        heading = normalise_heading(title)
        if heading is None:
            left_out["no heading"] += 1
            continue
        if topic_of(heading) is None:
            left_out["no title of an interpellation in the heading"] += 1
            continue
        grouped.setdefault(heading, []).append((anf_id, session))
    debates = []
    for heading, members in grouped.items():
        held_in = {session for _, session in members}
        if len(held_in) != 1:
            left_out["debate held in more than one session"] += len(members)
            continue
        debates.append(
            Debate(
                heading=heading,
                session=held_in.pop(),
                topic=topic_of(heading) or "",
                speeches=tuple(anf_id for anf_id, _ in members),
            )
        )
    debates.sort(key=lambda debate: (debate.session, debate.heading))
    return debates, left_out


def sessions_by_period(first_session: str, last_complete_session: str) -> dict[int, list[str]]:
    """The complete sessions whose two calendar years both lie in one period.

    Args:
        first_session: The corpus's first session, for example ``2003/04``.
        last_complete_session: The last session that has ended.

    Returns:
        The sessions of each period, by the period's first year. A session that
        straddles two periods, such as 2009/10, belongs to none.
    """
    first = session_start_year(first_session)
    last = session_start_year(last_complete_session)
    periods: dict[int, list[str]] = {start: [] for start in PERIOD_FIRST_YEARS}
    for year in range(first, last + 1):
        starts = [start for start in PERIOD_FIRST_YEARS if start <= year]
        ends = [start for start in PERIOD_FIRST_YEARS if start <= year + 1]
        if starts and starts[-1] == ends[-1]:
            periods[starts[-1]].append(f"{year}/{(year + 1) % 100:02d}")
    return periods


def draw_sessions(periods: dict[int, list[str]], per_period: int, seed: int) -> list[str]:
    """Draw sessions from each period.

    Args:
        periods: The sessions of each period.
        per_period: How many to draw from each.
        seed: Decides the draw.

    Returns:
        The drawn sessions, in order.
    """
    rng = random.Random(seed)
    drawn: list[str] = []
    for start in sorted(periods):
        drawn.extend(rng.sample(periods[start], per_period))
    return sorted(drawn)


def split_debates(debates: Sequence[Debate], seed: int) -> list[str]:
    """Assign debates to the splits: two of every five per session, in a random order.

    Args:
        debates: The debates, sorted by session and heading.
        seed: Decides the order within a session and the pattern's offset.

    Returns:
        The split of each debate, in the order given.
    """
    rng = random.Random(seed)
    offset = rng.randrange(5)
    splits: list[str] = [""] * len(debates)
    sessions = sorted({debate.session for debate in debates})
    for session in sessions:
        places = [place for place, debate in enumerate(debates) if debate.session == session]
        rng.shuffle(places)
        for place, split in zip(places, split_pattern(len(places), offset), strict=True):
            splits[place] = split
    return splits
