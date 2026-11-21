"""A small synthetic corpus on disk, shared by the tests of the index and the evaluation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from search.corpus.passages import pack
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_passages, write_speeches
from search.models import Anforande, deep_link, text_sha256


def speech(
    dok_id: str,
    number: int,
    paragraphs: list[str],
    *,
    party: str = "S",
    day: date = date(2015, 5, 12),
    title: str | None = None,
    activity: str | None = None,
    speaker: str = "Anna Andersson (S)",
) -> Anforande:
    """Build one speech of the synthetic corpus."""
    session_year = day.year if day.month >= 9 else day.year - 1
    return Anforande(
        anf_id=f"{dok_id}-{number}",
        dok_id=dok_id,
        rm=f"{session_year}/{(session_year + 1) % 100:02d}",
        dok_nummer=100,
        anf_nummer=number,
        date=day,
        speaker_raw=speaker,
        person_id=None,
        party=party,
        is_reply=False,
        title=title,
        activity=activity,
        rel_dok_id=None,
        layout="html",
        paragraphs=tuple(paragraphs),
        notes=(),
        text_sha256=text_sha256(paragraphs),
        url=deep_link(dok_id, number),
    )


#: Eight speeches over three periods and five parties. Packed towards 200 words,
#: each speech is one passage, ``{anf_id}:0-1``.
SPEECHES: list[Anforande] = [
    speech(
        "H209100",
        1,
        ["Kärnkraften ska byggas ut för att klara elförsörjningen.", "Vattenkraft räcker inte."],
        party="M",
        day=date(2015, 5, 12),
    ),
    speech(
        "H209100",
        2,
        ["Avveckling av kärnkraften är rätt väg.", "Sol och vind ger billig el."],
        party="MP",
        day=date(2015, 5, 12),
    ),
    speech(
        "GR0950",
        1,
        ["Skolan behöver fler lärare och mindre klasser.", "Lärarna ska få högre lön."],
        party="S",
        day=date(2004, 3, 2),
    ),
    speech(
        "GR0950",
        2,
        ["Friskolor ger valfrihet för föräldrar.", "Kommunala skolor ska också utvecklas."],
        party="M",
        day=date(2004, 3, 2),
    ),
    speech(
        "HB0960",
        1,
        ["Sjukvården behöver kortare köer.", "Fler sjuksköterskor ska anställas i regionerna."],
        party="SD",
        day=date(2023, 2, 14),
    ),
    speech(
        "HB0960",
        2,
        ["Vårdköerna har blivit längre under regeringens tid.", "Det är ett misslyckande."],
        party="V",
        day=date(2023, 2, 14),
    ),
    speech(
        "H109050",
        1,
        ["Kärnkraftsavvecklingen kostar jobb i industrin.", "Elpriset stiger."],
        party="M",
        day=date(2013, 11, 5),
    ),
    speech(
        "H109050",
        2,
        ["Skatten på arbete ska sänkas.", "Jobbskatteavdraget har gett fler jobb."],
        party="M",
        day=date(2013, 11, 5),
    ),
]


def write_corpus(directory: Path, speeches: list[Anforande] | None = None) -> tuple[Path, Path]:
    """Write the speeches and their passages, packed towards 200 words, to *directory*.

    Returns:
        The speeches file and the passages file.
    """
    speeches = SPEECHES if speeches is None else speeches
    speeches_path = directory / SPEECHES_NAME
    passages_path = directory / PASSAGES_NAME
    write_speeches(speeches_path, [speeches])
    write_passages(passages_path, [[p for item in speeches for p in pack(item, 200)]])
    return speeches_path, passages_path
