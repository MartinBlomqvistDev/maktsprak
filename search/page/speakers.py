"""The speakers the page's speaker filter offers, one per person.

The filter searches by ``person_id``, the Riksdag's own id for a member, so a
member keeps one entry whatever title or party the record gives a speech. The
Riksdag gives 246 speeches by several speakers the id ``0000000000000``, which
names nobody, so an id of zeros is left out, as is a speech without an id.

The name shown is read from the speaker field as published, in three steps:

1. A trailing ``replik`` and the party in parentheses are dropped.
2. A leading title is dropped, word by word: words that end in ``minister``,
   ``ministern``, ``rådet``, ``talmannen`` or ``presidenten``, words that end
   in a hyphen, as in ``Utbildnings- och kulturminister``, the words
   ``biträdande``, ``vice`` and the ordinals before ``vice``, and any word in
   lower case, as in ``Minister för civilt försvar``. A name begins with a
   capital, so the first word that is none of these begins the name.
3. Until 2014 the record prints names in capitals. A form in mixed case is
   preferred when the person has one. Otherwise the name is put in title case,
   with particles such as ``von`` and ``af`` and the ``i`` of a constituency,
   as in ``Göran Persson i Stjärnhov``, in lower case.

The most common form among those left wins, and a tie goes to the first in
alphabetical order, so the list comes out the same on every run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date

from pydantic import BaseModel, ConfigDict

#: Endings of a word that is part of a title.
TITLE_ENDINGS = ("minister", "ministern", "rådet", "talmannen", "presidenten")

#: Words that join or qualify the parts of a title.
TITLE_WORDS = frozenset({"och", "vice", "förste", "andre", "tredje", "biträdande"})

#: Words a name in capitals keeps in lower case.
LOWER_WORDS = frozenset({"i", "von", "af", "de", "der", "van", "la", "du", "di", "da"})


class Speaker(BaseModel):
    """One person the filter can search by.

    Attributes:
        id: The Riksdag's ``person_id``.
        name: The name shown.
        parties: The parties the person's speeches are recorded under, in the
            site's party order.
        first: The year of the person's first speech in the corpus.
        last: The year of the latest.
        speeches: How many speeches the corpus holds by the person.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    parties: tuple[str, ...]
    first: int
    last: int
    speeches: int


def bare_name(speaker_raw: str) -> str:
    """The name in a published speaker field, without title, party or ``replik``.

    Args:
        speaker_raw: The field as published, such as
            ``Statsminister GÖRAN PERSSON (s) replik``.

    Returns:
        The name as it is written there, such as ``GÖRAN PERSSON``.
    """
    text = " ".join(speaker_raw.split())
    if text.lower().endswith(" replik"):
        text = text[: -len(" replik")]
    if text.endswith(")") and "(" in text:
        text = text[: text.rindex("(")]
    words = text.split()
    while len(words) > 1 and _is_title_word(words[0]):
        words = words[1:]
    return " ".join(words)


def _is_title_word(word: str) -> bool:
    lowered = word.lower()
    return (
        lowered in TITLE_WORDS
        or word.endswith("-")
        or lowered.endswith(TITLE_ENDINGS)
        or word[:1].islower()
    )


def is_placeholder(person_id: str | None) -> bool:
    """Whether a ``person_id`` names nobody: missing, or nothing but zeros."""
    return not person_id or set(person_id) == {"0"}


def is_capitals(name: str) -> bool:
    """Whether a name is printed in capitals, as the record printed names until 2014.

    A name counts as printed in capitals when one of its words of two letters
    or more has no lower-case letter, so ``GÖRAN PERSSON i Stjärnhov`` counts
    and ``Carl B Hamilton`` does not.
    """
    for word in name.split():
        letters = [char for char in word if char.isalpha()]
        if len(letters) > 1 and not any(char.islower() for char in letters):
            return True
    return False


def title_case(name: str) -> str:
    """Put a name printed in capitals in title case.

    Each part of a hyphenated name is capitalised, and the particles in
    :data:`LOWER_WORDS` are kept in lower case unless they begin the name.

    Args:
        name: The name in capitals, such as ``ANN-CHRISTIN AHLBERG``.

    Returns:
        The name in title case, such as ``Ann-Christin Ahlberg``.
    """
    words = []
    for position, word in enumerate(name.split()):
        lowered = word.lower()
        if position > 0 and lowered in LOWER_WORDS:
            words.append(lowered)
        else:
            words.append("-".join(part[:1].upper() + part[1:].lower() for part in word.split("-")))
    return " ".join(words)


def display_name(forms: Mapping[str, int]) -> str:
    """The name shown for one person, from the speaker fields of their speeches.

    Args:
        forms: Each published speaker field and how many speeches carry it.

    Returns:
        The most common name in mixed case, or, when the person has none, the
        most common name in capitals put in title case.
    """
    names: Counter[str] = Counter()
    for form, count in forms.items():
        names[bare_name(form)] += count
    pool: Counter[str] = Counter(
        {name: count for name, count in names.items() if not is_capitals(name)}
    )
    if not pool:
        for name, count in names.items():
            pool[title_case(name)] += count
    return min(pool, key=lambda name: (-pool[name], name))


class SpeechRow(BaseModel):
    """The fields of one speech the list is built from."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    person_id: str | None
    speaker_raw: str
    party: str
    date: date


def speakers(rows: Iterable[SpeechRow], party_order: tuple[str, ...]) -> list[Speaker]:
    """Every person with a ``person_id`` in the corpus, sorted by name and then by id.

    Args:
        rows: One row per speech.
        party_order: The site's party order, which the parties follow.

    Returns:
        One entry per person. A speech whose ``person_id`` names nobody is left out.
    """
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    parties: dict[str, set[str]] = defaultdict(set)
    years: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.person_id is None or is_placeholder(row.person_id):
            continue
        forms[row.person_id][row.speaker_raw] += 1
        parties[row.person_id].add(row.party)
        years[row.person_id].append(row.date.year)
    entries = [
        Speaker(
            id=person,
            name=display_name(forms[person]),
            parties=tuple(party for party in party_order if party in parties[person]),
            first=min(years[person]),
            last=max(years[person]),
            speeches=len(years[person]),
        )
        for person in forms
    ]
    return sorted(entries, key=lambda entry: (entry.name.casefold(), entry.id))
