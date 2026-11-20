"""Party constants for the search package.

A copy of the definitions in ``src/maktsprak_pipeline/config.py``. They are
copied because importing that package requires Supabase credentials.
``tests/search/test_parties.py`` fails when the two copies differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

#: The eight Riksdag parties the corpus covers.
VALID_PARTIES: Final[frozenset[str]] = frozenset({"C", "KD", "L", "M", "MP", "S", "SD", "V"})

#: Historical renames, applied so that one party keeps one label across time.
#: Folkpartiet (``FP``) became Liberalerna (``L``) in 2015.
PARTY_RENAMES: Final[Mapping[str, str]] = MappingProxyType({"FP": "L"})

#: Left-to-right display order, the same as on the site.
PARTY_ORDER: Final[tuple[str, ...]] = ("V", "MP", "S", "C", "L", "KD", "M", "SD")


def canonical_party(raw_party: str) -> str | None:
    """Map a party abbreviation as published to its canonical form.

    Upper-cases the abbreviation, applies :data:`PARTY_RENAMES` and validates
    the result against :data:`VALID_PARTIES`.

    Args:
        raw_party: The abbreviation as it appears in the source, in either case.

    Returns:
        The canonical party code, or ``None`` when the abbreviation is not one
        of the eight parties.
    """
    party = raw_party.strip().upper()
    party = PARTY_RENAMES.get(party, party)
    return party if party in VALID_PARTIES else None
