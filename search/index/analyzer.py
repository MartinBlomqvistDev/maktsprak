"""Turning text into the terms the lexical index holds.

Text is lower-cased and split into runs of letters and digits; everything else,
hyphens included, separates two tokens. An analyzer may then drop stop words
and stem what remains with the Snowball stemmer for Swedish. Stop words are
dropped before stemming, so the list is matched against the words as written.

The same analyzer turns passages into index terms and questions into query
terms, and the check on generated questions uses the stemming one, so a word
counts as the same word wherever the evaluation compares two texts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import snowballstemmer
from pydantic import BaseModel, ConfigDict

#: The Snowball project's stop words for Swedish, in its own order.
STOPWORDS_FILE: Path = Path(__file__).resolve().with_name("stopwords_sv.txt")

#: A token is a run of letters and digits. ``\w`` alone would also take the underscore.
_TOKEN = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Lower-case *text* and split it into runs of letters and digits.

    Args:
        text: Any text.

    Returns:
        The tokens, in order. ``"EU-medlemskapet, år 2003!"`` gives
        ``["eu", "medlemskapet", "år", "2003"]``.
    """
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """Read the stop words, one per line; lines starting with ``#`` are comments.

    Returns:
        The stop words, lower-case.
    """
    lines = STOPWORDS_FILE.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


@lru_cache(maxsize=1)
def _stemmer() -> Any:
    """Build the Swedish stemmer once. With PyStemmer installed it is the C implementation."""
    return snowballstemmer.stemmer("swedish")


@lru_cache(maxsize=500_000)
def stem(token: str) -> str:
    """Stem one lower-case token with the Snowball stemmer for Swedish.

    Args:
        token: A token as :func:`tokenize` returns it.

    Returns:
        The stem. ``"kärnkraftsavvecklingen"`` gives ``"kärnkraftsavveckling"``.
    """
    return _stemmer().stemWord(token)


def stem_all(tokens: Iterable[str]) -> list[str]:
    """Stem many tokens in one call, which is faster than one call each.

    Args:
        tokens: Tokens as :func:`tokenize` returns them.

    Returns:
        Their stems, in the same order.
    """
    return _stemmer().stemWords(list(tokens))


class Analyzer(BaseModel):
    """How text becomes terms: whether stop words are dropped and whether tokens are stemmed.

    Attributes:
        stem: Whether each token is reduced to its Snowball stem.
        stopwords: Whether the Snowball stop words are dropped first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stem: bool
    stopwords: bool

    @property
    def name(self) -> str:
        """A short name for the analyzer, used for its directory on disk."""
        return f"{'stem' if self.stem else 'nostem'}-{'stop' if self.stopwords else 'nostop'}"

    def term_of(self, token: str) -> str | None:
        """Turn one token into its term, or ``None`` when the token is a dropped stop word.

        Args:
            token: A token as :func:`tokenize` returns it.

        Returns:
            The term, or ``None``.
        """
        if self.stopwords and token in stopwords():
            return None
        return stem(token) if self.stem else token

    def terms(self, text: str) -> list[str]:
        """Turn text into its terms, in order, repeats kept.

        Args:
            text: Any text.

        Returns:
            The terms.
        """
        found = (self.term_of(token) for token in tokenize(text))
        return [term for term in found if term is not None]


#: The four analyzers that the tuning on the development split chooses between.
ANALYZERS: tuple[Analyzer, ...] = tuple(
    Analyzer(stem=stem_on, stopwords=stop_on)
    for stem_on in (True, False)
    for stop_on in (True, False)
)

#: The analyzer whose stems define a rare word in the check on generated questions.
STEMMING = Analyzer(stem=True, stopwords=False)
