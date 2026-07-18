"""Tone-lexicon scoring for rhetorical analysis.

Assigns weighted category scores to each speech in a DataFrame using the
curated ``politisk_ton_lexikon.csv`` (columns ``ord``, ``kategori``, ``vikt``).

The scorer matches both single words (exact, case-folded token match) and
**multi-word phrases** (e.g. ``vi mot dem``, ``lag och ordning``), the latter
being essential for the populist-rhetoric markers, which are mostly phrases.
An earlier version split text on whitespace only, so every multi-word entry
silently never matched.  Scoring is vectorised, no ``iterrows``, so it scales
to tens of thousands of speeches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..logger import get_logger

logger = get_logger()


def _load_lexicon(
    lexicon_path: Path,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, list[tuple[str, float]]]]:
    """Load and normalise the lexicon into per-category single-words and phrases.

    Args:
        lexicon_path: Path to the ``ord``, ``kategori``, ``vikt`` CSV.

    Returns:
        Tuple of ``(categories, single_words, phrases)`` where
        ``single_words[cat]`` maps a lower-cased single word to its weight and
        ``phrases[cat]`` is a list of ``(lower-cased phrase, weight)`` tuples.
    """
    lex = pd.read_csv(lexicon_path)
    lex.columns = pd.Index(["ord", "kategori", "vikt"])
    # Case-fold defensively so matching never depends on CSV casing.
    lex["ord"] = lex["ord"].astype(str).str.strip().str.lower()
    lex["vikt"] = pd.to_numeric(lex["vikt"], errors="coerce").fillna(0.0)

    categories: list[str] = lex["kategori"].unique().tolist()
    single_words: dict[str, dict[str, float]] = {}
    phrases: dict[str, list[tuple[str, float]]] = {}

    for cat in categories:
        sub = lex[lex["kategori"] == cat]
        singles: dict[str, float] = {}
        phrase_list: list[tuple[str, float]] = []
        for word, weight in zip(sub["ord"], sub["vikt"], strict=False):
            if " " in word:
                phrase_list.append((word, float(weight)))
            else:
                singles[word] = float(weight)
        single_words[cat] = singles
        phrases[cat] = phrase_list

    return categories, single_words, phrases


def apply_ton_lexicon(
    df: pd.DataFrame,
    text_col: str = "text",
    lexicon_path: Path | None = None,
) -> pd.DataFrame:
    """Score each row in *df* against the weighted rhetorical tone lexicon.

    For every lexicon category, a score is computed as::

        score = (sum of matched single-word weights
                 + sum of matched phrase weights) / total_words * 100

    Single words match exact lower-cased tokens; phrases match as substrings of
    the lower-cased text.  The function returns a **copy** of the input
    DataFrame with one new column per lexicon category.  The original is never
    mutated.

    Args:
        df:           DataFrame containing a text column to analyse.
        text_col:     Name of the column holding the speech text.
        lexicon_path: Path to the lexicon CSV.  If ``None`` or the file does not
                      exist, *df* is returned unchanged.

    Returns:
        A copy of *df* with one lexicon-category score column appended.
    """
    if lexicon_path is None or not Path(lexicon_path).exists():
        logger.debug("Lexicon path not available, skipping tone scoring.")
        return df.copy()

    categories, single_words, phrases = _load_lexicon(Path(lexicon_path))
    result = df.copy()

    lowered = result[text_col].fillna("").astype(str).str.lower()
    tokens = lowered.str.split()
    # Guard against division by zero for empty speeches.
    n_tokens = tokens.str.len().clip(lower=1)

    for cat in categories:
        singles = single_words[cat]
        single_set = set(singles)

        def _single_score(
            toks: list[str], _s: dict[str, float] = singles, _k: set[str] = single_set
        ) -> float:
            return sum(_s[t] for t in toks if t in _k)

        score = tokens.apply(_single_score).astype(float)

        for phrase, weight in phrases[cat]:
            score = score + lowered.str.count(re.escape(phrase)) * weight

        result[cat] = (score / n_tokens) * 100.0
        logger.debug(f"Scored category '{cat}'.")

    logger.info(f"Tone lexicon scoring complete: {len(categories)} categories, {len(result)} rows.")
    return result
