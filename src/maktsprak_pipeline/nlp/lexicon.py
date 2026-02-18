"""Tone-lexicon scoring for rhetorical analysis.

Assigns weighted category scores to each speech in a DataFrame using the
``politisk_ton_lexikon.csv`` file.  Scoring is fully vectorised — no
``iterrows`` — so it scales to tens of thousands of speeches without issue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logger import get_logger

logger = get_logger()


def apply_ton_lexicon(
    df: pd.DataFrame,
    text_col: str = "text",
    lexicon_path: Path | None = None,
) -> pd.DataFrame:
    """Score each row in *df* against a weighted rhetorical tone lexicon.

    For every lexicon category, a score is computed as::

        score = (sum of word weights for matching words) / total_words * 100

    The function returns a **copy** of the input DataFrame with one new column
    per lexicon category.  The original DataFrame is never mutated.

    Args:
        df:           DataFrame containing a text column to analyse.
        text_col:     Name of the column holding the speech text.
        lexicon_path: Path to the ``politisk_ton_lexikon.csv`` file
                      (columns: ``ord``, ``kategori``, ``vikt``).
                      If ``None`` or the file does not exist, *df* is returned
                      unchanged.

    Returns:
        A copy of *df* with lexicon-category score columns appended.
    """
    if lexicon_path is None or not lexicon_path.exists():
        logger.debug("Lexicon path not available — skipping tone scoring.")
        return df.copy()

    lex_df = pd.read_csv(lexicon_path)
    lex_df.columns = pd.Index(["ord", "kategori", "vikt"])

    categories: list[str] = lex_df["kategori"].unique().tolist()
    word_weight: dict[str, float] = lex_df.set_index("ord")["vikt"].to_dict()
    cat_words: dict[str, set[str]] = {
        cat: set(lex_df.loc[lex_df["kategori"] == cat, "ord"]) for cat in categories
    }

    result = df.copy()

    # Tokenise all texts once — lower-cased, split on whitespace.
    token_series: pd.Series = result[text_col].fillna("").str.lower().str.split()

    for cat in categories:
        cat_set = cat_words[cat]

        def _score(tokens: list[str], _cat_set: set[str] = cat_set) -> float:
            if not tokens:
                return 0.0
            total_weight = sum(word_weight.get(w, 0.0) for w in tokens if w in _cat_set)
            return (total_weight / len(tokens)) * 100.0

        result[cat] = token_series.apply(_score)
        logger.debug(f"Scored category '{cat}'.")

    logger.info(f"Tone lexicon scoring complete: {len(categories)} categories, {len(result)} rows.")
    return result
