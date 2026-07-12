"""NLP utilities sub-package for MaktspråkAI.

Public API::

    from src.maktsprak_pipeline.nlp import clean_text, combined_stopwords, apply_ton_lexicon
"""

from .cleaning import clean_text, combined_stopwords, political_stopwords, swedish_stopwords
from .distinctiveness import (
    PARTY_NAME_STOPWORDS,
    POLITICIAN_NAME_STOPWORDS,
    distinctive_words,
    group_token_counts,
    speaker_name_stopwords,
    tokenize,
    weighted_log_odds,
    wordcloud_frequencies,
)
from .drift import (
    ISSUE_FRAMES,
    add_year,
    frame_trajectories,
    party_divergence_by_year,
    term_trajectories,
    top_movers,
    yearly_signatures,
)
from .lexicon import apply_ton_lexicon

__all__ = [
    "clean_text",
    "swedish_stopwords",
    "political_stopwords",
    "combined_stopwords",
    "apply_ton_lexicon",
    "tokenize",
    "group_token_counts",
    "weighted_log_odds",
    "distinctive_words",
    "wordcloud_frequencies",
    "PARTY_NAME_STOPWORDS",
    "add_year",
    "top_movers",
    "term_trajectories",
    "party_divergence_by_year",
    "yearly_signatures",
    "frame_trajectories",
    "ISSUE_FRAMES",
    "POLITICIAN_NAME_STOPWORDS",
    "speaker_name_stopwords",
]
