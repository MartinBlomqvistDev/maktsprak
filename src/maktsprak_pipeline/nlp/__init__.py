"""NLP utilities sub-package for MaktspråkAI.

Public API::

    from src.maktsprak_pipeline.nlp import clean_text, combined_stopwords, apply_ton_lexicon
"""

from .cleaning import clean_text, combined_stopwords, political_stopwords, swedish_stopwords
from .lexicon import apply_ton_lexicon

__all__ = [
    "clean_text",
    "swedish_stopwords",
    "political_stopwords",
    "combined_stopwords",
    "apply_ton_lexicon",
]
