"""Distinctive-vocabulary scoring via the Fightin' Words method.

Monroe, Colaresi & Quinn (2008), *Fightin' Words: Lexical Feature Selection
and Evaluation for Identifying the Content of Political Conflict.*

A raw word-frequency cloud surfaces the same generic political vocabulary for
every party — the words common to all of them dominate everyone's cloud
equally, so nothing distinguishes the parties.  This module instead computes a
**weighted log-odds-ratio with an informative Dirichlet prior**: for each
party it scores every word by how over-represented it is relative to the other
parties combined, with rare words shrunk toward the shared background so they
can't top the ranking on noise alone.  The result is each party's *distinctive*
vocabulary — exactly what a "rhetorical fingerprint" word cloud should show.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .cleaning import combined_stopwords

# Latin-1 letters (a-z plus the accented range: å ä ö é ü …); drops digits and
# punctuation.  The é matters: without it "moské" tokenizes to "mosk", and every
# form ("moskén", "moskéer") collapses to the same truncated stem.
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+")

# Party self-references (and their morphological variants) are trivially
# distinctive — every party names itself and its rivals — but tell you nothing
# about *rhetoric*.  Filter them so the distinctiveness clouds surface subject
# matter and framing rather than letterhead.
PARTY_NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        # Socialdemokraterna
        "socialdemokrat",
        "socialdemokrater",
        "socialdemokraterna",
        "socialdemokraternas",
        "socialdemokratisk",
        "socialdemokratiska",
        "socialdemokraten",
        "sossar",
        "sosse",
        # Moderaterna
        "moderat",
        "moderater",
        "moderaterna",
        "moderaternas",
        "moderata",
        "moderatledd",
        "moderatledda",
        "moderaten",
        # Sverigedemokraterna
        "sverigedemokrat",
        "sverigedemokrater",
        "sverigedemokraterna",
        "sverigedemokraternas",
        "sverigedemokratisk",
        "sverigedemokratiska",
        "sverigedemokraten",
        # Vänsterpartiet
        "vänsterparti",
        "vänsterpartiet",
        "vänsterpartiets",
        "vänsterpartist",
        "vänsterpartister",
        "vänsterpartisterna",
        # Miljöpartiet
        "miljöparti",
        "miljöpartiet",
        "miljöpartiets",
        "miljöpartist",
        "miljöpartister",
        "miljöpartisterna",
        # Centerpartiet
        "centerparti",
        "centerpartiet",
        "centerpartiets",
        "centerpartist",
        "centerpartister",
        "centerpartisterna",
        # Kristdemokraterna
        "kristdemokrat",
        "kristdemokrater",
        "kristdemokraterna",
        "kristdemokraternas",
        "kristdemokratisk",
        "kristdemokratiska",
        "kristdemokraten",
        # Liberalerna (keep the ideological adjectives "liberal"/"liberala")
        "liberaler",
        "liberalerna",
        "liberalernas",
        "liberalen",
    }
)

# Default stop-word set for distinctiveness scoring: the general Swedish +
# political list, plus the party self-names above.
_DISTINCT_STOPWORDS: frozenset[str] = combined_stopwords | PARTY_NAME_STOPWORDS


def tokenize(
    text: str,
    min_length: int = 3,
    stopwords: frozenset[str] = combined_stopwords,
) -> list[str]:
    """Lower-case, extract word tokens, and drop stop words and short tokens."""
    if not isinstance(text, str):
        return []
    return [
        w for w in _TOKEN_RE.findall(text.lower()) if len(w) >= min_length and w not in stopwords
    ]


def group_token_counts(
    df: pd.DataFrame,
    group_col: str = "party",
    text_col: str = "text",
    min_length: int = 3,
    stopwords: frozenset[str] = _DISTINCT_STOPWORDS,
) -> dict[str, Counter[str]]:
    """Aggregate token counts per group (e.g. per party).

    Args:
        df:         DataFrame with a group column and a text column.
        group_col:  Column identifying the group (party).
        text_col:   Column holding the speech text.
        min_length: Minimum token length to keep.
        stopwords:  Words to drop before counting.

    Returns:
        Mapping of group value → ``Counter`` of token → count.
    """
    counts: dict[str, Counter[str]] = {}
    for group, sub in df.groupby(group_col, observed=True):
        counter: Counter[str] = Counter()
        for text in sub[text_col].dropna():
            counter.update(tokenize(text, min_length=min_length, stopwords=stopwords))
        counts[str(group)] = counter
    return counts


def weighted_log_odds(
    group_counts: dict[str, Counter[str]],
    alpha: float = 0.01,
    min_count: int = 5,
) -> dict[str, dict[str, float]]:
    """Fightin' Words z-scores: each group vs. the pooled rest.

    Args:
        group_counts: Mapping of group → token counts (from
            :func:`group_token_counts`).
        alpha:        Informative-prior scale.  The Dirichlet prior for a word
            is ``alpha * background_count``, so rare words are shrunk toward the
            corpus mean.  Larger ``alpha`` = stronger shrinkage.
        min_count:    Ignore words occurring fewer than this many times in the
            whole corpus (removes hapax noise from the vocabulary).

    Returns:
        Mapping of group → {word → z-score}.  Positive z means the word is
        over-represented in that group relative to the others.
    """
    background: Counter[str] = Counter()
    for counter in group_counts.values():
        background.update(counter)

    vocab = [w for w, n in background.items() if n >= min_count]
    if not vocab:
        return {g: {} for g in group_counts}

    bg = np.array([background[w] for w in vocab], dtype=float)
    prior = alpha * bg  # informative Dirichlet prior ∝ background frequency
    a0 = prior.sum()

    results: dict[str, dict[str, float]] = {}
    for group, counter in group_counts.items():
        y_i = np.array([counter.get(w, 0) for w in vocab], dtype=float)
        n_i = y_i.sum()
        y_j = bg - y_i  # the pooled rest
        n_j = bg.sum() - n_i

        # log-odds of word w in group i vs. the rest, each smoothed by the prior
        log_odds_i = np.log((y_i + prior) / (n_i + a0 - y_i - prior))
        log_odds_j = np.log((y_j + prior) / (n_j + a0 - y_j - prior))
        delta = log_odds_i - log_odds_j
        variance = 1.0 / (y_i + prior) + 1.0 / (y_j + prior)
        z = delta / np.sqrt(variance)

        results[group] = dict(zip(vocab, z.tolist(), strict=False))
    return results


def distinctive_words(
    scores: dict[str, dict[str, float]],
    group: str,
    top_n: int = 100,
) -> list[tuple[str, float]]:
    """Return a group's top *top_n* over-represented words, highest z first.

    Args:
        scores:  Output of :func:`weighted_log_odds`.
        group:   Which group to rank for.
        top_n:   How many words to return.

    Returns:
        List of ``(word, z_score)`` with ``z_score > 0``, descending.
    """
    ranked = sorted(scores.get(group, {}).items(), key=lambda kv: kv[1], reverse=True)
    return [(word, z) for word, z in ranked[:top_n] if z > 0]


def wordcloud_frequencies(
    scores: dict[str, dict[str, float]],
    group: str,
    top_n: int = 100,
) -> dict[str, float]:
    """Distinctive words as positive weights for ``WordCloud.generate_from_frequencies``.

    Word size is proportional to distinctiveness (z-score), not raw frequency,
    so the cloud shows what sets the party apart.
    """
    return {word: z for word, z in distinctive_words(scores, group, top_n=top_n)}
