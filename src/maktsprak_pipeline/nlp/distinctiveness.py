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

# Runs of 2+ consecutive uppercase letters — how Riksdag protocols render a
# speaker's actual name ("JOSEF FRANSSON"), as opposed to their title, which is
# mixed case ("Statsminister", "Näringsminister").
_CAPS_RUN_RE = re.compile(r"[A-ZÅÄÖ]{2,}")


def speaker_name_stopwords(
    speakers: pd.Series | list[str],
    min_length: int = 3,
    exclude: frozenset[str] = combined_stopwords,
) -> frozenset[str]:
    """Derive a name stopword set directly from the corpus's own speaker labels.

    ``POLITICIAN_NAME_STOPWORDS`` below is a manually curated list — it only
    covers the handful of ministers and party leaders someone happened to
    notice leaking into a sample.  With ~1,600 unique speakers across
    2002-2026, manual curation cannot keep up (older years, in particular,
    surface many names nobody has reviewed).  This instead reads the corpus's
    own ``speaker`` column, which is ground truth for exactly one thing: every
    person who actually spoke.

    The extraction exploits how the protocols render names: the person's name
    is in ALL CAPS ("JOSEF FRANSSON"); a title, when present, is mixed case
    ("Statsminister", "Näringsminister") and is skipped automatically because
    it never matches the all-caps run.  Tokens already present in *exclude*
    are dropped — this is what keeps ordinary words that double as first names
    safe (e.g. a speaker surnamed "Hans" would otherwise reintroduce "hans" =
    "his" as a stopword; "hans" is already in ``combined_stopwords``, so it's
    excluded here for free, with no manual allowlist needed).

    Args:
        speakers:   The corpus's ``speaker`` column (or any iterable of raw
            speaker strings).
        min_length: Minimum token length to keep.
        exclude:    Tokens that must never be added, even if they appear in an
            all-caps name run. Defaults to the general stopword list.

    Returns:
        Lower-cased name tokens (given names, surnames, hyphenated parts) safe
        to use as an additional ``stopwords`` set.
    """
    names: set[str] = set()
    for raw in speakers:
        if not isinstance(raw, str):
            continue
        for run in _CAPS_RUN_RE.findall(raw):
            for word in _TOKEN_RE.findall(run.lower()):
                if len(word) >= min_length and word not in exclude:
                    names.add(word)
    return frozenset(names)

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
        # Liberalerna (keep the ideological adjectives "liberal"/"liberala").
        # Pre-2015 protocols use the party's old name, Folkpartiet.
        "liberaler",
        "liberalerna",
        "liberalernas",
        "liberalen",
        "folkparti",
        "folkpartiet",
        "folkpartiets",
        "folkpartist",
        "folkpartister",
    }
)

# Politician given/family names.  A politician's name is trivially
# "distinctive" — to their own party (they're quoted, they speak in first
# person, allies reference them) and to whichever years they held office — but
# it's not rhetoric, just a proper noun.  Left unfiltered it crowds out actual
# subject matter in both the per-party fingerprints and the time-based drift
# analyses (top_movers, yearly_signatures).  This is a best-effort, manually
# curated v1 covering ministers and party leaders active 2002-2026.  Names
# that double as ordinary Swedish words ("hans" = his, "bo" = to live,
# "sten" = stone, "per"/"jan" as loose fragments) are deliberately left out to
# avoid false-positive filtering.  A systematic (NER-based) filter is a
# future improvement.
POLITICIAN_NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        # surnames (low collision risk)
        "löfven", "wallström", "hallengren", "kinberg", "batra", "svantesson",
        "forssell", "strömmer", "stenergard", "roswall", "pourmokhtari",
        "sabuni", "reinfeldt", "romson", "fridolin", "sjöstedt", "dadgostar",
        "damberg", "hultqvist", "ygeman", "shekarabi", "kristersson",
        "åkesson", "lööf", "wikström", "regnér", "fritzon", "barenfeld",
        "beckman", "björnsdotter", "brandberg", "thorell", "edholm", "britz",
        "busch", "pehrson", "carlson", "bolund", "stenevi", "dousa", "linde",
        "borg", "smith", "schulte", "strandhäll", "johansson", "morgan",
        "odell", "littorin", "björck", "olofsson",
        # given names (checked for common-word collisions; ambiguous ones like
        # "hans" [=his], "bo" [=live], "sten" [=stone], "per"/"jan" excluded)
        "erik", "stefan", "jonas", "margot", "ylva", "isabella", "roger",
        "gustav", "sven", "johan", "karin", "maria", "lars", "mikael",
        "fredrik", "magdalena", "ulf", "jimmie", "annie", "nooshi", "peter",
        "thomas", "carl", "göran", "mona", "lena", "elisabeth", "kristina",
        "catarina", "monica", "birgitta", "gunnar", "bengt", "nils", "olof",
        "johanna", "emma", "sofia", "daniel", "david", "markus", "rickard",
        "teresa", "serkan", "benjamin", "edward", "tobias", "ebba", "paulina",
        "linus", "gabriel", "helene", "anders", "anette", "romina", "ann",
        "alexandra", "märta", "patrik", "maud",
    }
)

# Default stop-word set for distinctiveness scoring: the general Swedish +
# political list, plus the party self-names and curated politician names above.
_DISTINCT_STOPWORDS: frozenset[str] = (
    combined_stopwords | PARTY_NAME_STOPWORDS | POLITICIAN_NAME_STOPWORDS
)


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
