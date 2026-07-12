"""Temporal drift of political language: what the debate moved toward and away.

This is the neutral, data-driven half of the analysis.  It does not anchor on
any single thesis (migration, the SD turn, ...); it surfaces *whatever* shifted.
Three lenses, all computed straight from the speech corpus:

- :func:`top_movers` — the words that rose or fell most across the window,
  scored with the **Fightin' Words** method (Monroe et al. 2008) but with *time*
  as the grouping axis (early era vs late era) instead of party.  This reuses the
  distinctiveness machinery: a word "distinctive to the late era" is a riser.
- :func:`term_trajectories` — per-year relative frequency (per 10k tokens, so
  years with more debate don't dominate) for chosen terms, for line charts.
- :func:`party_divergence_by_year` — mean pairwise Jensen-Shannon divergence
  between parties' yearly vocabularies: are the parties' languages converging or
  diverging over time?  One neutral lens, not a headline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .distinctiveness import (
    _DISTINCT_STOPWORDS,
    POLITICIAN_NAME_STOPWORDS,  # noqa: F401 — re-exported, used by nlp/__init__ and tests
    distinctive_words,
    group_token_counts,
    tokenize,
    weighted_log_odds,
)


def add_year(
    df: pd.DataFrame, date_col: str = "protocol_date", year_col: str = "year"
) -> pd.DataFrame:
    """Return a copy of *df* with an integer *year_col* derived from *date_col*.

    Rows whose date cannot be parsed are dropped.
    """
    out = df.copy()
    years = pd.to_datetime(out[date_col], errors="coerce").dt.year
    out[year_col] = years
    return out[years.notna()].astype({year_col: int})


def top_movers(
    df: pd.DataFrame,
    split_year: int | None = None,
    text_col: str = "text",
    year_col: str = "year",
    top_n: int = 40,
    min_count: int = 10,
    stopwords: frozenset[str] = _DISTINCT_STOPWORDS,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Words that rose / fell most between the early and late halves of the window.

    Splits the corpus at *split_year* (default: the median speech year) into an
    "early" and a "late" era, then runs Fightin' Words on those two groups.  A
    word over-represented in the late era is a **riser**; one over-represented in
    the early era is a **faller**.  The method's Dirichlet prior shrinks rare
    words, so a term must be both distinctive *and* well-attested to rank.

    Args:
        df:          Speeches with a text column and an integer *year_col*.
        split_year:  Boundary year; the late era is ``year >= split_year``.
            Defaults to the median year in *df*.
        text_col:    Column holding the speech text.
        year_col:    Integer year column (see :func:`add_year`).
        top_n:       How many risers and fallers to return.
        min_count:   Ignore words occurring fewer than this many times overall.
        stopwords:   Words to drop before counting.

    Returns:
        ``(risers, fallers)``; each a list of ``(word, z_score)`` descending by
        magnitude of the shift.
    """
    if split_year is None:
        split_year = int(df[year_col].median())

    tagged = df.copy()
    tagged["era"] = np.where(tagged[year_col] >= split_year, "late", "early")
    counts = group_token_counts(
        tagged, group_col="era", text_col=text_col, stopwords=stopwords
    )
    scores = weighted_log_odds(counts, min_count=min_count)
    risers = distinctive_words(scores, "late", top_n=top_n)
    fallers = distinctive_words(scores, "early", top_n=top_n)
    return risers, fallers


def term_trajectories(
    df: pd.DataFrame,
    terms: list[str],
    text_col: str = "text",
    year_col: str = "year",
    per: float = 10_000.0,
    stopwords: frozenset[str] = frozenset(),
) -> dict[int, dict[str, float]]:
    """Per-year relative frequency (per *per* tokens) for each term in *terms*.

    Relative rather than raw frequency, so years with more total debate do not
    dominate.  Terms are matched as lower-cased whole tokens.

    Args:
        df:        Speeches with a text column and an integer *year_col*.
        terms:     Terms to trace (lower-cased before matching).
        text_col:  Column holding the speech text.
        year_col:  Integer year column.
        per:       Scale for the rate (default per 10 000 tokens).
        stopwords: Words to drop before counting (empty by default so any chosen
            term is countable, even a common one).

    Returns:
        ``{year: {term: rate}}`` sorted by year.
    """
    wanted = [t.lower() for t in terms]
    counts = group_token_counts(
        df, group_col=year_col, text_col=text_col, min_length=1, stopwords=stopwords
    )
    out: dict[int, dict[str, float]] = {}
    for year in sorted(counts, key=int):
        counter = counts[year]
        total = sum(counter.values()) or 1
        out[int(year)] = {t: counter.get(t, 0) / total * per for t in wanted}
    return out


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two probability vectors."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _year_party_divergence(
    party_counts: dict[str, Counter[str]], min_count: int
) -> float | None:
    """Mean pairwise JS divergence across parties for one year.

    Builds each party's probability distribution over the shared vocabulary
    (words with at least *min_count* total occurrences that year) and averages
    the Jensen-Shannon divergence over every party pair.  ``None`` if fewer than
    two parties have data or the vocabulary is empty.
    """
    background: Counter[str] = Counter()
    for counter in party_counts.values():
        background.update(counter)
    vocab = [w for w, n in background.items() if n >= min_count]
    parties = [p for p, c in party_counts.items() if sum(c.values()) > 0]
    if len(parties) < 2 or not vocab:
        return None

    dists: dict[str, np.ndarray] = {}
    for party in parties:
        counter = party_counts[party]
        vec = np.array([counter.get(w, 0) for w in vocab], dtype=float)
        s = vec.sum()
        if s == 0:
            continue
        dists[party] = vec / s

    present = list(dists)
    if len(present) < 2:
        return None
    divs = [
        _js_divergence(dists[a], dists[b])
        for i, a in enumerate(present)
        for b in present[i + 1 :]
    ]
    return float(np.mean(divs)) if divs else None


def party_divergence_by_year(
    df: pd.DataFrame,
    text_col: str = "text",
    year_col: str = "year",
    party_col: str = "party",
    min_count: int = 5,
    stopwords: frozenset[str] = _DISTINCT_STOPWORDS,
) -> dict[int, float]:
    """Mean pairwise JS divergence between parties' vocabularies, per year.

    A rising series means the parties are talking *differently* over time
    (diverging); a falling series means their vocabularies are growing more
    alike (converging).

    Args:
        df:        Speeches with text, an integer *year_col* and a party column.
        text_col:  Column holding the speech text.
        year_col:  Integer year column.
        party_col: Column identifying the party.
        min_count: Minimum yearly occurrences for a word to enter that year's
            shared vocabulary.
        stopwords: Words to drop before counting.

    Returns:
        ``{year: mean_pairwise_js}`` sorted by year (years with <2 parties or no
        vocabulary are omitted).
    """
    result: dict[int, float] = {}
    for year, sub in df.groupby(year_col, observed=True):
        party_counts = group_token_counts(
            sub, group_col=party_col, text_col=text_col, stopwords=stopwords
        )
        div = _year_party_divergence(party_counts, min_count=min_count)
        if div is not None:
            result[int(year)] = div
    return dict(sorted(result.items()))


def yearly_signatures(
    df: pd.DataFrame,
    text_col: str = "text",
    year_col: str = "year",
    top_n: int = 15,
    min_count: int = 15,
    stopwords: frozenset[str] = _DISTINCT_STOPWORDS,
) -> dict[int, list[tuple[str, float]]]:
    """Each year's most distinctive words, scored against every other year.

    Unlike :func:`top_movers` (a single early/late split), this runs Fightin'
    Words with **every individual year** as its own group against the pooled
    rest, so each year gets its own "signature": the words that most set that
    year apart from all the others. Built for a year-by-year scrubber UI where
    dragging to a year should surface what made it distinct.

    Args:
        df:        Speeches with a text column and an integer *year_col*.
        text_col:  Column holding the speech text.
        year_col:  Integer year column (see :func:`add_year`).
        top_n:     How many words to keep per year.
        min_count: Ignore words occurring fewer than this many times overall.
        stopwords: Words to drop before counting (defaults to the general
            stopword list plus curated politician names).

    Returns:
        ``{year: [(word, z_score), ...]}`` sorted by year, each year's list
        descending by distinctiveness.
    """
    counts = group_token_counts(df, group_col=year_col, text_col=text_col, stopwords=stopwords)
    scores = weighted_log_odds(counts, min_count=min_count)
    return {
        int(year): distinctive_words(scores, year, top_n=top_n) for year in sorted(counts, key=int)
    }


#: Issue frames as sets of Swedish word stems, matched as substrings of tokens
#: (so declensions/compounds match: "gäng" also hits "gängkriminalitet").
#: Curated product content, not a statistical artefact — extend as needed.
ISSUE_FRAMES: dict[str, list[str]] = {
    "Brott & straff": [
        "gäng", "kriminell", "skjutning", "spräng", "straff", "brottsling",
        "brottslig", "otrygg", "utvisning", "fängelse", "poliser",
    ],
    "Migration": [
        "invandr", "migration", "asyl", "flykting", "nyanländ", "integration",
        "uppehållstillstånd", "medborgarskap", "anhörig", "utlänning",
    ],
    "Klimat": [
        "klimat", "utsläpp", "fossil", "förnybar", "omställning", "koldioxid",
        "vindkraft",
    ],
    "Välfärd": [
        "välfärd", "sjukvård", "äldreomsorg", "förskola", "vårdkö",
        "underskötersk", "trygghetssystem",
    ],
    "Ekonomi": [
        "inflation", "ränta", "skattesänk", "tillväxt", "statsfinans", "hushållens",
    ],
    "Skola & utbildning": [
        "skola", "skolan", "elev", "lärare", "grundskol", "gymnasi", "läromedel",
        "betyg", "utbildning",
    ],
    "Försvar & säkerhet": [
        "försvar", "totalförsvar", "nato", "militär", "beredskap", "säkerhetspolitik",
        "underrättelse", "värnplikt",
    ],
    "Energi": [
        "kärnkraft", "elpris", "elnät", "elförsörjning", "energipolitik", "kärnkraften",
        "kraftverk", "elproduktion",
    ],
    "Bostad": [
        "bostad", "bostäder", "hyresrätt", "bostadsbrist", "bostadsmarknad",
        "amortering", "boende",
    ],
    "Jämställdhet": [
        "jämställd", "kvinnor", "mäns våld", "diskriminering", "hbtq", "föräldraledig",
    ],
}


def frame_trajectories(
    df: pd.DataFrame,
    frames: dict[str, list[str]] = ISSUE_FRAMES,
    text_col: str = "text",
    year_col: str = "year",
    group_col: str = "party",
    per: float = 10_000.0,
) -> dict[str, dict[str, dict[int, float]]]:
    """Per-year relative frequency of each issue frame, broken out by group.

    A *frame* is a curated set of word stems (see :data:`ISSUE_FRAMES`), e.g.
    "gäng", "kriminell", ... for "Brott & straff". A token counts toward the
    frame if any stem is a substring of it. This is how "who talks about this
    issue, and has that changed" becomes visible: plot every party's line for
    one frame and watch them converge or diverge — e.g. whether a party's
    crime-frame rate has climbed toward another's over time.

    Args:
        df:        Speeches with text, an integer *year_col* and *group_col*.
        frames:    Mapping of frame name -> list of stems.
        text_col:  Column holding the speech text.
        year_col:  Integer year column.
        group_col: Column to break out by (default: party).
        per:       Scale for the rate (default per 10 000 tokens).

    Returns:
        ``{frame: {group: {year: rate}}}``.
    """
    out: dict[str, dict[str, dict[int, float]]] = {f: {} for f in frames}
    for (year, group), sub in df.groupby([year_col, group_col], observed=True):
        # Count once per (year, group), then match stems against the *unique*
        # vocabulary rather than every token instance — a corpus this size has
        # heavy repetition, so this is an order of magnitude fewer substring
        # checks than scanning the full token stream per frame.
        counter: Counter[str] = Counter()
        for text in sub[text_col].dropna():
            counter.update(tokenize(text, min_length=1, stopwords=frozenset()))
        total = sum(counter.values()) or 1
        for frame, stems in frames.items():
            hits = sum(n for word, n in counter.items() if any(s in word for s in stems))
            out[frame].setdefault(str(group), {})[int(year)] = round(hits / total * per, 2)
    return out
