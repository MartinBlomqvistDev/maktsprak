"""Text cleaning and Swedish stop-word management.

The combined stop-word set (:data:`combined_stopwords`) is built once at
module import time and reused across all callers.  It merges a file-based
general Swedish list with a hand-curated political vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Load file-based Swedish stop words
# ---------------------------------------------------------------------------
_STOPWORDS_FILE: Path = Path(__file__).parents[3] / "data" / "processed" / "stopwords-sv.txt"

try:
    with _STOPWORDS_FILE.open(encoding="utf-8") as _fh:
        swedish_stopwords: frozenset[str] = frozenset(
            line.strip().lower() for line in _fh if line.strip()
        )
except FileNotFoundError:
    import warnings
    warnings.warn(
        f"Swedish stop-word file not found: {_STOPWORDS_FILE}. "
        "Using an empty set — word cloud quality will be reduced.",
        RuntimeWarning,
        stacklevel=2,
    )
    swedish_stopwords = frozenset()

# ---------------------------------------------------------------------------
# Curated political stop words
# ---------------------------------------------------------------------------
political_stopwords: frozenset[str] = frozenset(
    {
        # Titles and roles
        "herr", "fru", "talman", "statsrådet", "ministern", "finansministern",
        "ledamot", "ledamoten", "partiledare", "tjänstgörande", "ersättare",
        "vice", "andre", "tredje", "förste",
        # Debate-specific terms
        "anförande", "replik", "interpellation", "interpellationer",
        "fråga", "frågan", "frågor", "svar", "svaret", "debatt", "debatten",
        "kammaren", "protokoll", "prot", "utskottet", "betänkande", "ärende",
        "yrkande", "bifall", "avslag", "tackar", "mfl", "res",
        # General political terms
        "regeringen", "regeringens", "riksdagen", "riksdagens",
        "partiet", "partierna", "partiernas",
        "sverige", "landet", "svensk", "svenska", "land", "nation",
        "medborgare", "medborgarna", "samhället", "frågeställning",
        "dessa", "andra",
        "politik", "politiken", "förslag", "budgeten",
        "miljarder", "kronor", "procent", "sveriges",
        # Common verbs and nouns in parliamentary debates
        "tack", "gäller", "finns", "handlar", "betyder", "innebär",
        "anser", "tycker", "menar", "tror", "göra", "säga", "se", "vet",
        "behöver", "borde", "självklart", "därför", "också", "samtidigt",
        "väldigt", "helt", "bara", "kanske", "ytterligare",
        "tid", "gång", "nya", "stora", "olika", "viktigt", "får",
        # Basic Swedish function words (extra safety net)
        "vi", "att", "för", "på", "och", "men", "eller", "nu", "som",
        "med", "är", "den", "det", "ett", "en", "av", "om", "till",
        "har", "vår", "vårt", "våra", "de", "dem", "dig", "oss",
        "måste", "skall", "ska", "barn", "både", "människor", "ändå",
        "bör", "åtgärder", "stöd", "uppdrag", "staten", "personer",
        "person", "talma", "fortsätt", "fortsätta", "mar",
        # Month names (avoid date-related noise in word clouds)
        "januari", "februari", "mars", "april", "maj", "juni",
        "juli", "augusti", "september", "oktober", "november", "december",
    }
)

#: Union of the file-based and political stop-word sets.
combined_stopwords: frozenset[str] = swedish_stopwords | political_stopwords


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise a raw speech string for downstream model consumption.

    Joins hyphenated line-breaks (``demo-\\ncrat`` → ``democrat``), collapses
    all whitespace sequences to a single space, and strips leading/trailing
    whitespace.

    Args:
        text: Raw speech text, potentially containing PDF extraction artefacts.

    Returns:
        Cleaned string, or an empty string if *text* is not a ``str``.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"-\n", "", text)          # heal hyphenated line-breaks
    text = re.sub(r"\s+", " ", text)         # collapse whitespace
    return text.strip()
