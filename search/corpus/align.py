"""Aligning the older corpus's speeches with the new corpus's, one protocol at a time.

The older corpus was parsed from the protocol PDFs. Its record is one
speaker's contributions to one protocol, joined with a blank line, and it never
captured the speech numbers. The only link from one of its speeches to a speech
in the new corpus is therefore the protocol, the party, the speaker and the
order. This module turns that link into pairs and scores each pair by its
words. Nothing here reads or writes a file.

Texts are compared as tokens. The older text has its whitespace collapsed and
its line-end hyphens removed without regard to the word, so a comparison that
kept hyphens or line breaks would count the parser's habits as differences in
what was said. Whether two texts are the same speech is decided on pairs of
adjacent tokens, which two different speeches by one speaker rarely share. How
alike two texts are is reported on single tokens, counted as a multiset.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

#: Letters the older corpus folds to a Latin base before it drops what is not
#: ASCII. A copy of ``_SLUG_FOLD`` in ``src/maktsprak_pipeline/pipeline/transform.py``,
#: which cannot be imported without credentials.
_SLUG_FOLD = str.maketrans({"å": "a", "ä": "a", "ö": "o", "é": "e", "è": "e", "ü": "u", "ø": "o"})

#: Hyphen and dash characters, which the tokeniser removes wherever they stand:
#: the hyphen-minus, the soft hyphen, the hyphen, the non-breaking hyphen, the
#: figure dash, the two typographic dashes and the minus sign. The protocol PDFs
#: set a range such as 2002-2006 with a typographic dash where the new corpus
#: has a hyphen-minus, so the two must be treated alike.
DASHES = frozenset("-\u00ad\u2010\u2011\u2012\u2013\u2014\u2212")

#: How a pair was found.
IN_ORDER = "in_order"
BY_CONTENT = "by_content"
ACROSS_SLUGS = "across_slugs"


class AlignParams(BaseModel):
    """The parameters of the alignment, written into the report with its results.

    The values come from the ``reconcile`` section of ``search/config.yaml``;
    this model has no defaults of its own, and the functions that need a
    parameter take the whole model as an argument.

    Attributes:
        edge_tokens: How many tokens at the start and at the end of two texts
            are compared for agreement.
        min_sequence_share: The least share of the new speech's pairs of
            adjacent tokens that must be found in the older speech for the two
            to count as the same speech.
        min_shared_tokens: The least number of shared tokens for a pair, so that
            two very short texts cannot pair on a greeting alone.
        missing_tokens: The number of the new speech's tokens absent from the
            older speech above which the report counts the pair as missing words.
        surplus_tokens: The number of surplus tokens in an older speech above
            which the report counts the pair as carrying a surplus.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_tokens: int
    min_sequence_share: float
    min_shared_tokens: int
    missing_tokens: int
    surplus_tokens: int


class _TokenTable(dict[int, str]):
    """A translation table for :meth:`str.translate` that decides each character once.

    A letter or a digit is kept, a dash is removed, and anything else becomes
    a space. The decision is cached, so translating the whole corpus costs one
    lookup per character after the first sight of it.
    """

    def __missing__(self, code: int) -> str:
        char = chr(code)
        if char in DASHES:
            value = ""
        elif char.isalnum():
            value = char
        else:
            value = " "
        self[code] = value
        return value


_TOKEN_TABLE = _TokenTable()


def tokenise(text: str) -> tuple[str, ...]:
    """Split a text into lower-case tokens of letters and digits.

    Hyphens and dashes are removed before the split. A hyphenated word
    therefore becomes one token: "EU-länder" and the older corpus's
    "EUländer" both give ``euländer``. A range of years becomes one token
    however it is set, and a free-standing dash leaves nothing behind. Every
    other character that is neither a letter nor a digit separates tokens.

    Args:
        text: The text to split.

    Returns:
        The tokens, in the order they stand in the text.
    """
    return tuple(text.lower().translate(_TOKEN_TABLE).split())


def speaker_slug(name: str) -> str:
    """Fold a speaker's name into the slug the older corpus keys its records on.

    The same steps as ``_speaker_slug`` in the older pipeline: lower-case, fold
    the Swedish letters, drop what is not ASCII, and turn every run of other
    characters into one hyphen, trimmed at both ends.

    Args:
        name: The name with any title, for example ``Statsminister ULF KRISTERSSON``.

    Returns:
        The slug, for example ``statsminister-ulf-kristersson``.
    """
    folded = name.strip().lower().translate(_SLUG_FOLD)
    ascii_only = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode()
    pieces: list[str] = []
    for char in ascii_only:
        if char.isascii() and char.isalnum() and not char.isupper():
            pieces.append(char)
        elif not pieces or pieces[-1] != "-":
            pieces.append("-")
    return "".join(pieces).strip("-")


#: The marker the new corpus puts after the party of a reply, in the forms it
#: takes there: ``NAME (M) replik`` in the sessions 2003/04 to 2013/14,
#: ``NAME (M)replik`` once in 2011/12 and ``NAME (M) (replik)`` three times in
#: 2014/15. No slug of the older corpus ends in it. The form in parentheses is
#: tried first.
REPLY_MARKERS = ("(replik)", "replik")


def strip_reply_marker(speaker_raw: str) -> str:
    """Remove the reply marker that may follow the party in a new corpus speaker."""
    speaker = speaker_raw.strip()
    lowered = speaker.lower()
    for marker in REPLY_MARKERS:
        if lowered.endswith(marker):
            return speaker[: -len(marker)].rstrip()
    return speaker


def strip_party(speaker_raw: str) -> str:
    """Remove the party in parentheses that ends a speaker as the new corpus has it.

    A reply marker after the party is removed first, so that ``NAME (M) replik``
    gives ``NAME`` as ``NAME (M)`` does.
    """
    speaker = strip_reply_marker(speaker_raw)
    if speaker.endswith(")") and " (" in speaker:
        return speaker.rsplit(" (", 1)[0]
    return speaker


def new_speaker_slug(speaker_raw: str) -> str:
    """Turn a ``speaker_raw`` from the new corpus into the older corpus's slug."""
    return speaker_slug(strip_party(speaker_raw))


#: The least likeness, by :class:`difflib.SequenceMatcher`, at which an older
#: slug and a new slug agree on a name whose tokens differ. The slugs are
#: compared with their hyphens removed. The older slug
#: ``arbetsmarknadsminister-elisabeth-svantessson`` has a letter too many and
#: is 0.988 like the new slug of the same title and name.
SLUG_LIKENESS = 0.95


def names_agree(old_slug: str, new_slug: str) -> bool:
    """Say whether an older slug and a new slug agree on the speaker's name.

    The older slugs come from names read out of the protocol PDFs. A line
    break there can split a word with a hyphen, as in ``storckenf-eldt``, and
    a letter can be misprinted, as in ``svantessson``. A title can stand on
    one side only or in two forms, and either slug can end in the member's
    place, as in ``ulrika-carlsson-i-skovde``.

    Identical slugs agree. Otherwise a place at the end of either slug is set
    apart first, by :func:`split_place`, and what is left of each slug is its
    name tokens, any title included. The likeness of two slugs is the
    likeness of their name tokens joined without hyphens.

    Two slugs that both carry a place agree only when the places are the
    same. When one slug carries a place and the other does not, they agree
    only when they are at least :data:`SLUG_LIKENESS` alike: the token test
    below, which lets a title stand on one side, is not applied to them.
    Otherwise, when the new slug has at
    least two name tokens, its last is taken as the surname and the one
    before it as the given name, and the slugs agree when each of the two is
    found among the older slug's name tokens, as one token or as a run of
    adjacent tokens joined together. They also agree when they are at least
    :data:`SLUG_LIKENESS` alike.

    The test compares names and places. Two members of one party who have the
    same name agree with each other's slugs where neither slug gives a place,
    and where one slug gives it and the other holds the name alone.

    Args:
        old_slug: The slug of the older speech's speaker.
        new_slug: The slug of the new speech's speaker.

    Returns:
        Whether the two slugs agree on the name.
    """
    if old_slug == new_slug:
        return True
    old_names, old_place = split_place(old_slug)
    new_names, new_place = split_place(new_slug)
    if old_place and new_place and old_place != new_place:
        return False
    likeness = SequenceMatcher(None, "".join(old_names), "".join(new_names), autojunk=False)
    if bool(old_place) != bool(new_place):
        return likeness.ratio() >= SLUG_LIKENESS
    if len(new_names) >= 2 and all(_is_token_run(name, old_names) for name in new_names[-2:]):
        return True
    return likeness.ratio() >= SLUG_LIKENESS


#: The token that sets a member's place after the name in a slug, as in
#: ``ulrika-carlsson-i-skovde``.
PLACE_MARKER = "i"

#: The most tokens a place has, as in ``i-hisings-backa``. In both corpora a
#: longer run after the marker occurs only in older slugs that hold words of
#: the protocol before the speaker's name, and the run then ends in the name.
PLACE_TOKENS = 2


def split_place(slug: str) -> tuple[list[str], tuple[str, ...]]:
    """Split a slug into the tokens of the name and the place that ends it.

    The place is the last one or :data:`PLACE_TOKENS` tokens of the slug,
    when the token before them is :data:`PLACE_MARKER` and at least one token
    stands before the marker. The shorter place is tried first.

    Args:
        slug: A speaker's slug.

    Returns:
        The tokens before the marker and the tokens of the place. A slug
        without a place gives all of its tokens and an empty place.
    """
    tokens = [token for token in slug.split("-") if token]
    for size in range(1, PLACE_TOKENS + 1):
        index = len(tokens) - size - 1
        if index >= 1 and tokens[index] == PLACE_MARKER:
            return tokens[:index], tuple(tokens[index + 1 :])
    return tokens, ()


def _is_token_run(word: str, tokens: Sequence[str]) -> bool:
    """Say whether a word equals one token, or a run of adjacent tokens joined together."""
    for start in range(len(tokens)):
        joined = ""
        for token in tokens[start:]:
            joined += token
            if joined == word:
                return True
            if len(joined) >= len(word):
                break
    return False


class Speech(NamedTuple):
    """A speech reduced to what the alignment compares.

    Attributes:
        key: The speech's identifier on its own side.
        slug: The speaker's slug, see :func:`speaker_slug`.
        tokens: The tokens of the speech; for the new corpus, of the speaker's
            paragraphs only.
        note_tokens: The tokens of the notes the new corpus set aside. Empty
            for the older corpus, which has none.
    """

    key: str
    slug: str
    tokens: tuple[str, ...]
    note_tokens: tuple[str, ...] = ()


class PairScore(NamedTuple):
    """How alike a new speech and an older speech are.

    Attributes:
        new_tokens: Tokens in the new speech's paragraphs.
        old_tokens: Tokens in the older speech.
        note_tokens: Tokens in the new speech's notes.
        shared: Tokens the two have in common, counted as a multiset.
        shared_with_notes: The same, with the notes added to the new speech.
        new_bigrams: Pairs of adjacent tokens in the new speech's paragraphs.
        shared_bigrams: Pairs of adjacent tokens the two have in common,
            counted as a multiset.
        head_agrees: Whether the older speech starts with the new speech's
            first ``edge_tokens`` tokens, or with the whole new speech when it
            is shorter than that. False when the new speech has no tokens.
        tail_agrees: Whether the older speech ends with the new speech's last
            tokens, counted the same way.
        identical: Whether the two token sequences are the same.
        old_extends_new: Whether the older speech starts with the whole new
            speech and goes on after it.
    """

    new_tokens: int
    old_tokens: int
    note_tokens: int
    shared: int
    shared_with_notes: int
    new_bigrams: int
    shared_bigrams: int
    head_agrees: bool
    tail_agrees: bool
    identical: bool
    old_extends_new: bool

    @property
    def new_in_old(self) -> float:
        """The share of the new speech's tokens found in the older speech."""
        return self.shared / self.new_tokens if self.new_tokens else 0.0

    @property
    def sequence_share(self) -> float:
        """The share of the new speech's pairs of adjacent tokens found in the older speech."""
        return self.shared_bigrams / self.new_bigrams if self.new_bigrams else 0.0

    @property
    def old_in_new(self) -> float:
        """The share of the older speech's tokens found in the new speech's paragraphs."""
        return self.shared / self.old_tokens if self.old_tokens else 0.0

    @property
    def old_in_new_with_notes(self) -> float:
        """The share of the older speech's tokens found in the paragraphs and notes."""
        return self.shared_with_notes / self.old_tokens if self.old_tokens else 0.0

    @property
    def surplus(self) -> int:
        """Tokens of the older speech that the new speech's paragraphs do not have."""
        return self.old_tokens - self.shared

    @property
    def surplus_with_notes(self) -> int:
        """Tokens of the older speech that neither the paragraphs nor the notes have."""
        return self.old_tokens - self.shared_with_notes

    @property
    def missing(self) -> int:
        """Tokens of the new speech that the older speech does not have."""
        return self.new_tokens - self.shared


def edges_agree(
    new_tokens: Sequence[str], old_tokens: Sequence[str], edge: int
) -> tuple[bool, bool]:
    """Say whether two token sequences agree at their start and at their end.

    Each end is compared over ``edge`` tokens, or over the whole new speech
    when it is shorter, so that a short new speech that opens the older one
    agrees at the head. An empty new speech agrees at neither end.

    Args:
        new_tokens: The new speech's tokens.
        old_tokens: The older speech's tokens.
        edge: How many tokens to compare at each end.

    Returns:
        Whether the heads agree, and whether the tails agree.
    """
    size = min(edge, len(new_tokens))
    if not size:
        return False, False
    head = tuple(new_tokens[:size]) == tuple(old_tokens[:size])
    tail = tuple(new_tokens[len(new_tokens) - size :]) == tuple(
        old_tokens[max(len(old_tokens) - size, 0) :]
    )
    return head, tail


def overlap(first: Mapping[str, int], second: Mapping[str, int]) -> int:
    """Count the tokens two bags have in common, each token by the smaller of its two counts."""
    if len(first) > len(second):
        first, second = second, first
    return sum(min(count, second.get(token, 0)) for token, count in first.items())


class _Bags:
    """Token counts per speech, computed once for each speech the alignment touches.

    The cache is keyed on the identity of each speech object and holds a
    reference to the object, so that identity cannot be reused while the cache
    lives. Two speeches with the same ``key`` on the two sides therefore never
    share counts.
    """

    def __init__(self) -> None:
        self._bags: dict[tuple[int, str], tuple[Speech, Counter[str]]] = {}
        self._bigrams: dict[int, tuple[Speech, Counter[tuple[str, str]]]] = {}

    def of(self, speech: Speech, with_notes: bool = False) -> Counter[str]:
        """Return the bag of a speech's tokens, with its notes added when asked."""
        key = (id(speech), "notes" if with_notes else "text")
        cached = self._bags.get(key)
        if cached is None:
            bag = Counter(speech.tokens)
            if with_notes:
                bag.update(speech.note_tokens)
            cached = self._bags[key] = (speech, bag)
        return cached[1]

    def bigrams_of(self, speech: Speech) -> Counter[tuple[str, str]]:
        """Return the bag of a speech's pairs of adjacent tokens, notes left out."""
        cached = self._bigrams.get(id(speech))
        if cached is None:
            bag = Counter(zip(speech.tokens, speech.tokens[1:], strict=False))
            cached = self._bigrams[id(speech)] = (speech, bag)
        return cached[1]


def score_pair(
    new: Speech, old: Speech, params: AlignParams, bags: _Bags | None = None
) -> PairScore:
    """Score one new speech against one older speech.

    Args:
        new: The speech from the new corpus.
        old: The speech from the older corpus.
        params: The alignment's parameters; ``edge_tokens`` is used here.
        bags: A cache of token counts, so that a speech compared many times is
            counted once.

    Returns:
        The overlap in both directions and the agreement at both ends.
    """
    bags = bags or _Bags()
    old_bag = bags.of(old)
    shared = overlap(bags.of(new), old_bag)
    shared_with_notes = (
        overlap(bags.of(new, with_notes=True), old_bag) if new.note_tokens else shared
    )
    head_agrees, tail_agrees = edges_agree(new.tokens, old.tokens, params.edge_tokens)
    return PairScore(
        new_tokens=len(new.tokens),
        old_tokens=len(old.tokens),
        note_tokens=len(new.note_tokens),
        shared=shared,
        shared_with_notes=shared_with_notes,
        new_bigrams=max(len(new.tokens) - 1, 0),
        shared_bigrams=overlap(bags.bigrams_of(new), bags.bigrams_of(old)),
        head_agrees=head_agrees,
        tail_agrees=tail_agrees,
        identical=new.tokens == old.tokens,
        old_extends_new=(
            len(old.tokens) > len(new.tokens) and old.tokens[: len(new.tokens)] == new.tokens
        ),
    )


def is_same_speech(score: PairScore, params: AlignParams) -> bool:
    """Decide from a score whether two texts are the same speech.

    Two conditions must hold. The texts share at least ``min_shared_tokens``
    tokens, so that two very short texts cannot pair on a greeting alone. And
    at least ``min_sequence_share`` of the new speech's pairs of adjacent
    tokens occur in the older speech. The share is taken of the new speech
    because the older one may carry the record's own text after the speech
    and so be much longer. Pairs of adjacent tokens are counted because two
    different speeches by one speaker share many single words but few pairs.

    The test runs in one direction: it asks whether most of the new speech is
    found in the older text. A new speech that has lost words can therefore
    pass, and the coverage built on the test does not show whether the new
    speeches are complete. The other direction is described by
    :attr:`PairScore.old_in_new` and by the surplus figures in the report of
    ``03_reconcile_legacy.py``. Where both corpora carry the same words, the
    comparison cannot tell the record's text from the speaker's. That is why
    the paragraphs set aside as the record's own are checked by reading them
    when the corpus is built.

    Args:
        score: The score of the two texts, from :func:`score_pair`.
        params: The alignment's parameters.

    Returns:
        Whether the two texts count as the same speech.
    """
    return (
        score.shared >= params.min_shared_tokens
        and score.sequence_share >= params.min_sequence_share
    )


class Match(NamedTuple):
    """A new speech and an older speech found to be the same.

    Attributes:
        old_key: The older speech's key.
        new_key: The new speech's key.
        method: How the pair was found. ``in_order``: the slug has as many
            speeches on both sides and they pair one by one. ``by_content``:
            the two stand under one slug and were paired by
            :func:`monotone_alignment`. ``across_slugs``: the same, under two
            slugs that agree on the speaker's name.
        score: How alike the two are.
    """

    old_key: str
    new_key: str
    method: str
    score: PairScore


class Alignment(NamedTuple):
    """The pairs found in one protocol and party, and what was left on either side.

    Attributes:
        matches: The pairs, in the order of the older corpus's speeches.
        unmatched_old: Keys of the older speeches without a pair.
        unmatched_new: Keys of the new speeches without a pair, in their order.
    """

    matches: tuple[Match, ...]
    unmatched_old: tuple[str, ...]
    unmatched_new: tuple[str, ...]


def monotone_alignment(
    old: Sequence[Speech], new: Sequence[Speech], params: AlignParams, bags: _Bags
) -> list[tuple[int, int, PairScore]]:
    """Pair two sequences of speeches by content without changing either order.

    Among all ways to pair the speeches so that the order is kept on both sides
    and every pair passes :func:`is_same_speech`, this picks the one with the
    most shared pairs of adjacent tokens in total. Speeches left over stay unpaired.

    Args:
        old: The older speeches, in their order.
        new: The new speeches, in their order.
        params: The alignment's parameters.
        bags: The cache of token counts.

    Returns:
        The pairs as positions in ``old`` and ``new`` with their scores, in order.
    """
    rows, cols = len(old), len(new)
    scores: dict[tuple[int, int], PairScore] = {}
    for i, old_speech in enumerate(old):
        for j, new_speech in enumerate(new):
            score = score_pair(new_speech, old_speech, params, bags)
            if is_same_speech(score, params):
                scores[i, j] = score
    if not scores:
        return []

    best = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            value = max(best[i - 1][j], best[i][j - 1])
            score = scores.get((i - 1, j - 1))
            if score is not None:
                value = max(value, best[i - 1][j - 1] + score.shared_bigrams)
            best[i][j] = value

    pairs: list[tuple[int, int, PairScore]] = []
    i, j = rows, cols
    while i > 0 and j > 0:
        score = scores.get((i - 1, j - 1))
        if score is not None and best[i][j] == best[i - 1][j - 1] + score.shared_bigrams:
            pairs.append((i - 1, j - 1, score))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def align_speaker(
    old: Sequence[Speech], new: Sequence[Speech], params: AlignParams, bags: _Bags | None = None
) -> tuple[list[Match], list[int], list[int]]:
    """Pair one speaker's older speeches with the same speaker's new ones.

    When the counts are equal the speeches are paired in order, and the pairing
    stands if every pair passes :func:`is_same_speech`. Otherwise the pairs are
    found by :func:`monotone_alignment`.

    Args:
        old: The speaker's older speeches, in order.
        new: The speaker's new speeches, in order.
        params: The alignment's parameters.
        bags: A cache of token counts.

    Returns:
        The pairs, and the positions left unpaired in ``old`` and in ``new``.
    """
    bags = bags or _Bags()
    if len(old) == len(new):
        scores = [score_pair(n, o, params, bags) for o, n in zip(old, new, strict=True)]
        if all(is_same_speech(score, params) for score in scores):
            matches = [
                Match(o.key, n.key, IN_ORDER, score)
                for o, n, score in zip(old, new, scores, strict=True)
            ]
            return matches, [], []
    pairs = monotone_alignment(old, new, params, bags)
    matches = [Match(old[i].key, new[j].key, BY_CONTENT, score) for i, j, score in pairs]
    paired_old = {i for i, _, _ in pairs}
    paired_new = {j for _, j, _ in pairs}
    return (
        matches,
        [i for i in range(len(old)) if i not in paired_old],
        [j for j in range(len(new)) if j not in paired_new],
    )


def _by_slug(speeches: Iterable[Speech]) -> dict[str, list[Speech]]:
    """Group speeches by speaker, keeping their order within each speaker."""
    groups: dict[str, list[Speech]] = {}
    for speech in speeches:
        groups.setdefault(speech.slug, []).append(speech)
    return groups


def align_party(old: Sequence[Speech], new: Sequence[Speech], params: AlignParams) -> Alignment:
    """Align the older and the new speeches of one party in one protocol.

    The work is done in two steps. First each speaker's older speeches are
    paired with the new speeches under the same slug, by :func:`align_speaker`.
    Then the older speeches left unpaired are aligned, slug by slug in sorted
    order, against the unpaired new speeches whose slug agrees on the
    speaker's name by :func:`names_agree`. The second step finds a speaker
    whose name the two sources write differently: a minister's title in
    another form, a place after the name, a surname split by a hyphen. A new
    speech whose slug :func:`names_agree` refuses stays unpaired, however
    alike the words are. That test compares the places the slugs end in, the
    given name and surname the new slug ends in, and the likeness of the two
    names, so two members of one party with the same name, one of them
    written without a place, can pair across their slugs.

    Args:
        old: The older speeches, in document order within each speaker.
        new: The new speeches, in document order.
        params: The alignment's parameters.

    Returns:
        The pairs and what was left over on either side.
    """
    bags = _Bags()
    old_groups = _by_slug(old)
    new_groups = _by_slug(new)
    matches: list[Match] = []
    left_old: dict[str, list[Speech]] = {}
    paired_new: set[str] = set()

    for slug in sorted(old_groups):
        speeches = old_groups[slug]
        if slug not in new_groups:
            left_old[slug] = speeches
            continue
        found, rest, _ = align_speaker(speeches, new_groups[slug], params, bags)
        matches.extend(found)
        paired_new.update(match.new_key for match in found)
        if rest:
            left_old[slug] = [speeches[i] for i in rest]

    for slug in sorted(left_old):
        pool = [
            speech
            for speech in new
            if speech.key not in paired_new and names_agree(slug, speech.slug)
        ]
        for i, j, score in monotone_alignment(left_old[slug], pool, params, bags):
            method = BY_CONTENT if pool[j].slug == slug else ACROSS_SLUGS
            matches.append(Match(left_old[slug][i].key, pool[j].key, method, score))
            paired_new.add(pool[j].key)

    paired_old = {match.old_key for match in matches}
    order = {speech.key: position for position, speech in enumerate(old)}
    matches.sort(key=lambda match: order[match.old_key])
    return Alignment(
        matches=tuple(matches),
        unmatched_old=tuple(speech.key for speech in old if speech.key not in paired_old),
        unmatched_new=tuple(speech.key for speech in new if speech.key not in paired_new),
    )


#: Why a speech was left without a pair.
TOO_SHORT = "too_short"
PARTY_ABSENT = "party_absent"
NO_CONTENT_MATCH = "no_content_match"

#: Where the words of a speech left without a pair were found after all.
UNDER_ANOTHER_NAME = "under_another_name"
IN_ANOTHER_PARTY = "in_another_party"
IN_A_PAIRED_SPEECH = "in_a_paired_speech"


class Unmatched(NamedTuple):
    """A speech left without a pair, and what the protocol says about why.

    Attributes:
        key: The speech's key on its own side.
        party: The party it was filed under on its own side.
        reason: ``too_short`` when the speech has fewer tokens than a pair
            needs in common, so that it can never be paired; ``party_absent``
            when the other side has no speech of that party in the protocol;
            ``no_content_match`` otherwise.
        found_in: Where the same words were found on the other side despite
            that. ``under_another_name``: in an unpaired speech of the same
            party whose slug does not agree on the speaker's name.
            ``in_another_party``: in an unpaired speech of another party.
            ``in_a_paired_speech``: in a speech that is already paired.
            ``None``: nowhere. The places are tried in that order. This is a
            diagnosis and never makes a pair.
    """

    key: str
    party: str
    reason: str
    found_in: str | None


class ProtocolAlignment(NamedTuple):
    """The pairs found in one protocol, and every speech left over on either side.

    Attributes:
        matches: The pairs, party by party in alphabetical order.
        unmatched_old: The older speeches without a pair.
        unmatched_new: The new speeches without a pair.
    """

    matches: tuple[Match, ...]
    unmatched_old: tuple[Unmatched, ...]
    unmatched_new: tuple[Unmatched, ...]


def _found_in(
    speech: Speech,
    is_old: bool,
    candidates: Iterable[tuple[Speech, str]],
    params: AlignParams,
    bags: _Bags,
) -> str | None:
    """Look for a speech's words among the other side's speeches, for the diagnosis."""
    for other, where in candidates:
        new, old = (other, speech) if is_old else (speech, other)
        if is_same_speech(score_pair(new, old, params, bags), params):
            return where
    return None


def reconcile_protocol(
    old: Mapping[str, Sequence[Speech]],
    new: Mapping[str, Sequence[Speech]],
    params: AlignParams,
) -> ProtocolAlignment:
    """Align one protocol's older and new speeches, party by party.

    Pairs are only ever made within a party, by :func:`align_party`. A speech
    left over is then compared with the other side's unpaired speeches and
    with the other side's paired speeches. That tells a speech the other side
    does not have from three others: one that stands there under another
    speaker's name, one filed there under another party, and one whose words
    are part of a speech that is already paired.

    Args:
        old: The older speeches by party, in document order within each speaker.
        new: The new speeches by party, in document order.
        params: The alignment's parameters.

    Returns:
        The pairs and what was left over, with the reason for each.
    """
    matches: list[Match] = []
    left_old: list[tuple[str, Speech]] = []
    left_new: list[tuple[str, Speech]] = []
    for party in sorted(set(old) | set(new)):
        old_speeches = old.get(party, ())
        new_speeches = new.get(party, ())
        alignment = align_party(old_speeches, new_speeches, params)
        matches.extend(alignment.matches)
        rest_old = set(alignment.unmatched_old)
        rest_new = set(alignment.unmatched_new)
        left_old.extend((party, speech) for speech in old_speeches if speech.key in rest_old)
        left_new.extend((party, speech) for speech in new_speeches if speech.key in rest_new)

    paired_old_keys = {match.old_key for match in matches}
    paired_new_keys = {match.new_key for match in matches}
    paired_old = [s for speeches in old.values() for s in speeches if s.key in paired_old_keys]
    paired_new = [s for speeches in new.values() for s in speeches if s.key in paired_new_keys]
    bags = _Bags()

    def diagnose(
        party: str, speech: Speech, is_old: bool, others: list[tuple[str, Speech]]
    ) -> Unmatched:
        other_side = new if is_old else old
        if len(speech.tokens) < params.min_shared_tokens:
            reason = TOO_SHORT
        elif other_side.get(party):
            reason = NO_CONTENT_MATCH
        else:
            reason = PARTY_ABSENT

        def other_name(other: Speech) -> bool:
            old_slug, new_slug = (speech.slug, other.slug) if is_old else (other.slug, speech.slug)
            return not names_agree(old_slug, new_slug)

        candidates = [(s, UNDER_ANOTHER_NAME) for p, s in others if p == party and other_name(s)]
        candidates += [(s, IN_ANOTHER_PARTY) for p, s in others if p != party]
        candidates += [(s, IN_A_PAIRED_SPEECH) for s in (paired_new if is_old else paired_old)]
        return Unmatched(
            speech.key, party, reason, _found_in(speech, is_old, candidates, params, bags)
        )

    return ProtocolAlignment(
        matches=tuple(matches),
        unmatched_old=tuple(diagnose(p, s, True, left_new) for p, s in left_old),
        unmatched_new=tuple(diagnose(p, s, False, left_old) for p, s in left_new),
    )


#: The older corpus joins one speaker's speeches in a protocol with a blank line.
LEGACY_SEPARATOR = "\n\n"


def split_legacy_text(text: str) -> list[str]:
    """Split an older record's text into its speeches, in document order."""
    return [speech.strip() for speech in text.split(LEGACY_SEPARATOR) if speech.strip()]


def legacy_slug(record_id: str, protocol_id: str, party: str) -> str:
    """Read the speaker's slug out of an older record's id, ``{protocol}_{slug}_{party}``.

    Args:
        record_id: The older record's id.
        protocol_id: The protocol the record belongs to.
        party: The record's party.

    Returns:
        The slug between the protocol and the party.

    Raises:
        ValueError: If the id does not have that form.
    """
    head = f"{protocol_id}_"
    tail = f"_{party.lower()}"
    if not (record_id.startswith(head) and record_id.endswith(tail)):
        raise ValueError(f"{record_id}: the id does not read as {head}<slug>{tail}.")
    return record_id[len(head) : len(record_id) - len(tail)]


def legacy_key(record_id: str, position: int) -> str:
    """Name one speech in an older record by its position, counted from zero."""
    return f"{record_id}#{position}"


def prefix_offset(dok_id: str, rm: str) -> int:
    """Relate a protocol id's two-character prefix to the year its session starts.

    The Riksdag's ids count sessions in base 36 in their first two characters:
    ``GR`` is 2003/04, ``GZ`` 2011/12 and ``H0`` 2012/13. The difference between
    the prefix read as a number and the session's first year is therefore the
    same for every protocol. ``03_reconcile_legacy.py`` checks that over the
    whole new corpus before it relies on the offset.

    Args:
        dok_id: A protocol id, for example ``HC09139``.
        rm: The protocol's session, for example ``2024/25``.

    Returns:
        The prefix read in base 36, minus the session's first year.
    """
    return int(dok_id[:2], 36) - int(rm.split("/", 1)[0])


def session_of(dok_id: str, offset: int) -> str:
    """Name the session a protocol id belongs to, given the offset of :func:`prefix_offset`."""
    start = int(dok_id[:2], 36) - offset
    return f"{start}/{(start + 1) % 100:02d}"
