"""Counting tokens in the corpus with a tokenizer.

The cost of embedding the corpus depends on how many tokens an embedding model
reads, and every model has its own tokenizer. These functions count with any
``tokenizers.Tokenizer``, so the same corpus can be measured against each
tokenizer family under consideration. Counts leave out special tokens, which a
model adds per input and which depend on how the corpus is cut into inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from tokenizers import Tokenizer

DEFAULT_BATCH_SIZE = 2048

#: The longest input the project's classifier takes, in tokens: the
#: ``max_position_embeddings`` in its ``config.json``. Its tokenizer adds two
#: special tokens to a text, which also count towards that limit. The counts
#: leave special tokens out and are compared with this figure as they are, so a
#: text counts as over the limit only when its tokens alone number more than
#: 512; a text of 511 or 512 tokens does not.
INPUT_LIMIT_TOKENS = 512


class LengthSpread(BaseModel):
    """How long the texts were, in tokens.

    Percentiles use the nearest-rank method: the smallest length such that at
    least that share of the texts is no longer than it.

    Attributes:
        median: The 50th percentile.
        p90: The 90th percentile.
        p99: The 99th percentile.
        maximum: The longest text.
        over_input_limit: How many texts have more than ``INPUT_LIMIT_TOKENS``
            tokens, special tokens left out. A text of 511 or 512 tokens is
            not in this figure although the classifier's two special tokens
            would take its input past the limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    median: int
    p90: int
    p99: int
    maximum: int
    over_input_limit: int


class TokenizerCount(BaseModel):
    """The corpus counted with one tokenizer.

    Attributes:
        name: A short name for the tokenizer family.
        model_type: The tokenizer's algorithm as its own file declares it, for
            example ``WordPiece`` or ``Unigram``.
        source: Where the tokenizer came from.
        serialised_sha256: The SHA-256 of the serialised form of the tokenizer
            the counts were made with, see :func:`serialised_sha256_of`.
        vocab_size: The size of its vocabulary.
        total_tokens: The tokens in every paragraph, special tokens left out.
        tokens_per_word: ``total_tokens`` over the number of words.
        tokens_per_paragraph: The spread of the paragraphs' lengths in tokens.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    model_type: str
    source: str
    serialised_sha256: str
    vocab_size: int
    total_tokens: int
    tokens_per_word: float
    tokens_per_paragraph: LengthSpread


class TokenCountReport(BaseModel):
    """What the corpus holds, counted in words and in tokens.

    Attributes:
        paragraphs: The speakers' paragraphs, which are the text that is quoted.
        words: Whitespace-separated words in those paragraphs.
        characters: Characters in those paragraphs, whitespace included.
        tokenizers: One entry per tokenizer that could be loaded.
        unavailable: For each tokenizer that could not be loaded, why.
        note: What the counts are and are not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    paragraphs: int
    words: int
    characters: int
    tokenizers: tuple[TokenizerCount, ...]
    unavailable: dict[str, str]
    note: str


def _batches(texts: Iterable[str], size: int) -> Iterator[list[str]]:
    """Cut an iterable into lists of at most *size* items."""
    iterator = iter(texts)
    while batch := list(islice(iterator, size)):
        yield batch


def token_lengths(
    texts: Iterable[str], tokenizer: Tokenizer, batch_size: int = DEFAULT_BATCH_SIZE
) -> Iterator[int]:
    """Yield the number of tokens in each text, special tokens left out.

    Args:
        texts: The texts to count, in any order.
        tokenizer: The tokenizer to count with. It must not truncate or pad,
            because either would make every count a function of its setting.
        batch_size: How many texts to encode in one call.

    Yields:
        One count per text, in the order the texts came.

    Raises:
        ValueError: If the tokenizer truncates or pads, or the batch size is
            smaller than one.
    """
    if batch_size < 1:
        raise ValueError(f"The batch size must be at least 1; got {batch_size}.")
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError(
            "The tokenizer truncates or pads, which would distort the counts. "
            "Call no_truncation() and no_padding() on it first."
        )
    for batch in _batches(texts, batch_size):
        for encoding in tokenizer.encode_batch_fast(batch, add_special_tokens=False):
            yield len(encoding.ids)


def percentile(sorted_values: Sequence[int], share: float) -> int:
    """Return the nearest-rank percentile of values that are already sorted.

    Args:
        sorted_values: The values, smallest first. There must be at least one.
        share: The share of the values the result must cover, between 0 and 1.

    Returns:
        The smallest value such that at least *share* of the values are no
        larger than it.

    Raises:
        ValueError: If there are no values or the share is outside 0 to 1.
    """
    if not sorted_values:
        raise ValueError("A percentile needs at least one value.")
    if not 0 <= share <= 1:
        raise ValueError(f"The share must lie between 0 and 1; got {share}.")
    rank = max(1, math.ceil(share * len(sorted_values)))
    return sorted_values[rank - 1]


def spread_of(lengths: Iterable[int]) -> LengthSpread:
    """Summarise a collection of token counts.

    Args:
        lengths: One count per text. There must be at least one.

    Returns:
        The percentiles, the maximum and how many exceed the input limit.

    Raises:
        ValueError: If there are no counts.
    """
    ordered = sorted(lengths)
    return LengthSpread(
        median=percentile(ordered, 0.5),
        p90=percentile(ordered, 0.9),
        p99=percentile(ordered, 0.99),
        maximum=ordered[-1],
        over_input_limit=sum(1 for length in ordered if length > INPUT_LIMIT_TOKENS),
    )


def model_type_of(tokenizer: Tokenizer) -> str:
    """Return the algorithm a tokenizer declares in its own serialised form."""
    return str(json.loads(tokenizer.to_str())["model"]["type"])


def serialised_sha256_of(tokenizer: Tokenizer) -> str:
    """Hash a tokenizer's serialised form, which holds its vocabulary and its settings.

    The form is the JSON that ``tokenizer.to_str()`` returns, hashed as UTF-8.
    It includes the truncation and padding settings, so a tokenizer hashes
    differently with either switched on. The ``tokenizers`` library lays the
    JSON out in its own way, so the hash can differ from that of the
    ``tokenizer.json`` file the tokenizer was read from.

    Args:
        tokenizer: The tokenizer.

    Returns:
        The SHA-256, in hex.
    """
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
