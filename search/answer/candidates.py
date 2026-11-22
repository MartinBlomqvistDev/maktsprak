"""The candidates a selector chooses from: the passages BM25 ranks highest for a question.

Both selectors get the same candidates, the ten passages ranked highest, under
the question's filter when it has one. The index is used only if it was built
from the corpus on disk: :func:`check_index` compares the index's manifest with
the passage table and with the hashes of both files, and refuses an index built
from anything else. A passage's position in the index is its row in
``passages.parquet``, so an index built from another passages file would put
the right score under the wrong passage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from search.answer.models import Candidate, Speech
from search.corpus.table import PassageTable
from search.eval.generation import passage_span
from search.index.lexical import (
    Bm25Params,
    Hit,
    IndexManifest,
    LexicalIndex,
    file_sha256,
    passage_ids_sha256,
)
from search.models import text_sha256

#: How many passages each question gets.
CANDIDATES = 10

#: The columns of ``anforanden.parquet`` a candidate and a quotation need.
SPEECH_COLUMNS = ["anf_id", "party", "speaker_raw", "date", "title", "url", "paragraphs"]


class StaleIndexError(RuntimeError):
    """The index was not built from the corpus it is used with."""


def check_index(
    manifest: IndexManifest, table: PassageTable, speeches_path: Path, passages_path: Path
) -> None:
    """Refuse an index whose manifest does not match the corpus on disk.

    Args:
        manifest: The index's manifest.
        table: The passages, read from *passages_path*.
        speeches_path: ``anforanden.parquet``.
        passages_path: ``passages.parquet``.

    Raises:
        StaleIndexError: If the number of passages, the hash of their ids in
            order, or the hash of either file differs from the manifest.
    """
    problems = []
    if manifest.passages != len(table):
        problems.append(f"{manifest.passages} passages in the index, {len(table)} in the table")
    if manifest.passage_ids_sha256 != passage_ids_sha256(table.passage_ids):
        problems.append("the passage ids differ")
    if manifest.passages_sha256 != file_sha256(passages_path):
        problems.append(f"{passages_path.name} differs from the one the index was built from")
    if manifest.speeches_sha256 != file_sha256(speeches_path):
        problems.append(f"{speeches_path.name} differs from the one the index was built from")
    if problems:
        raise StaleIndexError("The index does not match the corpus: " + "; ".join(problems) + ".")


def load_speeches(speeches_path: Path, anf_ids: Iterable[str]) -> dict[str, Speech]:
    """Read some speeches from the corpus, each with the hash of its text.

    The hash is computed from the paragraphs as read, so it always describes the
    text a candidate shows and a quotation quotes.

    Args:
        speeches_path: ``anforanden.parquet``.
        anf_ids: The speeches' keys.

    Returns:
        The speeches, by key. A key the corpus lacks is left out.
    """
    wanted = sorted(set(anf_ids))
    if not wanted:
        return {}
    rows = pq.read_table(
        speeches_path, columns=SPEECH_COLUMNS, filters=pc.field("anf_id").isin(wanted)
    ).to_pylist()
    return {
        row["anf_id"]: Speech(
            anf_id=row["anf_id"],
            party=row["party"],
            speaker=row["speaker_raw"],
            date=row["date"],
            title=row["title"],
            url=row["url"],
            paragraphs=tuple(row["paragraphs"]),
            text_sha256=text_sha256(row["paragraphs"]),
        )
        for row in rows
    }


def rank_passages(
    index: LexicalIndex,
    query: str,
    params: Bm25Params,
    allowed: np.ndarray | None = None,
    limit: int = CANDIDATES,
) -> list[Hit]:
    """The passages BM25 ranks highest for a question, best first.

    Args:
        index: The BM25 index.
        query: The question, as written.
        params: ``k1`` and ``b``.
        allowed: A mask of the passages the question's filter allows, or ``None``.
        limit: How many passages.

    Returns:
        At most *limit* hits; fewer when fewer passages hold a term of the question.
    """
    return index.search(query, params, limit=limit, allowed=allowed)


def build_candidates(
    hits: Sequence[Hit], table: PassageTable, speeches: Mapping[str, Speech]
) -> list[Candidate]:
    """Turn ranked hits into candidates, each with its paragraphs and its speech's fields.

    Args:
        hits: The hits, best first.
        table: The passages, in index order.
        speeches: The speeches the hits belong to, by key.

    Returns:
        The candidates, ranked from 1.

    Raises:
        StaleIndexError: If a hit's speech is missing or too short for its passage.
    """
    candidates = []
    for rank, hit in enumerate(hits, start=1):
        key = table.passage_ids[hit.passage]
        anf_id, first, last = passage_span(key)
        speech = speeches.get(anf_id)
        if speech is None or len(speech.paragraphs) <= last:
            raise StaleIndexError(f"{key} does not fit the speech the corpus holds.")
        candidates.append(
            Candidate(
                rank=rank,
                passage_id=key,
                anf_id=anf_id,
                first=first,
                last=last,
                score=hit.score,
                party=speech.party,
                speaker=speech.speaker,
                date=speech.date,
                title=speech.title,
                url=speech.url,
                paragraphs=speech.paragraphs[first : last + 1],
                text_sha256=speech.text_sha256,
            )
        )
    return candidates
