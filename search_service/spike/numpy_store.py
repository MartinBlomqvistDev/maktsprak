"""Option B: the BM25 index of :mod:`search.index.lexical`, and the vectors as a numpy matrix.

The store is a directory of plain files. The BM25 index is the one the tuning
chose, copied unchanged, so the lexical leg is the BM25 of every comparison. The
text and the passage ids are UTF-8 files with offsets, and the fields a search
filters on are arrays, one value per passage: the party as a code in the site's
order of the parties, the date as days since 1970-01-01, the speaker as a code
into a list of the Riksdag's person ids. Every vector file is a matrix with one
row per passage, searched exactly.

Everything large is memory-mapped, so opening the store reads almost nothing:
the first searches read what they touch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.index.analyzer import Analyzer
from search.index.lexical import (
    ARRAY_NAMES,
    MANIFEST_NAME,
    TERMS_NAME,
    Bm25Params,
    IndexManifest,
    LexicalIndex,
    build_index,
    file_sha256,
    passage_ids_sha256,
    tokenize_corpus,
)
from search.parties import PARTY_ORDER
from search_service.spike.blob import StringBlob, write_strings
from search_service.spike.contracts import (
    Hit,
    PartyViewRequest,
    PartyViewResponse,
    SearchFilter,
    SearchRequest,
    SearchResponse,
    Timings,
)
from search_service.spike.fusion import rrf
from search_service.spike.inputs import (
    CorpusFiles,
    day_number,
    day_of,
    iter_texts,
    read_passage_columns,
)
from search_service.spike.settings import SearchSettings, VectorFormat, VectorSettings
from search_service.spike.vectors import (
    VectorMatrix,
    query_vector,
    scales_name,
    top_positions,
    vectors_name,
    write_vectors,
)

STORE_DIR_NAME = "numpy"
STORE_MANIFEST_NAME = "store.json"
LEXICAL_NAME = "lexical"
PARTY_NAME = "party.npy"
DAY_NAME = "day.npy"
PERSON_NAME = "person.npy"
PERSONS_NAME = "persons.json"
PASSAGE_IDS_STEM = "passage_ids"
TEXT_STEM = "text"


class VectorFile(BaseModel):
    """One matrix of vectors in a numpy store.

    Attributes:
        dim: The dimension.
        format: How each vector is stored.
        file: The file's name.
        bytes: Its size on disk, the scales of an int8 matrix included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dim: int
    format: VectorFormat
    file: str
    bytes: int


class NumpyStoreManifest(BaseModel):
    """What a numpy store was built from and what it holds.

    Attributes:
        passages: How many passages it holds.
        passage_ids_sha256: The hash of the passage ids in order, one per line.
        passages_sha256: The hash of the passages file.
        speeches_sha256: The hash of the speeches file.
        analyzer: The BM25 index's analyzer.
        params: The BM25 parameters the tuning chose.
        persons: How many distinct speakers the speaker codes name.
        vector_seed: The seed of the synthetic vectors.
        chunk_rows: The size of their chunks.
        vectors: The vector files.
        bytes: The size of each part of the store on disk, by part.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passages: int
    passage_ids_sha256: str
    passages_sha256: str
    speeches_sha256: str
    analyzer: Analyzer
    params: Bm25Params
    persons: int
    vector_seed: int
    chunk_rows: int
    vectors: tuple[VectorFile, ...]
    bytes: dict[str, int]


def _size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def _copy_lexical(files: CorpusFiles, target: Path, expected: dict[str, object]) -> None:
    """Copy the chosen BM25 index after checking that it was built from these passages.

    Args:
        files: The corpus files; ``files.lexical`` is the index's directory.
        target: Where to copy it.
        expected: The value each checked field of the index's manifest must have.

    Raises:
        ValueError: If the index was built from another corpus or with another analyzer.
    """
    manifest = IndexManifest.model_validate_json(
        (files.lexical / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise ValueError(
                f"The BM25 index in {files.lexical} differs from the corpus in {field}."
            )
    target.mkdir(parents=True, exist_ok=True)
    for name in [f"{array}.npy" for array in ARRAY_NAMES] + [TERMS_NAME, MANIFEST_NAME]:
        shutil.copyfile(files.lexical / name, target / name)


def build_numpy_store(
    directory: Path,
    files: CorpusFiles,
    shapes: list[tuple[int, VectorFormat]],
    vectors: VectorSettings,
    limit: int | None = None,
) -> NumpyStoreManifest:
    """Build a numpy store of the passages, with one vector file per shape.

    With every passage, the BM25 index is the tuning's own, copied after its
    manifest is checked against the corpus. With *limit*, the index is built
    again over the passages kept, with the same analyzer.

    Args:
        directory: Where to write; created if missing.
        files: The corpus files.
        shapes: The ``(dimension, format)`` of each vector file.
        vectors: How the synthetic vectors are drawn.
        limit: Keep only the first *limit* passages; ``None`` keeps all.

    Returns:
        The store's manifest, also written as ``store.json``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    columns = read_passage_columns(files.passages, limit)
    rows = len(columns)

    np.save(
        directory / PARTY_NAME, np.array([PARTY_ORDER.index(p) for p in columns.parties], np.uint8)
    )
    np.save(directory / DAY_NAME, columns.days)
    persons = sorted({person for person in columns.person_ids if person is not None})
    code = {person: number for number, person in enumerate(persons)}
    person_codes = [code[person] if person is not None else -1 for person in columns.person_ids]
    np.save(directory / PERSON_NAME, np.array(person_codes, dtype=np.int32))
    (directory / PERSONS_NAME).write_text(json.dumps(persons), encoding="utf-8", newline="\n")
    write_strings(directory / PASSAGE_IDS_STEM, columns.passage_ids)

    source = {
        "analyzer": files.choice.analyzer,
        "passages": rows,
        "passage_ids_sha256": passage_ids_sha256(columns.passage_ids),
        "passages_sha256": file_sha256(files.passages),
        "speeches_sha256": file_sha256(files.speeches),
    }
    pairs = iter_texts(files, limit)
    if limit is None:
        write_strings(directory / TEXT_STEM, (text for _, text in pairs))
        _copy_lexical(files, directory / LEXICAL_NAME, source)
    else:
        kept = list(pairs)
        write_strings(directory / TEXT_STEM, (text for _, text in kept))
        index = build_index(tokenize_corpus(kept), files.choice.analyzer)
        index.save(
            directory / LEXICAL_NAME,
            IndexManifest(
                **source,
                terms=len(index.terms),
                postings=len(index.doc_ids),
                tokens=int(index.doc_lengths.sum()),
                mean_passage_terms=round(index.mean_length, 3),
            ),
        )

    written = []
    for dim, fmt in shapes:
        path = write_vectors(directory, rows, dim, fmt, vectors.seed, vectors.chunk_rows)
        extra = [directory / scales_name(dim)] if fmt == "int8" else []
        written.append(VectorFile(dim=dim, format=fmt, file=path.name, bytes=_size([path, *extra])))

    lexical_files = list((directory / LEXICAL_NAME).iterdir())
    manifest = NumpyStoreManifest(
        passages=rows,
        passage_ids_sha256=str(source["passage_ids_sha256"]),
        passages_sha256=str(source["passages_sha256"]),
        speeches_sha256=str(source["speeches_sha256"]),
        analyzer=files.choice.analyzer,
        params=files.choice.params,
        persons=len(persons),
        vector_seed=vectors.seed,
        chunk_rows=vectors.chunk_rows,
        vectors=tuple(written),
        bytes={
            "lexical": _size(lexical_files),
            "text": _size([directory / "text.bin", directory / "text.offsets.npy"]),
            "passage_ids": _size(
                [directory / "passage_ids.bin", directory / "passage_ids.offsets.npy"]
            ),
            "filters": _size(
                [directory / name for name in (PARTY_NAME, DAY_NAME, PERSON_NAME, PERSONS_NAME)]
            ),
        },
    )
    (directory / STORE_MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


class NumpyStore:
    """A numpy store opened for search, with the vectors of one shape and format.

    Attributes:
        manifest: What the store holds.
        dim: The vectors' dimension.
        format: Their format.
        settings: The shape of one search.
    """

    def __init__(
        self, directory: Path, dim: int, fmt: VectorFormat, settings: SearchSettings
    ) -> None:
        self.manifest = NumpyStoreManifest.model_validate_json(
            (directory / STORE_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        if not any(v.dim == dim and v.format == fmt for v in self.manifest.vectors):
            raise ValueError(f"The store in {directory} has no {dim}-dimensional {fmt} vectors.")
        self.dim = dim
        self.format = fmt
        self.settings = settings
        self.params = self.manifest.params
        self.lexical, _ = LexicalIndex.load(directory / LEXICAL_NAME)
        self.parties = np.load(directory / PARTY_NAME)
        self.days = np.load(directory / DAY_NAME)
        self.persons = np.load(directory / PERSON_NAME)
        names = json.loads((directory / PERSONS_NAME).read_text(encoding="utf-8"))
        self.person_codes = {person: number for number, person in enumerate(names)}
        self.passage_ids = StringBlob.open(directory / PASSAGE_IDS_STEM)
        self.texts = StringBlob.open(directory / TEXT_STEM)
        self.vectors = VectorMatrix.load(directory, dim, fmt, self.manifest.chunk_rows)
        lengths = {
            self.lexical.passages,
            len(self.parties),
            len(self.days),
            len(self.persons),
            len(self.passage_ids),
            len(self.texts),
            len(self.vectors),
        }
        if lengths != {self.manifest.passages}:
            raise ValueError(f"The parts of the store in {directory} differ in length.")

    @property
    def passages(self) -> int:
        """How many passages the store holds."""
        return self.manifest.passages

    def mask(self, search_filter: SearchFilter) -> np.ndarray | None:
        """The passages a filter allows, or ``None`` when it allows all."""
        if search_filter.empty:
            return None
        allowed = np.ones(self.passages, dtype=bool)
        if search_filter.party is not None:
            allowed &= self.parties == PARTY_ORDER.index(search_filter.party)
        if search_filter.start is not None:
            allowed &= self.days >= day_number(search_filter.start)
        if search_filter.end is not None:
            allowed &= self.days <= day_number(search_filter.end)
        if search_filter.person_id is not None:
            code = self.person_codes.get(search_filter.person_id)
            if code is None:
                allowed[:] = False
            else:
                allowed &= self.persons == code
        return allowed

    def hit(self, position: int, score: float) -> Hit:
        """The hit of one passage, with its text."""
        return Hit(
            passage_id=self.passage_ids[position],
            party=PARTY_ORDER[int(self.parties[position])],
            date=day_of(int(self.days[position])),
            score=round(score, 6),
            text=self.texts[position],
        )

    def _lexical_leg(self, text: str, allowed: np.ndarray | None) -> list[int]:
        hits = self.lexical.search(text, self.params, limit=self.settings.depth, allowed=allowed)
        return [hit.passage for hit in hits]

    def _vector_leg(self, query: np.ndarray, allowed: np.ndarray | None) -> list[int]:
        rows = self.vectors.search(query, self.settings.depth, allowed, self.settings.gather_below)
        return rows.tolist()

    def query(self, vector_seed: int, vector: np.ndarray | None) -> np.ndarray:
        """The question's vector: the one given, or the synthetic one of its seed.

        Raises:
            ValueError: If the vector given does not have the store's dimension.
        """
        if vector is None:
            return query_vector(vector_seed, self.dim)
        if vector.shape != (self.dim,):
            raise ValueError(f"The store holds {self.dim} dimensions, not {vector.shape}.")
        return vector.astype(np.float32, copy=False)

    def search(self, request: SearchRequest, vector: np.ndarray | None = None) -> SearchResponse:
        """Search the store.

        Args:
            request: The question, its filter and which legs run.
            vector: The question's vector from a query model; ``None`` uses the
                synthetic vector of the request's seed.

        Returns:
            The best hits, with where the time went.
        """
        start = perf_counter()
        allowed = self.mask(request.filter)
        count = self.passages if allowed is None else int(allowed.sum())
        filtered = perf_counter()
        lexical: list[int] = []
        if count and request.mode != "vector":
            lexical = self._lexical_leg(request.text, allowed)
        lexical_done = perf_counter()
        nearest: list[int] = []
        if count and request.mode != "lexical":
            nearest = self._vector_leg(self.query(request.vector_seed, vector), allowed)
        vector_done = perf_counter()
        legs = [leg for leg in (lexical, nearest) if leg]
        ranked = rrf(legs, self.settings.rrf_k, self.settings.hits)
        fused = perf_counter()
        hits = tuple(self.hit(position, score) for position, score in ranked)
        end = perf_counter()
        return SearchResponse(
            hits=hits,
            allowed=count,
            timings=Timings(
                total_ms=_ms(start, end),
                filter_ms=_ms(start, filtered),
                lexical_ms=_ms(filtered, lexical_done) if request.mode != "vector" else None,
                vector_ms=_ms(lexical_done, vector_done) if request.mode != "lexical" else None,
                fusion_ms=_ms(vector_done, fused),
                fetch_ms=_ms(fused, end),
            ),
        )

    def party_view(
        self, request: PartyViewRequest, vector: np.ndarray | None = None
    ) -> PartyViewResponse:
        """The best passage of each party for one question.

        Both legs score every passage once; each party then takes its own best
        passages from those scores, under the request's filter.

        Args:
            request: The question and a filter that names no party.
            vector: The question's vector from a query model; ``None`` uses the
                synthetic vector of the request's seed.

        Returns:
            The best hit of each party, in the site's order.
        """
        start = perf_counter()
        base = self.mask(request.filter)
        lexical_scores = self.lexical.scores(request.text, self.params)
        lexical_done = perf_counter()
        vector_scores = self.vectors.scores(self.query(request.vector_seed, vector))
        vector_done = perf_counter()
        hits: dict[str, Hit | None] = {}
        depth = self.settings.depth
        for number, party in enumerate(PARTY_ORDER):
            allowed = self.parties == number
            if base is not None:
                allowed &= base
            rows = np.flatnonzero(allowed)
            matching = rows[lexical_scores[rows] > 0]
            lexical = top_positions(lexical_scores[matching], depth, matching).tolist()
            nearest = top_positions(vector_scores[rows], depth, rows).tolist() if len(rows) else []
            best = rrf([leg for leg in (lexical, nearest) if leg], self.settings.rrf_k, 1)
            hits[party] = self.hit(*best[0]) if best else None
        end = perf_counter()
        return PartyViewResponse(
            hits=hits,
            timings=Timings(
                total_ms=_ms(start, end),
                lexical_ms=_ms(start, lexical_done),
                vector_ms=_ms(lexical_done, vector_done),
                fusion_ms=_ms(vector_done, end),
            ),
        )


def vector_file_names(dim: int, fmt: VectorFormat) -> list[str]:
    """The files that hold the vectors of one shape and format in a numpy store."""
    names = [vectors_name(dim, fmt)]
    if fmt == "int8":
        names.append(scales_name(dim))
    return names
