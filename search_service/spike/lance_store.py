"""Option A: one LanceDB table inside the service, the index baked into the image.

The table holds every passage with its text, the fields a search filters on and
its vector. LanceDB's own full-text index ranks the lexical leg, set up to match
the tuned analyzer (``spike.yaml``), and scalar indexes serve the filters. A
search is LanceDB's hybrid query: the full-text and the vector search run under
the same filter, applied before each, and LanceDB's ``RRFReranker`` merges them.
The ranking reads only each candidate's id, and the columns of the hits returned
are read by row id afterwards, so a search reads the text of its hits alone.

LanceDB's BM25 is not the tuned one. On the real passages its scores equal
Lucene's BM25 with ``k1`` 1.2 and ``b`` 0.75, and a word repeated in a question
counts each time; the tuned BM25 has ``b`` 0.9 and counts it once (see
``probe_fts.py``).
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS, Bitmap, BTree, IvfHnswSq, IvfPq
from lancedb.rerankers import RRFReranker
from pydantic import BaseModel, ConfigDict

from search.index.lexical import file_sha256, passage_ids_sha256
from search.parties import PARTY_ORDER
from search_service.spike.contracts import (
    Hit,
    Mode,
    PartyViewRequest,
    PartyViewResponse,
    SearchFilter,
    SearchRequest,
    SearchResponse,
    Timings,
)
from search_service.spike.inputs import (
    CorpusFiles,
    PassageColumns,
    iter_texts,
    read_passage_columns,
)
from search_service.spike.settings import (
    AnnParams,
    FtsSettings,
    SpikeSettings,
    StoreSpec,
    VectorSettings,
)
from search_service.spike.vectors import encode, query_vector, synthetic_chunks

TABLE_NAME = "passages"
STORE_MANIFEST_NAME = "store.json"

#: The columns a search returns.
COLUMNS: tuple[str, ...] = ("passage_id", "party", "day", "text")

#: What a ranking reads of each candidate. The columns of the hits returned are
#: read afterwards by LanceDB's row id, so the text of a candidate that is not
#: returned is never read, as in a numpy store.
RANKED_COLUMNS: tuple[str, ...] = ("passage_id",)
ROW_ID = "_rowid"

_SCALAR = {"BTREE": BTree, "BITMAP": Bitmap}


class IndexInfo(BaseModel):
    """One index of the table, as LanceDB lists it.

    Attributes:
        name: The index's name.
        index_type: Its type.
        columns: The columns it covers.
        size_bytes: Its size on disk.
        details: What LanceDB records about how it was built.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    index_type: str
    columns: tuple[str, ...]
    size_bytes: int | None
    details: dict[str, Any]


class LanceStoreManifest(BaseModel):
    """What a LanceDB store was built from and what it holds.

    Attributes:
        passages: How many passages the table holds.
        passage_ids_sha256: The hash of the passage ids in order, one per line.
        passages_sha256: The hash of the passages file.
        speeches_sha256: The hash of the speeches file.
        lancedb: The version of LanceDB that built it.
        dim: The vectors' dimension.
        format: How each vector is stored.
        index: The vector index, or ``none``.
        fts: How the full-text index was set up.
        metric: The distance of the vector index and of every vector search.
        vector_seed: The seed of the synthetic vectors.
        chunk_rows: The size of their chunks.
        indices: Every index of the table.
        seconds: How long each step of the build took.
        bytes: The size of the store on disk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passages: int
    passage_ids_sha256: str
    passages_sha256: str
    speeches_sha256: str
    lancedb: str
    dim: int
    format: str
    index: str
    fts: FtsSettings
    metric: str
    vector_seed: int
    chunk_rows: int
    indices: tuple[IndexInfo, ...]
    seconds: dict[str, float]
    bytes: int


def table_schema(dim: int, fmt: str) -> pa.Schema:
    """The table's schema, with vectors of *dim* float16 or float32 values."""
    value = pa.float16() if fmt == "float16" else pa.float32()
    return pa.schema(
        [
            pa.field("passage_id", pa.string(), nullable=False),
            pa.field("party", pa.string(), nullable=False),
            pa.field("day", pa.date32(), nullable=False),
            pa.field("person_id", pa.string()),
            pa.field("text", pa.string(), nullable=False),
            pa.field("vector", pa.list_(value, dim), nullable=False),
        ]
    )


def record_batches(
    columns: PassageColumns,
    texts: Iterator[tuple[str, str]],
    spec: StoreSpec,
    vectors: VectorSettings,
) -> Iterator[pa.RecordBatch]:
    """The table's rows, one chunk of synthetic vectors at a time.

    Args:
        columns: The passages' fields, in index order.
        texts: ``(passage_id, text)`` pairs in the same order.
        spec: The store being built.
        vectors: How the synthetic vectors are drawn.

    Yields:
        One record batch per chunk.

    Raises:
        ValueError: If the texts and the passages are not in the same order.
    """
    schema = table_schema(spec.dim, spec.format)
    value_type = schema.field("vector").type.value_type
    start = 0
    for block in synthetic_chunks(len(columns), spec.dim, vectors.seed, vectors.chunk_rows):
        end = start + len(block)
        chunk = list(islice(texts, len(block)))
        ids = [passage for passage, _ in chunk]
        if ids != columns.passage_ids[start:end]:
            raise ValueError("The texts and the passages are not in the same order.")
        stored = encode(block, spec.format)
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(ids, pa.string()),
                pa.array(columns.parties[start:end], pa.string()),
                pa.array(columns.days[start:end], pa.int32()).cast(pa.date32()),
                pa.array(columns.person_ids[start:end], pa.string()),
                pa.array([text for _, text in chunk], pa.string()),
                pa.FixedSizeListArray.from_arrays(
                    pa.array(stored.reshape(-1), type=value_type), spec.dim
                ),
            ],
            schema=schema,
        )
        start = end


def _directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def build_lance_store(
    directory: Path,
    files: CorpusFiles,
    spec: StoreSpec,
    settings: SpikeSettings,
    limit: int | None = None,
) -> LanceStoreManifest:
    """Build the LanceDB table of one store, with its full-text, scalar and vector indexes.

    Args:
        directory: Where to write; created if missing.
        files: The corpus files.
        spec: The store being built; its kind must be ``lance``.
        settings: The spike's settings.
        limit: Keep only the first *limit* passages; ``None`` keeps all.

    Returns:
        The store's manifest, also written as ``store.json``.
    """
    if spec.kind != "lance":
        raise ValueError(f"{spec.name} is not a LanceDB store.")
    directory.mkdir(parents=True, exist_ok=True)
    columns = read_passage_columns(files.passages, limit)
    seconds: dict[str, float] = {}

    started = perf_counter()
    database = lancedb.connect(directory)
    schema = table_schema(spec.dim, spec.format)
    batches = record_batches(columns, iter_texts(files, limit), spec, settings.vectors)
    reader = pa.RecordBatchReader.from_batches(schema, batches)
    table = database.create_table(TABLE_NAME, data=reader, mode="overwrite")
    seconds["table"] = round(perf_counter() - started, 1)

    started = perf_counter()
    table.create_index("text", config=FTS(**settings.lance.fts.model_dump()))
    seconds["fts"] = round(perf_counter() - started, 1)

    started = perf_counter()
    for column, kind in settings.lance.scalar_indexes.items():
        table.create_index(column, config=_SCALAR[kind]())
    seconds["scalar"] = round(perf_counter() - started, 1)

    if spec.index != "none":
        started = perf_counter()
        config = IvfPq if spec.index == "IVF_PQ" else IvfHnswSq
        table.create_index("vector", config=config(distance_type=settings.lance.metric))
        seconds["vector"] = round(perf_counter() - started, 1)

    indices = tuple(
        IndexInfo(
            name=index.name,
            index_type=str(index.index_type),
            columns=tuple(index.columns),
            size_bytes=getattr(index, "size_bytes", None),
            details=dict(getattr(index, "index_details", None) or {}),
        )
        for index in table.list_indices()
    )
    manifest = LanceStoreManifest(
        passages=len(columns),
        passage_ids_sha256=passage_ids_sha256(columns.passage_ids),
        passages_sha256=file_sha256(files.passages),
        speeches_sha256=file_sha256(files.speeches),
        lancedb=importlib.metadata.version("lancedb"),
        dim=spec.dim,
        format=spec.format,
        index=spec.index,
        fts=settings.lance.fts,
        metric=settings.lance.metric,
        vector_seed=settings.vectors.seed,
        chunk_rows=settings.vectors.chunk_rows,
        indices=indices,
        seconds=seconds,
        bytes=_directory_bytes(directory),
    )
    (directory / STORE_MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def where_clause(search_filter: SearchFilter) -> str | None:
    """The filter as the SQL predicate LanceDB takes, or ``None`` when it allows all.

    The values are safe to put in the predicate as they are: the filter accepts
    only the eight parties, dates, and person ids made of digits.
    """
    clauses = []
    if search_filter.party is not None:
        clauses.append(f"party = '{search_filter.party}'")
    if search_filter.start is not None:
        clauses.append(f"day >= DATE '{search_filter.start.isoformat()}'")
    if search_filter.end is not None:
        clauses.append(f"day <= DATE '{search_filter.end.isoformat()}'")
    if search_filter.person_id is not None:
        clauses.append(f"person_id = '{search_filter.person_id}'")
    return " AND ".join(clauses) if clauses else None


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


class LanceStore:
    """A LanceDB store opened for search.

    Attributes:
        manifest: What the store holds.
        spec: The store's settings.
        settings: The spike's settings.
        table: The table.
    """

    def __init__(self, directory: Path, spec: StoreSpec, settings: SpikeSettings) -> None:
        self.manifest = LanceStoreManifest.model_validate_json(
            (directory / STORE_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        if (self.manifest.dim, self.manifest.format, self.manifest.index) != (
            spec.dim,
            spec.format,
            spec.index,
        ):
            raise ValueError(f"The store in {directory} was not built as {spec.name}.")
        self.spec = spec
        self.settings = settings
        self.table = lancedb.connect(directory).open_table(TABLE_NAME)
        self.reranker = RRFReranker(K=settings.search.rrf_k)
        if self.table.count_rows() != self.manifest.passages:
            raise ValueError(f"The table in {directory} does not hold what its manifest says.")

    @property
    def passages(self) -> int:
        """How many passages the store holds."""
        return self.manifest.passages

    def _with_ann(self, query: Any, ann: AnnParams) -> Any:
        if ann.nprobes is not None:
            query = query.nprobes(ann.nprobes)
        if ann.refine_factor is not None:
            query = query.refine_factor(ann.refine_factor)
        if ann.ef is not None:
            query = query.ef(ann.ef)
        return query

    def query(self, vector_seed: int, vector: np.ndarray | None) -> np.ndarray:
        """The question's vector: the one given, or the synthetic one of its seed.

        Raises:
            ValueError: If the vector given does not have the store's dimension.
        """
        if vector is None:
            return query_vector(vector_seed, self.spec.dim)
        if vector.shape != (self.spec.dim,):
            raise ValueError(f"The store holds {self.spec.dim} dimensions, not {vector.shape}.")
        return vector.astype(np.float32, copy=False)

    def _run(
        self,
        text: str,
        vector: np.ndarray,
        mode: Mode,
        where: str | None,
        ann: AnnParams,
        limit: int,
    ) -> pa.Table:
        """Rank the passages of one query, reading only their ids and scores, best first."""
        if mode == "lexical":
            query = self.table.search(text, query_type="fts", fts_columns="text")
        else:
            if mode == "vector":
                query = self.table.search(vector, vector_column_name="vector")
            else:
                query = (
                    self.table.search(
                        query_type="hybrid", vector_column_name="vector", fts_columns="text"
                    )
                    .vector(vector)
                    .text(text)
                    .rerank(self.reranker)
                )
            query = self._with_ann(query.distance_type(self.settings.lance.metric), ann)
        if where is not None:
            query = query.where(where, prefilter=True)
        return query.select(list(RANKED_COLUMNS)).with_row_id(True).limit(limit).to_arrow()

    def _hits(self, ranked: pa.Table, count: int, mode: Mode) -> tuple[Hit, ...]:
        """The first *count* ranked passages, their columns read by row id."""
        top = ranked.slice(0, count).to_pylist()
        if not top:
            return ()
        row_ids = [row[ROW_ID] for row in top]
        taken = self.table.take_row_ids(row_ids, with_row_id=True).select(list(COLUMNS))
        by_id = {row[ROW_ID]: row for row in taken.to_arrow().to_pylist()}
        return tuple(self._hit({**by_id[row[ROW_ID]], **row}, mode) for row in top)

    @staticmethod
    def _hit(row: dict[str, Any], mode: Mode) -> Hit:
        if mode == "hybrid":
            score = row["_relevance_score"]
        elif mode == "lexical":
            score = row["_score"]
        else:
            score = -row["_distance"]
        return Hit(
            passage_id=row["passage_id"],
            party=row["party"],
            date=row["day"],
            score=round(float(score), 6),
            text=row["text"],
        )

    def search(self, request: SearchRequest, vector: np.ndarray | None = None) -> SearchResponse:
        """Search the store with LanceDB's own query of the request's mode.

        Args:
            request: The question, its filter and which legs run.
            vector: The question's vector from a query model; ``None`` uses the
                synthetic vector of the request's seed.

        Returns:
            The best hits; LanceDB times its parts together, so only the total is given.
        """
        start = perf_counter()
        rows = self._run(
            request.text,
            self.query(request.vector_seed, vector),
            request.mode,
            where_clause(request.filter),
            request.ann,
            self.settings.search.depth,
        )
        hits = self._hits(rows, self.settings.search.hits, request.mode)
        return SearchResponse(
            hits=hits, allowed=None, timings=Timings(total_ms=_ms(start, perf_counter()))
        )

    def party_view(
        self, request: PartyViewRequest, vector: np.ndarray | None = None
    ) -> PartyViewResponse:
        """The best passage of each party: one hybrid query per party, under the request's filter.

        Args:
            request: The question and a filter that names no party.
            vector: The question's vector from a query model; ``None`` uses the
                synthetic vector of the request's seed.

        Returns:
            The best hit of each party, in the site's order.
        """
        start = perf_counter()
        query = self.query(request.vector_seed, vector)
        hits: dict[str, Hit | None] = {}
        for party in PARTY_ORDER:
            where = where_clause(request.filter.model_copy(update={"party": party}))
            rows = self._run(
                request.text, query, "hybrid", where, request.ann, self.settings.search.depth
            )
            best = self._hits(rows, 1, "hybrid")
            hits[party] = best[0] if best else None
        return PartyViewResponse(hits=hits, timings=Timings(total_ms=_ms(start, perf_counter())))
