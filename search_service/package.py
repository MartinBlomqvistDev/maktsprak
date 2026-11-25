"""The search service's package: its index, the speeches' fields, and a manifest of both.

Step 08 (``search/pipeline/08_package_index.py``) builds the package from the
corpus. It is published as a dataset, the service's image fetches it when the
image is built (``bake_index.py``), and the service checks it again when it
starts. The package holds:

- ``store/``: the numpy store that ADR-001 chose, built as the check before the
  test split built it: the BM25 index the tuning chose, the passages' text and
  filter fields, and the vectors of the hybrid's model as a float32 matrix. It
  holds no questions' vectors, since the service asks for each question's
  vector when the question comes.
- ``speeches.parquet``: the fields of every speech that has a passage, in the
  order of the passages, without the text.
- ``speech_rows.npy``: the speech of each passage, as a row of ``speeches.parquet``.
- ``package.json``: what the package holds, how the service searches and
  answers with it, and the size and SHA-256 of every other file.

A speech's text is not stored twice. Its passages are runs of whole paragraphs,
consecutive in the store and together the whole speech, so its paragraphs are
its passages' text in order. Building the package checks that for every speech:
its passages cover its paragraphs from the first to the last, each passage's
text is its paragraphs as the corpus holds them, and the paragraphs have the
hash the corpus gives the speech.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from search.answer.models import Speech
from search.models import text_sha256
from search_service.spike.blob import StringBlob
from search_service.spike.inputs import CorpusFiles, day_of
from search_service.spike.numpy_store import DAY_NAME, TEXT_STEM, build_numpy_store
from search_service.spike.settings import VectorSettings

#: The package's manifest, beside its other files.
MANIFEST_NAME = "package.json"
#: The numpy store, a directory of its own.
STORE_DIR = "store"
#: The speeches' fields.
SPEECHES_NAME = "speeches.parquet"
#: Each passage's speech, as a row of the speeches' table.
SPEECH_ROWS_NAME = "speech_rows.npy"

#: The fields of ``anforanden.parquet`` the package keeps for each speech.
SPEECH_FIELDS = (
    "anf_id",
    "rm",
    "dok_nummer",
    "party",
    "speaker_raw",
    "date",
    "title",
    "is_reply",
    "url",
    "text_sha256",
)

_READ_BATCH_ROWS = 2000
_HASH_CHUNK = 1 << 22


class StalePackageError(RuntimeError):
    """The package on disk is not the one its manifest describes."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PackageFile(_Frozen):
    """One file of the package.

    Attributes:
        bytes: Its size.
        sha256: Its SHA-256 in hex.
    """

    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Retrieval(_Frozen):
    """How the service searches: the hybrid the tuning chose, as the check built it.

    Attributes:
        analyzer: How the BM25 index turns text into terms.
        k1: BM25's ``k1``.
        b: BM25's ``b``.
        target_words: The size the passages were packed towards.
        rrf_k: The constant of reciprocal rank fusion.
        depth: How many passages each leg ranks before fusion.
        hits: How many passages a search returns.
        gather_below: Below this share of all passages allowed by a filter, the
            vector leg scores only those passages.
        chunk_rows: How many vectors are scored at a time.
        vector_wait_s: How long a question waits for its vector before the
            service answers from BM25 alone.
    """

    analyzer: str
    k1: float
    b: float
    target_words: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    depth: int = Field(gt=0)
    hits: int = Field(gt=0)
    gather_below: float = Field(ge=0, le=1)
    chunk_rows: int = Field(gt=0)
    vector_wait_s: float = Field(gt=0)


class Embedding(_Frozen):
    """The model that embeds a question, behind OpenRouter's API.

    Attributes:
        name: The candidate's name in the settings.
        model: The model's id.
        revision: The model's revision.
        provider: The one provider requests are routed to.
        quantization: The one precision the provider serves it in.
        dimension: The vectors' dimension.
        query_prompt: The text put before a question.
    """

    name: str
    model: str
    revision: str
    provider: str
    quantization: str
    dimension: int = Field(gt=0)
    query_prompt: str


class AnswerConfig(_Frozen):
    """How the service answers: the selector the test run measured, and its fallback.

    Attributes:
        model: The language model that selects.
        effort: How much it thinks.
        max_tokens: The most output tokens, thinking included.
        prompt_sha256: The hash of the selector's prompt; the service refuses another.
        threshold: The share of a question's highest possible score below which
            the baseline abstains.
        monthly_budget_usd: What the model may cost in a calendar month before
            the baseline answers.
        input_price: The model's list price per million input tokens, in US dollars.
        output_price: Its list price per million output tokens.
    """

    model: str
    effort: str
    max_tokens: int = Field(gt=0)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold: float = Field(ge=0)
    monthly_budget_usd: float = Field(gt=0)
    input_price: float = Field(ge=0)
    output_price: float = Field(ge=0)


class Sources(_Frozen):
    """What the package was built from, by hash.

    Attributes:
        speeches_sha256: ``anforanden.parquet``.
        passages_sha256: ``passages.parquet``.
        vectors_sha256: The vector index's matrix.
        bm25_tuning_sha256: The report of BM25's tuning.
        hybrid_tuning_sha256: The report of the hybrid's tuning.
        answer_run_sha256: The report of the answer step's run on the hybrid's
            candidates, which gives the threshold and the prompt.
    """

    speeches_sha256: str
    passages_sha256: str
    vectors_sha256: str
    bm25_tuning_sha256: str
    hybrid_tuning_sha256: str
    answer_run_sha256: str


class PackageManifest(_Frozen):
    """What a package holds and how the service uses it.

    Attributes:
        name: The package's name.
        created: When it was built.
        passages: How many passages it holds.
        speeches: How many speeches they belong to.
        first_date: The date of the earliest speech.
        last_date: The date of the latest speech.
        retrieval: How the service searches.
        embedding: The model that embeds a question.
        answer: How the service answers.
        sources: What it was built from.
        files: Every other file, by its path within the package.
    """

    name: str
    created: datetime
    passages: int = Field(gt=0)
    speeches: int = Field(gt=0)
    first_date: date
    last_date: date
    retrieval: Retrieval
    embedding: Embedding
    answer: AnswerConfig
    sources: Sources
    files: dict[str, PackageFile]


@dataclass(frozen=True)
class SpeechCount:
    """How the passages met the speeches.

    Attributes:
        speeches: Speeches with at least one passage, the rows of the table.
        skipped: Speeches of the corpus without a passage.
    """

    speeches: int
    skipped: int


def _speech_rows(speeches_path: Path) -> Iterator[dict[str, Any]]:
    columns = [*SPEECH_FIELDS, "paragraphs"]
    for batch in pq.ParquetFile(speeches_path).iter_batches(
        batch_size=_READ_BATCH_ROWS, columns=columns
    ):
        yield from batch.to_pylist()


def write_speech_table(
    speeches_path: Path, passages_path: Path, texts: StringBlob, out_dir: Path
) -> SpeechCount:
    """Write the speeches' fields and each passage's speech, checking every speech's text.

    The two files are read side by side in their order, as the store's text was
    written. A speech without a passage is skipped.

    Args:
        speeches_path: ``anforanden.parquet``.
        passages_path: ``passages.parquet``.
        texts: The store's text of each passage, in the passages' order.
        out_dir: The package's directory.

    Returns:
        How many speeches the table holds, and how many were skipped.

    Raises:
        StalePackageError: If the passages do not cover a speech from its first
            paragraph to its last, a passage's text is not its paragraphs, or a
            speech's paragraphs do not have the corpus's hash.
    """
    speeches = _speech_rows(speeches_path)
    rows: list[dict[str, Any]] = []
    owner = np.empty(len(texts), dtype=np.int32)
    current: dict[str, Any] | None = None
    covered = 0
    skipped = 0
    position = 0

    def finish(speech: dict[str, Any] | None) -> None:
        if speech is None:
            return
        if covered != len(speech["paragraphs"]):
            raise StalePackageError(f"The passages of {speech['anf_id']} do not cover it.")
        if text_sha256(speech["paragraphs"]) != speech["text_sha256"]:
            raise StalePackageError(f"{speech['anf_id']} does not have the corpus's hash.")

    for batch in pq.ParquetFile(passages_path).iter_batches(
        batch_size=_READ_BATCH_ROWS * 10, columns=["passage_id", "anf_id", "first", "last"]
    ):
        for passage in batch.to_pylist():
            if current is None or passage["anf_id"] != current["anf_id"]:
                finish(current)
                current = next(speeches, None)
                while current is not None and current["anf_id"] != passage["anf_id"]:
                    skipped += 1
                    current = next(speeches, None)
                if current is None:
                    raise StalePackageError(f"{passage['passage_id']} has no speech in order.")
                rows.append(
                    {field: current[field] for field in SPEECH_FIELDS}
                    | {"first_passage": position, "passages": 0}
                )
                covered = 0
            first, last = passage["first"], passage["last"]
            paragraphs = current["paragraphs"]
            if first != covered or last < first or last >= len(paragraphs):
                raise StalePackageError(
                    f"{passage['passage_id']} is not the next run of paragraphs."
                )
            if position >= len(texts) or texts[position] != "\n".join(paragraphs[first : last + 1]):
                raise StalePackageError(f"The store's text of {passage['passage_id']} differs.")
            covered = last + 1
            rows[-1]["passages"] += 1
            owner[position] = len(rows) - 1
            position += 1
    finish(current)
    skipped += sum(1 for _ in speeches)
    if position != len(texts):
        raise StalePackageError(f"The store holds {len(texts)} passages, the corpus {position}.")

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_dir / SPEECHES_NAME, compression="zstd")
    np.save(out_dir / SPEECH_ROWS_NAME, owner)
    return SpeechCount(speeches=len(rows), skipped=skipped)


def hash_file(path: Path) -> str:
    """The SHA-256 of a file, read in chunks of 4 MiB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(directory: Path) -> list[str]:
    """Every file of a package but its manifest, as paths within it with forward slashes."""
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )


def build_package(
    out_dir: Path,
    *,
    name: str,
    files: CorpusFiles,
    vectors_from: Path,
    vectors: VectorSettings,
    retrieval: Retrieval,
    embedding: Embedding,
    answer: AnswerConfig,
    report_hashes: dict[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> PackageManifest:
    """Build a package: the store, the speeches' table, and the manifest.

    Args:
        out_dir: Where to write; it must not exist yet.
        name: The package's name.
        files: The corpus files and the BM25 the tuning chose.
        vectors_from: The vector index of the hybrid's model.
        vectors: The chunks the store scores vectors in, as the check built it.
        retrieval: How the service searches.
        embedding: The model that embeds a question.
        answer: How the service answers.
        report_hashes: The hashes of the reports the settings come from:
            ``bm25_tuning``, ``hybrid_tuning`` and ``answer_run``.
        clock: When the package is built.

    Returns:
        The manifest, also written as ``package.json``.

    Raises:
        FileExistsError: If *out_dir* exists.
        StalePackageError: If a speech's text does not check out.
    """
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a package is built into a new directory.")
    out_dir.mkdir(parents=True)
    store_dir = out_dir / STORE_DIR
    store = build_numpy_store(
        store_dir, files, [(embedding.dimension, "float32")], vectors, vectors_from=vectors_from
    )
    if store.vectors_source_sha256 is None:
        raise StalePackageError("The store was built without the vectors it was given.")
    count = write_speech_table(
        files.speeches, files.passages, StringBlob.open(store_dir / TEXT_STEM), out_dir
    )
    days = np.load(store_dir / DAY_NAME)
    manifest = PackageManifest(
        name=name,
        created=clock(),
        passages=store.passages,
        speeches=count.speeches,
        first_date=day_of(int(days.min())),
        last_date=day_of(int(days.max())),
        retrieval=retrieval,
        embedding=embedding,
        answer=answer,
        sources=Sources(
            speeches_sha256=store.speeches_sha256,
            passages_sha256=store.passages_sha256,
            vectors_sha256=store.vectors_source_sha256,
            bm25_tuning_sha256=report_hashes["bm25_tuning"],
            hybrid_tuning_sha256=report_hashes["hybrid_tuning"],
            answer_run_sha256=report_hashes["answer_run"],
        ),
        files={
            relative: PackageFile(
                bytes=(out_dir / relative).stat().st_size, sha256=hash_file(out_dir / relative)
            )
            for relative in package_files(out_dir)
        },
    )
    (out_dir / MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def read_manifest(directory: Path) -> tuple[PackageManifest, str]:
    """Read a package's manifest, with the SHA-256 of its bytes."""
    raw = (directory / MANIFEST_NAME).read_bytes()
    return PackageManifest.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def verify_package(
    directory: Path, *, manifest_sha256: str | None = None, hashes: bool = True
) -> PackageManifest:
    """Check a package against its manifest, and the manifest against a pinned hash.

    Args:
        directory: The package.
        manifest_sha256: The hash the manifest must have, or ``None`` to skip that check.
        hashes: Whether to hash every file; without, only the set of files and
            their sizes are checked.

    Returns:
        The manifest.

    Raises:
        StalePackageError: If the manifest's hash differs, a file is missing or
            extra, or a file's size or hash differs from the manifest.
    """
    manifest, digest = read_manifest(directory)
    problems = []
    if manifest_sha256 is not None and digest != manifest_sha256:
        problems.append(f"the manifest's hash is {digest}, not the {manifest_sha256} pinned")
    present = set(package_files(directory))
    expected = set(manifest.files)
    problems += [f"{path} is missing" for path in sorted(expected - present)]
    problems += [f"{path} is not in the manifest" for path in sorted(present - expected)]
    for path in sorted(expected & present):
        entry = manifest.files[path]
        if (directory / path).stat().st_size != entry.bytes:
            problems.append(f"{path} has another size")
        elif hashes and hash_file(directory / path) != entry.sha256:
            problems.append(f"{path} has another hash")
    if problems:
        raise StalePackageError(
            f"The package in {directory} is not its manifest's: " + "; ".join(problems)
        )
    return manifest


class SpeechTable:
    """The speeches' fields, and their paragraphs read from the store's text.

    Attributes:
        rows: How many speeches the table holds.
    """

    def __init__(self, directory: Path, texts: StringBlob) -> None:
        self._table = pq.read_table(directory / SPEECHES_NAME)
        self._owner = np.load(directory / SPEECH_ROWS_NAME, mmap_mode="r")
        self._texts = texts
        self.rows = self._table.num_rows
        if len(self._owner) != len(texts):
            raise StalePackageError("The speeches' table does not cover the store's passages.")

    def row_of(self, position: int) -> int:
        """The row of the speech a passage belongs to."""
        return int(self._owner[position])

    def fields(self, row: int) -> dict[str, Any]:
        """The fields of one speech, as ``anforanden.parquet`` gives them."""
        return self._table.slice(row, 1).to_pylist()[0]

    def speech(self, row: int) -> Speech:
        """One speech, with its paragraphs read from its passages' text."""
        fields = self.fields(row)
        first = fields["first_passage"]
        paragraphs = tuple(
            paragraph
            for position in range(first, first + fields["passages"])
            for paragraph in self._texts[position].split("\n")
        )
        return Speech(
            anf_id=fields["anf_id"],
            party=fields["party"],
            speaker=fields["speaker_raw"],
            date=fields["date"],
            title=fields["title"],
            url=fields["url"],
            paragraphs=paragraphs,
            text_sha256=fields["text_sha256"],
        )
