"""Turning passages and questions into vectors, and keeping the vectors on disk in shards.

An embedder turns texts into unit vectors, one row per text. Three kinds stand
behind one interface: a deterministic embedder for the tests, which hashes words
into a vector and needs no model; a local model run through sentence-transformers;
and a model behind OpenRouter's API. The heavy libraries are imported only when
a local embedder is built, so the package and its tests run without them.

What a run embeds is an *input set*: a directory holding the passages' texts and
the questions' texts, with a manifest of their hashes. An input set can be
copied to another machine and embedded there.

A run embeds the passages in shards of a fixed number of rows. Each shard is
written beside its destination and moved into place once whole, and the run's
manifest records its hash after every shard. A run that stops can therefore be
started again, and it goes on from the first shard it lacks. A manifest written
by another embedder, or from other inputs, stops the run.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from search.config import CandidateSettings
from search.index.analyzer import STEMMING
from search.index.lexical import file_sha256

Role = Literal["query", "document"]
EmbedderKindName = Literal["hash", "sentence_transformers", "openrouter"]

INPUTS_MANIFEST = "inputs.json"
INPUT_PASSAGES = "passages.parquet"
INPUT_QUERIES = "queries.parquet"
RUN_MANIFEST = "manifest.json"
QUERY_VECTORS = "queries.npy"

#: How far from 1 the length of a vector may be before it counts as not normalised.
NORM_TOLERANCE = 1e-3

#: The software whose versions a run records, when it is installed.
RECORDED_PACKAGES = ("numpy", "torch", "transformers", "sentence-transformers", "httpx")

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_KEY = "OPENROUTER_API_KEY"

#: Status codes after which a request to the API is sent again.
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})


def now() -> datetime:
    """The current time in UTC, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def ids_sha256(ids: Sequence[str]) -> str:
    """Hash ids in order, one per line, as every manifest here records them."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


class _Frozen(BaseModel):
    """Base for the records here: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbedderSpec(_Frozen):
    """What produced a set of vectors, as every run's manifest records it.

    Attributes:
        name: The candidate's name, or ``hash`` for the test embedder.
        kind: How the model runs.
        model: The model's id.
        revision: The commit of its weights.
        dimension: The size of the vectors.
        query_prompt: The text put before a question.
        document_prompt: The text put before a passage.
        max_tokens: The most tokens of a passage a local model reads.
        provider: The provider an API model is routed to.
        quantization: The precision that provider serves the model in.
    """

    name: str
    kind: EmbedderKindName
    model: str
    revision: str
    dimension: int = Field(gt=0)
    query_prompt: str = ""
    document_prompt: str = ""
    max_tokens: int | None = None
    provider: str | None = None
    quantization: str | None = None

    @classmethod
    def from_candidate(cls, candidate: CandidateSettings) -> EmbedderSpec:
        """The spec of a candidate from the settings."""
        return cls(
            name=candidate.name,
            kind=candidate.kind,
            model=candidate.model,
            revision=candidate.revision,
            dimension=candidate.dimension,
            query_prompt=candidate.query_prompt,
            document_prompt=candidate.document_prompt,
            max_tokens=candidate.max_tokens,
            provider=candidate.provider,
            quantization=candidate.quantization,
        )

    def prompt(self, role: Role) -> str:
        """The text put before a text of *role*."""
        return self.query_prompt if role == "query" else self.document_prompt


class Usage(_Frozen):
    """What a run has used so far.

    Attributes:
        texts: How many texts were embedded.
        truncated: How many of them were longer than the model reads.
        requests: How many requests were sent to an API, retries included.
        tokens: How many input tokens an API counted.
        cost_usd: What an API charged, as it reported it.
    """

    texts: int = 0
    truncated: int = 0
    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def plus(self, other: Usage) -> Usage:
        """The sum of two accounts."""
        return Usage(
            texts=self.texts + other.texts,
            truncated=self.truncated + other.truncated,
            requests=self.requests + other.requests,
            tokens=self.tokens + other.tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 9),
        )


class Embedder(Protocol):
    """Turns texts into unit vectors."""

    spec: EmbedderSpec

    def embed(self, texts: Sequence[str], role: Role) -> np.ndarray:
        """Embed *texts* as questions or as passages.

        Args:
            texts: The texts, without any prompt; the embedder adds its own.
            role: ``query`` for questions, ``document`` for passages.

        Returns:
            One row of ``float32`` per text, each of length 1.
        """
        ...

    def take_usage(self) -> Usage:
        """Return what was used since the last call, and start a new account."""
        ...


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1.

    Raises:
        ValueError: If a row is zero or holds a value that is not finite.
    """
    data = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("A vector holds a value that is not finite.")
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("A vector is zero and has no direction.")
    return (data / norms).astype(np.float32)


class HashEmbedder:
    """A deterministic embedder for the tests: words hashed into a fixed number of dimensions.

    A text's terms are those of the stemming analyzer. Each distinct term adds
    ``1 + ln(count)`` at one dimension, with a sign, both taken from the term's
    BLAKE2b hash, so the vector is the same on every machine. Two texts that
    share terms point the same way, which is all the tests ask of it. A text with
    no term gets the first dimension, so that every vector has a direction.
    Prompts are ignored.
    """

    def __init__(self, dimension: int = 64, name: str = "hash") -> None:
        self.spec = EmbedderSpec(
            name=name, kind="hash", model="hash", revision="1", dimension=dimension
        )
        self._usage = Usage()

    def embed(self, texts: Sequence[str], role: Role) -> np.ndarray:
        """Embed *texts*; *role* makes no difference to this embedder."""
        rows = np.zeros((len(texts), self.spec.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(STEMMING.terms(text)).items():
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                column = int.from_bytes(digest[:4], "little") % self.spec.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                rows[row, column] += sign * (1.0 + math.log(count))
            if not rows[row].any():
                rows[row, 0] = 1.0
        self._usage = self._usage.plus(Usage(texts=len(texts)))
        return normalise(rows)

    def take_usage(self) -> Usage:
        """Return what was used since the last call, and start a new account."""
        used, self._usage = self._usage, Usage()
        return used


def _import(module: str) -> Any:
    """Import a heavy dependency, naming the file that installs it when it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{module} is not installed. Install requirements-embed.txt to run a local model."
        ) from error


class SentenceTransformerEmbedder:
    """A local model run through sentence-transformers, at a pinned revision.

    The model reads at most ``spec.max_tokens`` tokens of a text, or its own
    configured limit where that is lower. Texts longer than that, prompt
    included, are counted as truncated.
    """

    def __init__(
        self,
        spec: EmbedderSpec,
        device: str | None = None,
        batch_size: int = 32,
        precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    ) -> None:
        if spec.kind != "sentence_transformers" or spec.max_tokens is None:
            raise ValueError(f"{spec.name} is not a local model with a token limit.")
        library = _import("sentence_transformers")
        self.spec = spec
        self.batch_size = batch_size
        self.precision = precision
        self.model = library.SentenceTransformer(
            spec.model, revision=spec.revision, device=device, trust_remote_code=False
        )
        configured = self.model.max_seq_length or spec.max_tokens
        self.model.max_seq_length = min(spec.max_tokens, configured)
        if precision == "fp16":
            self.model.half()
        elif precision == "bf16":
            self.model.bfloat16()
        dimension_of = getattr(self.model, "get_embedding_dimension", None)
        produced = (dimension_of or self.model.get_sentence_embedding_dimension)()
        if produced != spec.dimension:
            raise ValueError(f"{spec.model} gives {produced} dimensions, not {spec.dimension}.")
        self.device = str(self.model.device)
        self._usage = Usage()

    @property
    def max_tokens(self) -> int:
        """The most tokens the model reads of a text."""
        return int(self.model.max_seq_length)

    def _count_truncated(self, texts: Sequence[str]) -> int:
        encoded = self.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return sum(1 for ids in encoded["input_ids"] if len(ids) > self.max_tokens)

    def embed(self, texts: Sequence[str], role: Role) -> np.ndarray:
        """Embed *texts* with the prompt of *role* before each.

        The prompt is passed even when it is empty, so that no default prompt of
        the model's own configuration is added instead.
        """
        prompt = self.spec.prompt(role)
        truncated = self._count_truncated([prompt + text for text in texts])
        vectors = self.model.encode(
            list(texts),
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._usage = self._usage.plus(Usage(texts=len(texts), truncated=truncated))
        return normalise(vectors)

    def take_usage(self) -> Usage:
        """Return what was used since the last call, and start a new account."""
        used, self._usage = self._usage, Usage()
        return used


class BudgetReached(RuntimeError):
    """A run through the API reached the most it may spend."""


class ApiError(RuntimeError):
    """The API refused a request, or answered with something that is not embeddings."""


def load_openrouter_key(env_file: Path) -> str:
    """Read the OpenRouter key from the environment, or else from a ``.env`` file.

    The key is never printed. Quotes around the value are removed.

    Args:
        env_file: The ``.env`` file to read when the environment has no key.

    Returns:
        The key.

    Raises:
        SystemExit: If neither holds it.
    """
    key = os.environ.get(OPENROUTER_KEY)
    if key:
        return key
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == OPENROUTER_KEY and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit(f"{OPENROUTER_KEY} is neither in the environment nor in {env_file}.")


@dataclass
class _Account:
    """The running account of one call to the API embedder."""

    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    providers: set[str] = field(default_factory=set)


class OpenRouterEmbedder:
    """A model behind OpenRouter's embeddings API, routed to one provider in one precision.

    The request asks for ``spec.dimension`` dimensions. A vector that comes back
    longer is cut to its first ``spec.dimension`` values, which a model trained
    with Matryoshka representation learning allows; every vector is normalised
    again either way. Fallback to other providers is switched off, so a request
    the chosen provider cannot serve fails instead of being served elsewhere.

    Requests run concurrently, a few at a time, and are sent again after a rate
    limit or a server error, honouring ``Retry-After``. No request starts once
    the run has spent its cap; the cost is what OpenRouter reports, or the
    listed price times the tokens where that is higher.
    """

    def __init__(
        self,
        spec: EmbedderSpec,
        key: str,
        price_per_million_usd: float,
        cap_usd: float,
        batch_inputs: int = 64,
        concurrency: int = 4,
        max_attempts: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
        url: str = OPENROUTER_URL,
        backoff_seconds: float = 2.0,
    ) -> None:
        if spec.kind != "openrouter" or spec.provider is None or spec.quantization is None:
            raise ValueError(f"{spec.name} is not an API model with a provider and a precision.")
        self.spec = spec
        self._key = key
        self.price = price_per_million_usd
        self.cap_usd = cap_usd
        self.batch_inputs = batch_inputs
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.transport = transport
        self.url = url
        self.backoff_seconds = backoff_seconds
        self.spent_usd = 0.0
        self.server_side_dimensions = 0
        self.client_side_dimensions = 0
        self.providers: set[str] = set()
        self._usage = Usage()

    def _body(self, inputs: Sequence[str]) -> dict[str, Any]:
        return {
            "model": self.spec.model,
            "input": list(inputs),
            "encoding_format": "base64",
            "dimensions": self.spec.dimension,
            "provider": {
                "order": [self.spec.provider],
                "allow_fallbacks": False,
                "quantizations": [self.spec.quantization],
            },
        }

    def _decode(self, payload: Mapping[str, Any], expected: int) -> np.ndarray:
        """Turn one response into vectors, in the order of the request's inputs."""
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise ApiError(f"The API returned {len(data or [])} embeddings for {expected} inputs.")
        rows: list[np.ndarray | None] = [None] * expected
        for item in data:
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < expected or rows[index] is not None:
                raise ApiError("The API returned embeddings out of order or twice.")
            raw = item.get("embedding")
            if isinstance(raw, str):
                vector = np.frombuffer(base64.b64decode(raw), dtype="<f4")
            else:
                vector = np.asarray(raw, dtype=np.float32)
            if vector.ndim != 1 or len(vector) < self.spec.dimension:
                raise ApiError(f"The API returned a vector of {vector.size} values.")
            if len(vector) > self.spec.dimension:
                self.client_side_dimensions += 1
            else:
                self.server_side_dimensions += 1
            rows[index] = vector[: self.spec.dimension]
        return normalise(np.stack([row for row in rows if row is not None]))

    def _charge(self, payload: Mapping[str, Any], account: _Account) -> None:
        usage = payload.get("usage") or {}
        tokens = int(usage.get("prompt_tokens") or 0)
        listed = tokens * self.price / 1_000_000
        cost = max(float(usage.get("cost") or 0.0), listed)
        account.tokens += tokens
        account.cost_usd += cost
        self.spent_usd += cost
        provider = payload.get("provider")
        if isinstance(provider, str):
            account.providers.add(provider)

    async def _send(
        self, client: httpx.AsyncClient, inputs: Sequence[str], account: _Account
    ) -> np.ndarray:
        for attempt in range(1, self.max_attempts + 1):
            if self.spent_usd >= self.cap_usd:
                raise BudgetReached(
                    f"The run has spent {self.spent_usd:.4f} USD of its cap of {self.cap_usd:.2f}."
                )
            account.requests += 1
            try:
                response = await client.post(self.url, json=self._body(inputs))
            except httpx.TransportError as error:
                if attempt == self.max_attempts:
                    raise ApiError(f"The API could not be reached: {error}") from error
                await asyncio.sleep(self.backoff_seconds * 2 ** (attempt - 1))
                continue
            if response.status_code == 200:
                payload = response.json()
                if "error" in payload:
                    raise ApiError(f"The API answered with an error: {payload['error']}")
                self._charge(payload, account)
                return self._decode(payload, len(inputs))
            if response.status_code in RETRY_STATUS and attempt < self.max_attempts:
                wait = response.headers.get("retry-after")
                delay = float(wait) if wait and wait.replace(".", "", 1).isdigit() else None
                await asyncio.sleep(delay or self.backoff_seconds * 2 ** (attempt - 1))
                continue
            raise ApiError(f"The API answered {response.status_code}: {response.text[:300]}")
        raise ApiError("The API did not answer within the attempts allowed.")

    async def _embed_all(self, batches: Sequence[Sequence[str]]) -> tuple[np.ndarray, _Account]:
        account = _Account()
        gate = asyncio.Semaphore(self.concurrency)
        headers = {"Authorization": f"Bearer {self._key}"}
        async with httpx.AsyncClient(
            transport=self.transport, headers=headers, timeout=120.0
        ) as client:

            async def one(batch: Sequence[str]) -> np.ndarray:
                async with gate:
                    return await self._send(client, batch, account)

            parts = await asyncio.gather(*(one(batch) for batch in batches))
        return np.concatenate(parts, axis=0), account

    def embed(self, texts: Sequence[str], role: Role) -> np.ndarray:
        """Embed *texts* with the prompt of *role* before each."""
        prompt = self.spec.prompt(role)
        inputs = [prompt + text for text in texts]
        batches = [
            inputs[start : start + self.batch_inputs]
            for start in range(0, len(inputs), self.batch_inputs)
        ]
        vectors, account = asyncio.run(self._embed_all(batches))
        self.providers |= account.providers
        self._usage = self._usage.plus(
            Usage(
                texts=len(texts),
                requests=account.requests,
                tokens=account.tokens,
                cost_usd=account.cost_usd,
            )
        )
        return vectors

    def take_usage(self) -> Usage:
        """Return what was used since the last call, and start a new account."""
        used, self._usage = self._usage, Usage()
        return used


def build_embedder(
    candidate: CandidateSettings,
    *,
    device: str | None = None,
    batch_size: int = 32,
    precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    key: str | None = None,
    cap_usd: float = 1.0,
    batch_inputs: int = 64,
    concurrency: int = 4,
) -> Embedder:
    """Build the embedder of a candidate from the settings.

    Args:
        candidate: The candidate.
        device: Where a local model runs: ``cpu``, ``mps``, ``cuda``, or ``None``
            to let sentence-transformers choose.
        batch_size: How many texts a local model embeds at a time.
        precision: The floating-point precision a local model runs in.
        key: The OpenRouter key, for an API model.
        cap_usd: The most a run through the API may spend.
        batch_inputs: How many texts one request to the API carries.
        concurrency: How many requests to the API are in flight at once.

    Returns:
        The embedder.

    Raises:
        ValueError: If an API model is asked for without a key.
    """
    spec = EmbedderSpec.from_candidate(candidate)
    if candidate.kind == "sentence_transformers":
        return SentenceTransformerEmbedder(spec, device, batch_size, precision)
    if key is None or candidate.price_per_million_usd is None:
        raise ValueError(f"{candidate.name} runs through the API and needs a key and a price.")
    return OpenRouterEmbedder(
        spec,
        key,
        candidate.price_per_million_usd,
        cap_usd,
        batch_inputs=batch_inputs,
        concurrency=concurrency,
    )


class InputsManifest(_Frozen):
    """What an input set holds, and where it came from.

    Attributes:
        name: The set's name, for example ``subset-200``.
        target_words: The size its passages were packed towards.
        passages: How many passages it holds.
        passage_ids_sha256: The hash of their ids, in order.
        positions_sha256: The hash of their positions in the passages file.
        texts_sha256: The hash of their texts, in order.
        queries: How many questions it holds.
        query_ids_sha256: The hash of the questions' ids, in order.
        source: The hashes of the files the set was made from, by file name.
        created: When the set was written.
    """

    name: str
    target_words: int
    passages: int
    passage_ids_sha256: str
    positions_sha256: str
    texts_sha256: str
    queries: int
    query_ids_sha256: str
    source: dict[str, str]
    created: datetime


@dataclass(frozen=True)
class EmbeddingInputs:
    """An input set read into memory.

    Attributes:
        manifest: What it holds.
        positions: Each passage's position in the passages file it came from.
        passage_ids: Each passage's id.
        texts: Each passage's text.
        query_ids: Each question's id.
        query_sets: Each question's set.
        query_texts: Each question's text.
    """

    manifest: InputsManifest
    positions: np.ndarray
    passage_ids: list[str]
    texts: list[str]
    query_ids: list[str]
    query_sets: list[str]
    query_texts: list[str]


def _texts_sha256(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def write_inputs(
    directory: Path,
    name: str,
    target_words: int,
    passages: Sequence[tuple[int, str, str]],
    queries: Sequence[tuple[str, str, str]],
    source: Mapping[str, str],
) -> InputsManifest:
    """Write an input set to *directory*.

    Args:
        directory: Where to write; created if missing.
        name: The set's name.
        target_words: The size its passages were packed towards.
        passages: ``(position, passage_id, text)`` for each passage, in position order.
        queries: ``(question_id, question_set, text)`` for each question.
        source: The hashes of the files the set was made from.

    Returns:
        The set's manifest.

    Raises:
        ValueError: If the passages are not in ascending position order or an id repeats.
    """
    positions = [position for position, _, _ in passages]
    if any(later <= earlier for earlier, later in zip(positions, positions[1:], strict=False)):
        raise ValueError("The passages of an input set must be in ascending position order.")
    passage_ids = [passage for _, passage, _ in passages]
    query_ids = [question for question, _, _ in queries]
    if len(set(passage_ids)) != len(passage_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("An id occurs twice in the input set.")
    directory.mkdir(parents=True, exist_ok=True)
    _write_table(
        directory / INPUT_PASSAGES,
        pa.table(
            {
                "position": pa.array(positions, type=pa.int64()),
                "passage_id": pa.array(passage_ids, type=pa.string()),
                "text": pa.array([text for _, _, text in passages], type=pa.string()),
            }
        ),
    )
    _write_table(
        directory / INPUT_QUERIES,
        pa.table(
            {
                "question_id": pa.array(query_ids, type=pa.string()),
                "question_set": pa.array([kind for _, kind, _ in queries], type=pa.string()),
                "text": pa.array([text for _, _, text in queries], type=pa.string()),
            }
        ),
    )
    manifest = InputsManifest(
        name=name,
        target_words=target_words,
        passages=len(passages),
        passage_ids_sha256=ids_sha256(passage_ids),
        positions_sha256=ids_sha256([str(position) for position in positions]),
        texts_sha256=_texts_sha256(text for _, _, text in passages),
        queries=len(queries),
        query_ids_sha256=ids_sha256(query_ids),
        source=dict(source),
        created=now(),
    )
    write_json(directory / INPUTS_MANIFEST, manifest)
    return manifest


def read_inputs(directory: Path) -> EmbeddingInputs:
    """Read an input set written by :func:`write_inputs`, checking it against its manifest.

    Raises:
        ValueError: If the files do not match the manifest.
    """
    manifest = InputsManifest.model_validate_json(
        (directory / INPUTS_MANIFEST).read_text(encoding="utf-8")
    )
    passages = pq.read_table(directory / INPUT_PASSAGES)
    queries = pq.read_table(directory / INPUT_QUERIES)
    inputs = EmbeddingInputs(
        manifest=manifest,
        positions=np.asarray(passages.column("position").to_pylist(), dtype=np.int64),
        passage_ids=passages.column("passage_id").to_pylist(),
        texts=passages.column("text").to_pylist(),
        query_ids=queries.column("question_id").to_pylist(),
        query_sets=queries.column("question_set").to_pylist(),
        query_texts=queries.column("text").to_pylist(),
    )
    checks = (
        (len(inputs.passage_ids), manifest.passages),
        (ids_sha256(inputs.passage_ids), manifest.passage_ids_sha256),
        (ids_sha256([str(int(p)) for p in inputs.positions]), manifest.positions_sha256),
        (_texts_sha256(inputs.texts), manifest.texts_sha256),
        (len(inputs.query_ids), manifest.queries),
        (ids_sha256(inputs.query_ids), manifest.query_ids_sha256),
    )
    if any(actual != recorded for actual, recorded in checks):
        raise ValueError(f"The input set in {directory} does not match its manifest.")
    return inputs


class ShardRecord(_Frozen):
    """One shard of a run.

    Attributes:
        number: The shard's number, from 0.
        first: The row of its first passage within the input set.
        rows: How many passages it holds.
        sha256: The hash of its file.
    """

    number: int
    first: int
    rows: int
    sha256: str


class QueriesRecord(_Frozen):
    """The questions' vectors of a run.

    Attributes:
        rows: How many questions.
        query_ids_sha256: The hash of their ids, in order.
        sha256: The hash of the file.
    """

    rows: int
    query_ids_sha256: str
    sha256: str


class RunManifest(_Frozen):
    """What a run embedded, with what, and how far it has come.

    Attributes:
        embedder: What produced the vectors.
        inputs: The input set's manifest.
        shard_rows: How many passages a shard holds; the last may hold fewer.
        shards: The shards written so far, in order.
        queries: The questions' vectors, once written.
        usage: What the run has used.
        effective_max_tokens: The most tokens the model read of a text.
        device: Where a local model ran.
        precision: The floating-point precision it ran in.
        providers: The providers an API reported serving the requests.
        dimensions_cut_client_side: How many vectors an API returned longer than
            asked, which the run cut.
        software: The versions of the software that ran.
        started: When the run started.
        updated: When the manifest was last written.
    """

    embedder: EmbedderSpec
    inputs: InputsManifest
    shard_rows: int = Field(gt=0)
    shards: tuple[ShardRecord, ...] = ()
    queries: QueriesRecord | None = None
    usage: Usage = Usage()
    effective_max_tokens: int | None = None
    device: str | None = None
    precision: str | None = None
    providers: tuple[str, ...] = ()
    dimensions_cut_client_side: int = 0
    software: dict[str, str] = {}
    started: datetime
    updated: datetime

    @property
    def expected_shards(self) -> int:
        """How many shards the run has when it is whole."""
        return math.ceil(self.inputs.passages / self.shard_rows)

    @property
    def complete(self) -> bool:
        """Whether every shard and the questions' vectors are written."""
        return len(self.shards) == self.expected_shards and self.queries is not None


def _write_table(path: Path, table: pa.Table) -> None:
    partial = path.with_name(path.name + ".part")
    try:
        pq.write_table(table, partial, compression="zstd")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def write_json(path: Path, record: BaseModel) -> None:
    """Write a record as indented JSON beside *path*, then move it into place."""
    partial = path.with_name(path.name + ".part")
    partial.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    partial.replace(path)


def save_array(path: Path, array: np.ndarray) -> str:
    """Save an array beside *path*, move it into place, and return the file's hash.

    Whatever stops the write, the partial file is removed and a file already at
    *path* is left as it was.
    """
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def shard_path(run_dir: Path, number: int) -> Path:
    """The file of one shard of a run."""
    return run_dir / f"shard-{number:05d}.npy"


def software_versions() -> dict[str, str]:
    """The versions of the recorded packages that are installed."""
    found = {}
    for name in RECORDED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _check_vectors(vectors: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    data = np.asarray(vectors, dtype=np.float32)
    if data.shape != (rows, dimension):
        raise ValueError(f"The embedder returned {data.shape}, not ({rows}, {dimension}).")
    if not np.isfinite(data).all():
        raise ValueError("The embedder returned a value that is not finite.")
    norms = np.linalg.norm(data, axis=1)
    if np.abs(norms - 1).max(initial=0.0) > NORM_TOLERANCE:
        raise ValueError("The embedder returned vectors that are not of length 1.")
    return data


def read_run_manifest(run_dir: Path) -> RunManifest:
    """Read the manifest of a run."""
    return RunManifest.model_validate_json((run_dir / RUN_MANIFEST).read_text(encoding="utf-8"))


def _resume(
    run_dir: Path, embedder: Embedder, inputs: EmbeddingInputs, shard_rows: int
) -> RunManifest:
    """The manifest to go on from: the existing one, checked, or a new one."""
    path = run_dir / RUN_MANIFEST
    if not path.is_file():
        return RunManifest(
            embedder=embedder.spec,
            inputs=inputs.manifest,
            shard_rows=shard_rows,
            started=now(),
            updated=now(),
        )
    manifest = read_run_manifest(run_dir)
    if manifest.embedder != embedder.spec:
        raise ValueError(f"The run in {run_dir} was made by another embedder.")
    if manifest.inputs.passage_ids_sha256 != inputs.manifest.passage_ids_sha256 or (
        manifest.inputs.texts_sha256 != inputs.manifest.texts_sha256
    ):
        raise ValueError(f"The run in {run_dir} was made from other inputs.")
    if manifest.shard_rows != shard_rows:
        raise ValueError(f"The run in {run_dir} uses shards of {manifest.shard_rows} rows.")
    for record in manifest.shards:
        file = shard_path(run_dir, record.number)
        if not file.is_file() or file_sha256(file) != record.sha256:
            raise ValueError(f"Shard {record.number} of {run_dir} no longer matches its manifest.")
    return manifest


def embed_inputs(
    inputs: EmbeddingInputs,
    embedder: Embedder,
    run_dir: Path,
    shard_rows: int,
    max_shards: int | None = None,
    progress: Any = None,
) -> RunManifest:
    """Embed the passages of an input set shard by shard, then its questions.

    A run that stopped goes on from the first shard it lacks.

    Args:
        inputs: The input set.
        embedder: The embedder.
        run_dir: The run's directory; created if missing.
        shard_rows: How many passages a shard holds.
        max_shards: Stop after writing this many new shards, for a short trial run.
        progress: Called with the manifest after every shard, if given.

    Returns:
        The manifest as it stands when the run stops.

    Raises:
        ValueError: If the directory holds a run by another embedder or from other
            inputs, a recorded shard no longer matches its hash, or the embedder
            returns vectors of the wrong shape or length.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resume(run_dir, embedder, inputs, shard_rows)
    dimension = embedder.spec.dimension
    written = 0
    while len(manifest.shards) < manifest.expected_shards:
        if max_shards is not None and written >= max_shards:
            return manifest
        number = len(manifest.shards)
        first = number * shard_rows
        texts = inputs.texts[first : first + shard_rows]
        vectors = _check_vectors(embedder.embed(texts, "document"), len(texts), dimension)
        sha256 = save_array(shard_path(run_dir, number), vectors)
        manifest = _updated(
            manifest,
            embedder,
            shards=(
                *manifest.shards,
                ShardRecord(number=number, first=first, rows=len(texts), sha256=sha256),
            ),
        )
        write_json(run_dir / RUN_MANIFEST, manifest)
        written += 1
        if progress is not None:
            progress(manifest)
    if manifest.queries is None:
        vectors = _check_vectors(
            embedder.embed(inputs.query_texts, "query"), len(inputs.query_texts), dimension
        )
        sha256 = save_array(run_dir / QUERY_VECTORS, vectors)
        record = QueriesRecord(
            rows=len(inputs.query_ids),
            query_ids_sha256=inputs.manifest.query_ids_sha256,
            sha256=sha256,
        )
        manifest = _updated(manifest, embedder, queries=record)
        write_json(run_dir / RUN_MANIFEST, manifest)
    return manifest


def _updated(manifest: RunManifest, embedder: Embedder, **changes: Any) -> RunManifest:
    """The manifest with the embedder's latest usage and state folded in, and *changes* applied."""
    usage = manifest.usage.plus(embedder.take_usage())
    state: dict[str, Any] = {"usage": usage, "updated": now(), "software": software_versions()}
    if isinstance(embedder, SentenceTransformerEmbedder):
        state.update(
            effective_max_tokens=embedder.max_tokens,
            device=embedder.device,
            precision=embedder.precision,
        )
    if isinstance(embedder, OpenRouterEmbedder):
        state.update(
            providers=tuple(sorted(set(manifest.providers) | embedder.providers)),
            dimensions_cut_client_side=embedder.client_side_dimensions,
        )
    return manifest.model_copy(update={**state, **changes})


def read_shards(run_dir: Path, manifest: RunManifest) -> np.ndarray:
    """Read every shard of a whole run into one matrix, checking each against its hash.

    Raises:
        ValueError: If the run is not whole or a shard does not match its record.
    """
    if not manifest.complete:
        raise ValueError(f"The run in {run_dir} is not complete.")
    parts = []
    for record in manifest.shards:
        file = shard_path(run_dir, record.number)
        if file_sha256(file) != record.sha256:
            raise ValueError(f"Shard {record.number} of {run_dir} does not match its manifest.")
        part = np.load(file, allow_pickle=False)
        if part.shape != (record.rows, manifest.embedder.dimension):
            raise ValueError(f"Shard {record.number} of {run_dir} has the shape {part.shape}.")
        parts.append(part)
    return np.concatenate(parts, axis=0)


def read_query_vectors(run_dir: Path, manifest: RunManifest) -> np.ndarray:
    """Read a run's question vectors, checking them against the manifest.

    Raises:
        ValueError: If they are missing or do not match their record.
    """
    record = manifest.queries
    file = run_dir / QUERY_VECTORS
    if record is None or not file.is_file() or file_sha256(file) != record.sha256:
        raise ValueError(f"The question vectors of {run_dir} are missing or do not match.")
    vectors = np.load(file, allow_pickle=False)
    if vectors.shape != (record.rows, manifest.embedder.dimension):
        raise ValueError(f"The question vectors of {run_dir} have the shape {vectors.shape}.")
    return vectors


def manifest_json(manifest: BaseModel) -> str:
    """The manifest as the JSON written to disk, for hashing or display."""
    return json.dumps(json.loads(manifest.model_dump_json()), indent=2, ensure_ascii=False)
