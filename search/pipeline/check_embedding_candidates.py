"""Check the embedding candidates against the Scandinavian Embedding Benchmark, and price them.

The benchmark is ``MTEB(Scandinavian, v1)`` in MTEB, and its results are read
from MTEB's results dataset at the pinned revision of its Parquet export, each
file checked against its hash. Of the benchmark's tasks, the seven retrieval
tasks are taken. Every model with results on both Swedish ones is ranked by
their mean, with the mean of all seven beside it where it has them. A model
with results at several revisions is scored at the revision with results on
most of the seven tasks, the later revision name on a tie.

For each candidate, the report compares the files of the revision the
settings pin with those of the revision MTEB evaluated, by the hashes Hugging
Face lists for them.

For the API models the addendum of 2026-11-21 weighs, the corpus's passages are
counted with each model's published tokenizer, at a pinned revision, as the
API would receive them, and priced at the prices listed on 2026-11-21. Step 13
reads the API candidate's count from the report; nothing else reads it.

Usage::

    python search/pipeline/check_embedding_candidates.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report
from search.corpus.tokens import token_lengths
from search.index.lexical import file_sha256, passage_texts

logger = logging.getLogger("candidates")

REPORT_NAME = "embedding_candidates.json"

BENCHMARK = "MTEB(Scandinavian, v1)"
RESULTS_REVISION = "dfe159b89b95b84787cd2a896e578991820545f4"
PARQUET_REVISION = "d9ec6baf92aaa634a81d7ac0d916bad2864e42a0"
PARQUET_URL = "https://huggingface.co/datasets/mteb/results/resolve/{revision}/default/train/{name}"
PARQUET_FILES = {
    "0000.parquet": "c994ad1d6b175245129221a7cdef8c282fd340532ac8e99a0fcb87cd2c309685",
    "0001.parquet": "8913ad6cba8068c746a5286d483f24aca1580c8eb6f6364390180f280e059807",
    "0002.parquet": "ff725ce5f8afa3ad245c634f48bf40f4147c08c1289862d4e278627d811fd199",
    "0003.parquet": "354fcf124203412bea64a5708f4da1aec585259e34853edf3335ef2b405c0d84",
}

#: The benchmark's retrieval tasks, as MTEB's definition of it lists them.
RETRIEVAL_TASKS = (
    "DanFeverRetrieval",
    "NorQuadRetrieval",
    "SNLRetrieval",
    "SwednRetrieval",
    "SweFaqRetrieval",
    "TV2Nordretrieval",
    "TwitterHjerneRetrieval",
)
SWEDISH_TASKS = ("SwednRetrieval", "SweFaqRetrieval")

#: The name each candidate has in MTEB's results. saga-embed_v1 was published
#: as nicher92/saga-embed_v1, which now redirects to SagaEmbed.
MTEB_NAMES = {
    "bekko-a25m": "hotchpotch/bekko-embedding-v1-a25m",
    "saga-embed": "nicher92/saga-embed_v1",
    "arctic-l-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
}

#: The Hugging Face repository of a candidate's published weights, where its
#: model id names something else: the API candidate's id is OpenRouter's.
WEIGHTS_REPOS = {"qwen3-8b": "Qwen/Qwen3-Embedding-8B"}

PRICES_LISTED_ON = date(2026, 11, 21)
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/embeddings/models"
GOOGLE_PRICES = "https://aistudio.google.com/docs/pricing"
VOYAGE_PRICES = "https://docs.voyageai.com/docs/pricing"

#: The batch discount Voyage lists, a third off the standard price.
VOYAGE_BATCH_DISCOUNT = 0.33

HF_FILE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"
HF_REVISION = "https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"


class _Frozen(BaseModel):
    """Base for the records here: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ApiModel(_Frozen):
    """An API model the addendum weighs, and where its price comes from.

    Attributes:
        mteb_name: Its name in MTEB's results.
        price_per_million_usd: The listed price of a million input tokens.
        batch_price_per_million_usd: The listed batch price, where there is one.
        tokenizer_repo: The Hugging Face repository of its published tokenizer.
        tokenizer_revision: The revision of that tokenizer.
        price_source: Where the price is listed.
        candidate: The candidate's name in the settings, for the one that is a candidate.
    """

    mteb_name: str
    price_per_million_usd: float
    batch_price_per_million_usd: float | None = None
    tokenizer_repo: str | None = None
    tokenizer_revision: str | None = None
    price_source: str
    candidate: str | None = None


API_MODELS = (
    ApiModel(
        mteb_name="google/gemini-embedding-001",
        price_per_million_usd=0.15,
        batch_price_per_million_usd=0.075,
        price_source=GOOGLE_PRICES,
    ),
    ApiModel(
        mteb_name="mongodb/voyage-finance-2",
        price_per_million_usd=0.12,
        batch_price_per_million_usd=round(0.12 * (1 - VOYAGE_BATCH_DISCOUNT), 6),
        tokenizer_repo="MongoDB/voyage-finance-2",
        tokenizer_revision="5354e8ce3b4b879a476df11ddabf3c4ec8a55571",
        price_source=VOYAGE_PRICES,
    ),
    ApiModel(
        mteb_name="mongodb/voyage-multilingual-2",
        price_per_million_usd=0.12,
        batch_price_per_million_usd=round(0.12 * (1 - VOYAGE_BATCH_DISCOUNT), 6),
        tokenizer_repo="MongoDB/voyage-multilingual-2",
        tokenizer_revision="b1cca9fa964524e1bf88ae532eff6d78b4d9c03f",
        price_source=VOYAGE_PRICES,
    ),
    ApiModel(
        mteb_name="mongodb/voyage-3.5",
        price_per_million_usd=0.06,
        batch_price_per_million_usd=round(0.06 * (1 - VOYAGE_BATCH_DISCOUNT), 6),
        tokenizer_repo="MongoDB/voyage-3.5",
        tokenizer_revision="47c5de6b4f45822930dc3beae47ef9d50ec490ff",
        price_source=VOYAGE_PRICES,
    ),
    ApiModel(
        mteb_name="openai/text-embedding-3-large",
        price_per_million_usd=0.13,
        price_source=OPENROUTER_MODELS,
    ),
    ApiModel(
        mteb_name="mongodb/voyage-3",
        price_per_million_usd=0.06,
        batch_price_per_million_usd=round(0.06 * (1 - VOYAGE_BATCH_DISCOUNT), 6),
        tokenizer_repo="MongoDB/voyage-3",
        tokenizer_revision="5d7d7779cf11f3f2d235a99579bc510536e91cbb",
        price_source=VOYAGE_PRICES,
    ),
    ApiModel(
        mteb_name="Qwen/Qwen3-Embedding-8B",
        price_per_million_usd=0.01,
        tokenizer_repo="Qwen/Qwen3-Embedding-8B",
        tokenizer_revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        price_source=OPENROUTER_MODELS,
        candidate="qwen3-8b",
    ),
)


class BenchmarkRow(_Frozen):
    """One model's retrieval results on the benchmark.

    Attributes:
        model: The model's name in MTEB's results.
        revision: The revision scored.
        swedish_retrieval: The mean of the two Swedish retrieval tasks.
        retrieval: The mean of all seven retrieval tasks, where it has results on all.
        tasks: The main score of each retrieval task it has.
    """

    model: str
    revision: str
    swedish_retrieval: float
    retrieval: float | None
    tasks: dict[str, float]


class RevisionCheck(_Frozen):
    """The revision a candidate is pinned to, against the one MTEB evaluated.

    Attributes:
        candidate: The candidate's name.
        model: The model's repository.
        pinned: The revision the settings pin.
        evaluated: The revision MTEB evaluated.
        same_files: The weight and tokenizer files whose hashes agree.
        differing_files: The files whose hashes differ or that only one revision has.
    """

    candidate: str
    model: str
    pinned: str
    evaluated: str
    same_files: tuple[str, ...]
    differing_files: tuple[str, ...]


class ApiPricing(_Frozen):
    """What embedding the whole corpus would cost with one API model.

    Attributes:
        model: Its name in MTEB's results.
        swedish_retrieval: Its mean on the two Swedish retrieval tasks.
        price_per_million_usd: The listed price of a million input tokens.
        batch_price_per_million_usd: The listed batch price, where there is one.
        tokenizer: The published tokenizer counted with, as ``repo@revision``.
        corpus_tokens: The corpus's passages counted with it.
        corpus_cost_usd: The corpus at the standard price.
        batch_corpus_cost_usd: The corpus at the batch price.
        one_token_per_word_cost_usd: The corpus at the standard price if every
            word were one token, a floor for a model without a published tokenizer.
        price_source: Where the price is listed.
        candidate: The candidate's name, for the one that is a candidate.
    """

    model: str
    swedish_retrieval: float | None
    price_per_million_usd: float
    batch_price_per_million_usd: float | None
    tokenizer: str | None
    corpus_tokens: int | None
    corpus_cost_usd: float | None
    batch_corpus_cost_usd: float | None
    one_token_per_word_cost_usd: float
    price_source: str
    candidate: str | None


class CandidatesReport(_Frozen):
    """The check of the embedding candidates.

    Attributes:
        benchmark: The benchmark's name in MTEB.
        results_revision: The revision of MTEB's results dataset.
        parquet_revision: The revision of its Parquet export that was read.
        parquet_sha256: The hash of each Parquet file.
        retrieval_tasks: The benchmark's retrieval tasks.
        swedish_tasks: The Swedish ones among them.
        ranking: Every model with results on both Swedish tasks, best first.
        revisions: Each local candidate's pinned revision against the evaluated one.
        corpus_passages: The passages counted.
        corpus_words: The words in them.
        api_models: The API models the addendum weighs, priced.
        api_candidates: The API candidates among them, by candidate name.
        prices_listed_on: The day the prices were listed.
    """

    benchmark: str
    results_revision: str
    parquet_revision: str
    parquet_sha256: dict[str, str]
    retrieval_tasks: tuple[str, ...]
    swedish_tasks: tuple[str, ...]
    ranking: tuple[BenchmarkRow, ...]
    revisions: tuple[RevisionCheck, ...]
    corpus_passages: int
    corpus_words: int
    api_models: tuple[ApiPricing, ...]
    api_candidates: dict[str, ApiPricing]
    prices_listed_on: date


def fetch(client: httpx.Client, url: str, path: Path, sha256: str | None = None) -> Path:
    """Download *url* to *path* unless it is there already, and check its hash when one is given."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        with client.stream("GET", url) as response, partial.open("wb") as handle:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                handle.write(chunk)
        partial.replace(path)
    if sha256 is not None and file_sha256(path) != sha256:
        raise SystemExit(f"{path} does not match its pinned hash.")
    return path


def rank_models(paths: list[Path]) -> list[BenchmarkRow]:
    """Every model with results on both Swedish retrieval tasks, ranked by their mean."""
    columns = ["model_name", "model_revision", "task_name", "score"]
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    table = table.filter(pc.is_in(table.column("task_name"), pa.array(RETRIEVAL_TASKS)))
    scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in table.to_pylist():
        scores[(row["model_name"], row["model_revision"])][row["task_name"]].append(row["score"])
    chosen: dict[str, str] = {}
    for model, revision in sorted(scores):
        current = chosen.get(model)
        if current is None or len(scores[(model, revision)]) >= len(scores[(model, current)]):
            chosen[model] = revision
    rows = []
    for model, revision in chosen.items():
        tasks = {
            name: round(fmean(values), 6) for name, values in scores[(model, revision)].items()
        }
        if not all(name in tasks for name in SWEDISH_TASKS):
            continue
        rows.append(
            BenchmarkRow(
                model=model,
                revision=revision,
                swedish_retrieval=round(fmean(tasks[name] for name in SWEDISH_TASKS), 6),
                retrieval=(
                    round(fmean(tasks[name] for name in RETRIEVAL_TASKS), 6)
                    if all(name in tasks for name in RETRIEVAL_TASKS)
                    else None
                ),
                tasks=dict(sorted(tasks.items())),
            )
        )
    rows.sort(key=lambda row: (-row.swedish_retrieval, row.model))
    return rows


def compare_revisions(
    client: httpx.Client, candidate: str, repo: str, pinned: str, evaluated: str
) -> RevisionCheck:
    """Compare the weight and tokenizer files of two revisions by their listed hashes."""

    def listed(revision: str) -> dict[str, str]:
        response = client.get(HF_REVISION.format(repo=repo, revision=revision))
        response.raise_for_status()
        return {
            item["rfilename"]: (item.get("lfs") or {}).get("sha256") or item.get("blobId", "")
            for item in response.json().get("siblings", [])
        }

    first, second = listed(pinned), listed(evaluated)
    names = sorted(set(first) | set(second))
    weights = [
        name
        for name in names
        if name.endswith((".safetensors", "tokenizer.json")) and first.get(name) == second.get(name)
    ]
    return RevisionCheck(
        candidate=candidate,
        model=repo,
        pinned=pinned,
        evaluated=evaluated,
        same_files=tuple(weights),
        differing_files=tuple(name for name in names if first.get(name) != second.get(name)),
    )


def main() -> None:
    """Parse the command line, read the benchmark, count the corpus and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    args = parser.parse_args()
    configure_logging()
    settings: Settings = settings_from(args)
    cache = settings.data.search / "candidate_check"

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        paths = [
            fetch(
                client,
                PARQUET_URL.format(revision=PARQUET_REVISION, name=name),
                cache / "mteb_results" / PARQUET_REVISION / name,
                sha256,
            )
            for name, sha256 in PARQUET_FILES.items()
        ]
        ranking = rank_models(paths)
        logger.info("%d models have results on both Swedish retrieval tasks", len(ranking))
        evaluated = {row.model: row.revision for row in ranking}
        revisions = tuple(
            compare_revisions(
                client,
                candidate.name,
                WEIGHTS_REPOS.get(candidate.name, candidate.model),
                candidate.revision,
                evaluated[MTEB_NAMES[candidate.name]],
            )
            for candidate in settings.vectors.candidates
        )
        tokenizers = {}
        for model in API_MODELS:
            if model.tokenizer_repo is None or model.tokenizer_revision is None:
                continue
            path = fetch(
                client,
                HF_FILE.format(
                    repo=model.tokenizer_repo,
                    revision=model.tokenizer_revision,
                    name="tokenizer.json",
                ),
                cache
                / "tokenizers"
                / model.tokenizer_repo
                / model.tokenizer_revision
                / "tokenizer.json",
            )
            tokenizers[model.mteb_name] = path

    texts = [
        text
        for _, text in passage_texts(
            settings.data.search / SPEECHES_NAME, settings.data.search / PASSAGES_NAME
        )
    ]
    words = sum(len(text.split()) for text in texts)
    scores = {row.model: row.swedish_retrieval for row in ranking}
    priced = []
    for model in API_MODELS:
        tokens = None
        tokenizer_name = None
        if model.mteb_name in tokenizers:
            tokenizer = Tokenizer.from_file(str(tokenizers[model.mteb_name]))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            tokens = sum(token_lengths(texts, tokenizer))
            tokenizer_name = f"{model.tokenizer_repo}@{model.tokenizer_revision}"
            logger.info("%s: %d tokens", model.mteb_name, tokens)
        priced.append(
            ApiPricing(
                model=model.mteb_name,
                swedish_retrieval=scores.get(model.mteb_name),
                price_per_million_usd=model.price_per_million_usd,
                batch_price_per_million_usd=model.batch_price_per_million_usd,
                tokenizer=tokenizer_name,
                corpus_tokens=tokens,
                corpus_cost_usd=(
                    None if tokens is None else round(tokens * model.price_per_million_usd / 1e6, 4)
                ),
                batch_corpus_cost_usd=(
                    None
                    if tokens is None or model.batch_price_per_million_usd is None
                    else round(tokens * model.batch_price_per_million_usd / 1e6, 4)
                ),
                one_token_per_word_cost_usd=round(words * model.price_per_million_usd / 1e6, 4),
                price_source=model.price_source,
                candidate=model.candidate,
            )
        )
    write_report(
        CandidatesReport(
            benchmark=BENCHMARK,
            results_revision=RESULTS_REVISION,
            parquet_revision=PARQUET_REVISION,
            parquet_sha256=PARQUET_FILES,
            retrieval_tasks=RETRIEVAL_TASKS,
            swedish_tasks=SWEDISH_TASKS,
            ranking=tuple(ranking),
            revisions=revisions,
            corpus_passages=len(texts),
            corpus_words=words,
            api_models=tuple(priced),
            api_candidates={p.candidate: p for p in priced if p.candidate is not None},
            prices_listed_on=PRICES_LISTED_ON,
        ),
        settings.tracked.reports / REPORT_NAME,
    )
    logger.info("Report written to %s", settings.tracked.reports / REPORT_NAME)


if __name__ == "__main__":
    main()
