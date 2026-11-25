"""Step 08: pack what the search service searches, with its manifest.

The package is the system the test split measured, so that the page runs it.
It holds the numpy store of ADR-001, rebuilt as the check before the test split
built it (``search_service/spike/service_storage.yaml``), with the BM25 the
tuning chose and the vectors of the hybrid's model, beside the speeches'
fields. Its manifest records how the service searches and answers, the hash of
every file, and the reports its settings come from. Each of those reports must
be committed as it stands, and comparison 1 must have put the hybrid in
production. The manifest is also copied to the tracked reports, byte for byte,
so the package an image bakes in can be checked against git.

Usage::

    python search/pipeline/08_package_index.py [--data-root DIR] [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from search.answer.selector import load_selector_prompt
from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import Settings
from search.eval.hybrid import require_committed, text_sha256
from search_service.package import (
    MANIFEST_NAME,
    AnswerConfig,
    Embedding,
    PackageManifest,
    Retrieval,
    build_package,
    read_manifest,
)
from search_service.spike.inputs import TUNING_NAME, CorpusFiles
from search_service.spike.settings import VectorSettings, load_spike_settings, settings_file

logger = logging.getLogger("package")

#: The settings of the check before the test split, whose store the package rebuilds.
CHECKED_SETTINGS = "service_storage.yaml"
#: The store that check measured.
CHECKED_STORE = "numpy-1024-float32-qwen3-8b"
HYBRID_TUNING = "hybrid_tuning_dev.json"
ANSWER_RUN = "answer_test_run_hybrid.json"
RETRIEVAL_TEST = "retrieval_test.json"
#: Where the packages go, below the search-data directory.
PACKAGES_DIR = "package"
#: The tracked copy of the manifest.
REPORT_NAME = "index_package.json"


def package_dir(settings: Settings) -> Path:
    """The directory of the package the settings name."""
    return settings.data.search / PACKAGES_DIR / settings.service.package


def _committed_json(path: Path) -> dict[str, Any]:
    require_committed(path)
    return json.loads(path.read_text(encoding="utf-8"))


def configuration(
    settings: Settings,
) -> tuple[Retrieval, Embedding, AnswerConfig, dict[str, str], Path, VectorSettings, CorpusFiles]:
    """What the package is built from and records, each checked against the others.

    Returns:
        How the service searches, the question's model, how it answers, the
        hashes of the reports, the vector index, the chunks of the checked
        store, and the corpus files.

    Raises:
        SystemExit: If a report is not committed as it stands, comparison 1 did
            not choose the hybrid, or the settings, the reports and the checked
            store disagree.
    """
    reports = settings.tracked.reports
    decision = _committed_json(reports / RETRIEVAL_TEST)["decision"]
    if decision["outcome"] != "A":
        raise SystemExit("Comparison 1 did not put the hybrid in production; no package is built.")
    hybrid = _committed_json(reports / HYBRID_TUNING)
    run = _committed_json(reports / ANSWER_RUN)
    require_committed(reports / TUNING_NAME)
    files = CorpusFiles.under(settings.data.search, reports / TUNING_NAME)

    name = hybrid["candidate"]
    target_words = int(hybrid["chosen"]["target_words"])
    rrf_k = hybrid["chosen"]["k"]
    candidate = settings.vectors.candidate(name)
    checked = load_spike_settings(settings_file(CHECKED_SETTINGS))
    store = checked.store(CHECKED_STORE)
    problems = []
    if store.vectors_from != f"vectors/index/full-{target_words}/{name}":
        problems.append(f"the checked store holds {store.vectors_from}, not the tuned model's")
    if (store.dim, store.format) != (candidate.dimension, "float32"):
        problems.append("the checked store does not hold the model's vectors as float32")
    if checked.search.rrf_k != rrf_k or checked.search.depth != settings.vectors.window:
        problems.append("the checked store does not fuse as the tuning chose")
    if checked.search.hits != run["retrieval"]["limit"]:
        problems.append("the checked store does not return as many passages as the answer step got")
    bm25 = files.choice
    if (bm25.analyzer.name, bm25.params.k1, bm25.params.b) != (
        run["retrieval"]["analyzer"],
        run["retrieval"]["k1"],
        run["retrieval"]["b"],
    ):
        problems.append("the answer step's run used another BM25")
    prompt = load_selector_prompt()
    if prompt.sha256 != run["prompt_sha256"] or run["model"] != settings.answer.model:
        problems.append("the selector's prompt or model differs from the run on the test split")
    if problems:
        raise SystemExit(
            "The package would not be the system that was measured: " + "; ".join(problems)
        )

    prices = settings.generation.prices[run["model"]]
    retrieval = Retrieval(
        analyzer=bm25.analyzer.name,
        k1=bm25.params.k1,
        b=bm25.params.b,
        target_words=target_words,
        rrf_k=int(rrf_k),
        depth=checked.search.depth,
        hits=checked.search.hits,
        gather_below=checked.search.gather_below,
        chunk_rows=checked.vectors.chunk_rows,
        vector_wait_s=settings.service.vector_wait_s,
    )
    embedding = Embedding(
        name=candidate.name,
        model=candidate.model,
        revision=candidate.revision,
        provider=str(candidate.provider),
        quantization=str(candidate.quantization),
        dimension=candidate.dimension,
        query_prompt=candidate.query_prompt,
    )
    answer = AnswerConfig(
        model=run["model"],
        effort=run["effort"],
        max_tokens=run["max_tokens"],
        prompt_sha256=run["prompt_sha256"],
        threshold=run["threshold"],
        monthly_budget_usd=settings.answer.monthly_budget_usd,
        input_price=prices.input,
        output_price=prices.output,
    )
    hashes = {
        "bm25_tuning": text_sha256(reports / TUNING_NAME),
        "hybrid_tuning": text_sha256(reports / HYBRID_TUNING),
        "answer_run": text_sha256(reports / ANSWER_RUN),
    }
    vectors_from = settings.data.search / str(store.vectors_from)
    return retrieval, embedding, answer, hashes, vectors_from, checked.vectors, files


def package(settings: Settings, rebuild: bool) -> PackageManifest:
    """Build the package, copy its manifest to the tracked reports, and return it."""
    retrieval, embedding, answer, hashes, vectors_from, vectors, files = configuration(settings)
    out = package_dir(settings)
    if rebuild and out.exists():
        shutil.rmtree(out)
    logger.info("Building %s from %s", out, vectors_from)
    manifest = build_package(
        out,
        name=settings.service.package,
        files=files,
        vectors_from=vectors_from,
        vectors=vectors,
        retrieval=retrieval,
        embedding=embedding,
        answer=answer,
        report_hashes=hashes,
    )
    shutil.copyfile(out / MANIFEST_NAME, settings.tracked.reports / REPORT_NAME)
    _, digest = read_manifest(out)
    size = sum(entry.bytes for entry in manifest.files.values())
    logger.info(
        "%d passages of %d speeches, %s to %s, %.2f GB in %d files; manifest %s",
        manifest.passages,
        manifest.speeches,
        manifest.first_date,
        manifest.last_date,
        size / 1e9,
        len(manifest.files),
        digest,
    )
    return manifest


def main() -> None:
    """Parse the command line and build the package."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--rebuild", action="store_true", help="Remove a package of the same name first."
    )
    args = parser.parse_args()

    configure_logging()
    package(settings_from(args), args.rebuild)


if __name__ == "__main__":
    main()
