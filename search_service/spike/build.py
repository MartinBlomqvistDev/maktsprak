"""Build the stores of the storage spike from the corpus, and prepare its query models.

Every numpy store shares one directory, ``numpy/``: its BM25 index, text and
filter fields once, and one vector file per numpy store selected. Each LanceDB
store is a directory of its own, named after the store, and so is a numpy store
of real vectors, which also holds the vector of every question it is to be
searched with. With ``--models``, every
query model the Hugging Face cache holds is prepared under ``models/`` in every
runtime and precision, and checked on the questions in ``--check-texts``; what
became of each variant is written to ``models/preparation.json``. ``run.py
--build`` runs this in a container without limits, writing into the Docker
volume.

Usage::

    python -m search_service.spike.build --search-dir DIR --tuning FILE --out DIR
        [--only NAME ...] [--limit N] [--models --check-texts FILE [--hf-cache DIR]]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from search_service.spike.inputs import CorpusFiles
from search_service.spike.numpy_store import STORE_DIR_NAME, build_numpy_store
from search_service.spike.settings import SpikeSettings, load_spike_settings, variant_name

logger = logging.getLogger("spike.build")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Build the stores of the storage spike.")
    parser.add_argument(
        "--search-dir",
        type=Path,
        required=True,
        help="The search-data directory, with anforanden.parquet, passages.parquet and lexical/.",
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        required=True,
        help="bm25_tuning_dev.json, which names the BM25 the tuning chose.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where the stores are written.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Build only these stores; --only without a name builds none.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Also prepare every query model in every runtime and precision, under models/.",
    )
    parser.add_argument(
        "--check-texts",
        type=Path,
        default=None,
        help="A JSON list of the questions every prepared model is checked on.",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="The Hugging Face hub cache the query models are read from; nothing is downloaded.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first N passages, for a quick check of the code.",
    )
    return parser.parse_args(argv)


def prepare_all(
    settings: SpikeSettings, out: Path, hf_cache: Path | None, texts: list[str]
) -> list[dict]:
    """Prepare every query model in every variant, and say what became of each.

    Returns:
        One entry per variant, as ``models/preparation.json`` holds them.
    """
    from search_service.spike.query_model import (
        MODELS_DIR_NAME,
        PREPARATION_NAME,
        Preparation,
        prepare_models,
    )

    models_dir = out / MODELS_DIR_NAME
    models_dir.mkdir(parents=True, exist_ok=True)
    variants = settings.query_models.variants()
    results: list[Preparation] = []
    for spec in settings.query_models.models:
        started = time.perf_counter()
        try:
            made = prepare_models(
                spec, variants, models_dir, hf_cache, settings.query_models.onnx, texts
            )
        except (OSError, ValueError) as error:
            reason = f"not loaded from the Hugging Face cache: {type(error).__name__}: {error}"
            made = [
                Preparation(label=variant_name(spec.name, r, p), prepared=None, error=reason)
                for r, p in variants
            ]
        for entry in made:
            if entry.prepared is None:
                logger.warning("%s was not prepared: %s", entry.label, entry.error)
            else:
                check = entry.prepared.check
                logger.info(
                    "%s: %d bytes, cosine with float32 %s at worst",
                    entry.label,
                    entry.prepared.bytes,
                    check.cosine_min if check else "unknown",
                )
        logger.info("%s: %.0f s", spec.name, time.perf_counter() - started)
        results.extend(made)
    entries = [entry.model_dump(mode="json") for entry in results]
    (models_dir / PREPARATION_NAME).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return entries


def main(argv: list[str] | None = None) -> int:
    """Build the stores the command line names, or all of them."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    settings = load_spike_settings()
    files = CorpusFiles.under(args.search_dir, args.tuning)
    specs = [spec for spec in settings.stores if args.only is None or spec.name in args.only]
    unknown = set(args.only or ()) - {spec.name for spec in settings.stores}
    if unknown:
        logger.error("No store is called %s.", ", ".join(sorted(unknown)))
        return 2
    if args.models and args.check_texts is None:
        logger.error("--models needs --check-texts, the questions each model is checked on.")
        return 2

    for spec in specs:
        if spec.kind != "numpy" or not spec.real:
            continue
        started = time.perf_counter()
        manifest = build_numpy_store(
            args.out / spec.name,
            files,
            [(spec.dim, spec.format)],
            settings.vectors,
            vectors_from=args.search_dir / str(spec.vectors_from),
            queries_from=args.search_dir / str(spec.queries_from),
        )
        logger.info(
            "%s: %d passages of real vectors, %d questions, %.0f s",
            spec.name,
            manifest.passages,
            manifest.query_vectors,
            time.perf_counter() - started,
        )
    shapes = [(spec.dim, spec.format) for spec in specs if spec.kind == "numpy" and not spec.real]
    if shapes:
        started = time.perf_counter()
        manifest = build_numpy_store(
            args.out / STORE_DIR_NAME, files, shapes, settings.vectors, args.limit
        )
        logger.info(
            "numpy: %d passages, %d vector files, %.0f s",
            manifest.passages,
            len(manifest.vectors),
            time.perf_counter() - started,
        )
    lance_specs = [spec for spec in specs if spec.kind == "lance"]
    if lance_specs:
        from search_service.spike.lance_store import build_lance_store

        for spec in lance_specs:
            manifest = build_lance_store(args.out / spec.name, files, spec, settings, args.limit)
            logger.info("%s: %d passages, %s", spec.name, manifest.passages, manifest.seconds)
    if args.models and not settings.query_models.models:
        logger.info("The settings name no query model, so the models already prepared stay.")
    elif args.models:
        texts = json.loads(args.check_texts.read_text(encoding="utf-8"))
        if not texts:
            logger.error("%s holds no questions to check the models on.", args.check_texts)
            return 2
        prepare_all(settings, args.out, args.hf_cache, texts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
