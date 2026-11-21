"""Step 05: build the BM25 index over the passages, once for each analyzer.

Reads ``anforanden.parquet`` and ``passages.parquet``, splits every passage into
tokens once, and builds one inverted index for each of the four analyzers the
tuning chooses between: with and without stemming, with and without the stop
words. The indexes go to ``lexical/`` in the search-data directory. The report,
which is tracked, records what each index was built from and how large it is.

Usage::

    python search/pipeline/05_build_lexical_index.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ConfigDict

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report
from search.index.analyzer import ANALYZERS, STOPWORDS_FILE
from search.index.lexical import (
    IndexManifest,
    build_index,
    file_sha256,
    lexical_dir,
    passage_ids_sha256,
    passage_texts,
    tokenize_corpus,
)

logger = logging.getLogger("lexical_index")

REPORT_NAME = "lexical_index.json"


class LexicalIndexReport(BaseModel):
    """What the four indexes were built from and what they hold.

    Attributes:
        stemmer: The package and version that stemmed the tokens.
        stopwords_sha256: The hash of the stop-word file.
        indexes: One manifest per analyzer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stemmer: str
    stopwords_sha256: str
    indexes: tuple[IndexManifest, ...]


def _stemmer_name() -> str:
    """Name the stemmer in use: PyStemmer when it is installed, else snowballstemmer."""
    try:
        return f"PyStemmer {version('PyStemmer')}"
    except PackageNotFoundError:
        return f"snowballstemmer {version('snowballstemmer')}"


def main() -> None:
    """Parse the command line and build the indexes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    speeches_path = settings.data.search / SPEECHES_NAME
    passages_path = settings.data.search / PASSAGES_NAME
    for path, step in (
        (speeches_path, "02_build_anforanden.py"),
        (passages_path, "04_pack_passages.py"),
    ):
        if not path.is_file():
            raise SystemExit(f"{path} does not exist. Run {step} first.")

    started = time.monotonic()
    corpus = tokenize_corpus(passage_texts(speeches_path, passages_path))
    logger.info(
        "%d passages split into %d tokens of %d distinct forms in %.0f s",
        corpus.passages,
        len(corpus.tokens),
        len(corpus.vocabulary),
        time.monotonic() - started,
    )
    speeches_sha256 = file_sha256(speeches_path)
    passages_sha256 = file_sha256(passages_path)
    ids_sha256 = passage_ids_sha256(corpus.passage_ids)

    manifests = []
    for analyzer in ANALYZERS:
        started = time.monotonic()
        index = build_index(corpus, analyzer)
        manifest = IndexManifest(
            analyzer=analyzer,
            passages=index.passages,
            passage_ids_sha256=ids_sha256,
            passages_sha256=passages_sha256,
            speeches_sha256=speeches_sha256,
            terms=len(index.terms),
            postings=len(index.doc_ids),
            tokens=int(index.doc_lengths.sum()),
            mean_passage_terms=round(index.mean_length, 3),
        )
        index.save(lexical_dir(settings.data.search, analyzer), manifest)
        manifests.append(manifest)
        logger.info(
            "%s: %d terms, %d postings, %.1f terms per passage, built in %.0f s",
            analyzer.name,
            manifest.terms,
            manifest.postings,
            manifest.mean_passage_terms,
            time.monotonic() - started,
        )
        del index

    report = LexicalIndexReport(
        stemmer=_stemmer_name(),
        stopwords_sha256=file_sha256(STOPWORDS_FILE),
        indexes=tuple(manifests),
    )
    write_report(report, settings.tracked.reports / REPORT_NAME)
    logger.info("Indexes written to %s", settings.data.search / "lexical")


if __name__ == "__main__":
    main()
