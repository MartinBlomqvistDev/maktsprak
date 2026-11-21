"""Count the corpus in words and in tokens, with each tokenizer family available.

Reads only the paragraphs column of the speech corpus. Two tokenizer families
are tried: the Swedish WordPiece tokenizer of the project's own classifier, and
the SentencePiece tokenizer of the multilingual model XLM-RoBERTa. The report
records the SHA-256 of the serialised form of each tokenizer the corpus was
counted with. A tokenizer that cannot be loaded is recorded in the report and
the others are counted. No later step reads this script's report, so the script
is unnumbered.

Usage::

    python search/pipeline/count_tokens.py
    python search/pipeline/count_tokens.py --classifier-dir path/to/party_classifier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.corpus.build import BUILD_REPORT_NAME, BuildReport
from search.corpus.store import BATCH_ROWS, SPEECHES_NAME, write_report
from search.corpus.tokens import (
    TokenCountReport,
    TokenizerCount,
    model_type_of,
    serialised_sha256_of,
    spread_of,
    token_lengths,
)

logger = logging.getLogger("count_tokens")

REPORT_NAME = "token_counts.json"
PROGRESS_EVERY = 25
CLASSIFIER_TOKENIZER = "tokenizer.json"
CLASSIFIER_CONFIG = "config.json"
CLASSIFIER_NAME = "swedish_wordpiece"
XLM_ROBERTA_REPO = "xlm-roberta-base"
XLM_ROBERTA_NAME = "xlm_roberta_sentencepiece"

NOTE = (
    "These are tokenizer families. No embedding model has been chosen, and a "
    "model's own tokenizer gives its own count. Words are whitespace-separated, "
    "as in the build report. Token counts leave out special tokens and are made "
    "paragraph by paragraph, without truncation or padding."
)


class TokenizerUnavailable(Exception):
    """A tokenizer could not be loaded; the message says why."""


def load_classifier_tokenizer(directory: Path) -> tuple[Tokenizer, str]:
    """Load the tokenizer of the project's own classifier.

    Args:
        directory: The classifier's model directory.

    The ``config.json`` beside the tokenizer is read only to name the model the
    tokenizer belongs to. A config that is missing or cannot be read leaves that
    name out of the description.

    Args:
        directory: The classifier's model directory.

    Returns:
        The tokenizer, with truncation and padding switched off, and a
        description of where it came from.

    Raises:
        TokenizerUnavailable: If the directory has no ``tokenizer.json`` or the
            file cannot be parsed.
    """
    path = directory / CLASSIFIER_TOKENIZER
    if not path.is_file():
        raise TokenizerUnavailable(f"No {CLASSIFIER_TOKENIZER} in {directory.name}.")
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as error:
        raise TokenizerUnavailable(f"{type(error).__name__} while loading it.") from error
    tokenizer.no_truncation()
    tokenizer.no_padding()
    source = f"local file {directory.name}/{CLASSIFIER_TOKENIZER}"
    base = _base_model_of(directory / CLASSIFIER_CONFIG)
    if base:
        source += f", the tokenizer of {base}"
    return tokenizer, source


def _base_model_of(config: Path) -> str | None:
    """Read the model name from a ``config.json``, or None if it holds none.

    Args:
        config: The path of the file. It may be missing or unreadable.

    Returns:
        The ``_name_or_path`` entry, or None if the file is missing, cannot be
        parsed, or has no text entry of that name.
    """
    try:
        content = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    base = content.get("_name_or_path") if isinstance(content, dict) else None
    return base if isinstance(base, str) and base else None


def load_xlm_roberta_tokenizer() -> tuple[Tokenizer, str]:
    """Load the XLM-RoBERTa tokenizer, from the Hugging Face cache or by downloading it.

    Returns:
        The tokenizer, with truncation and padding switched off, and a
        description of where it came from.

    Raises:
        TokenizerUnavailable: If it cannot be downloaded or read.
    """
    try:
        tokenizer = Tokenizer.from_pretrained(XLM_ROBERTA_REPO)
    except Exception as error:
        raise TokenizerUnavailable(f"{type(error).__name__} while loading it.") from error
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, f"huggingface.co/{XLM_ROBERTA_REPO}, tokenizer.json"


def paragraph_batches(path: Path) -> Iterator[list[str]]:
    """Read the speakers' paragraphs from the corpus, one list per batch of speeches."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=["paragraphs"]):
        yield batch.column("paragraphs").flatten().to_pylist()


def count_corpus(
    path: Path, tokenizers: dict[str, tuple[Tokenizer, str]], unavailable: dict[str, str]
) -> TokenCountReport:
    """Count the corpus once, with every tokenizer.

    Args:
        path: The corpus Parquet file.
        tokenizers: For each tokenizer that was loaded, its name, the tokenizer
            and its source.
        unavailable: For each tokenizer that could not be loaded, why.

    Returns:
        The report.
    """
    paragraphs = words = characters = 0
    lengths: dict[str, list[int]] = {name: [] for name in tokenizers}
    for number, batch in enumerate(paragraph_batches(path), start=1):
        paragraphs += len(batch)
        words += sum(len(paragraph.split()) for paragraph in batch)
        characters += sum(len(paragraph) for paragraph in batch)
        for name, (tokenizer, _) in tokenizers.items():
            lengths[name].extend(token_lengths(batch, tokenizer))
        if number % PROGRESS_EVERY == 0:
            logger.info("%d paragraphs counted", paragraphs)

    counts = tuple(
        TokenizerCount(
            name=name,
            model_type=model_type_of(tokenizer),
            source=source,
            serialised_sha256=serialised_sha256_of(tokenizer),
            vocab_size=tokenizer.get_vocab_size(),
            total_tokens=sum(lengths[name]),
            tokens_per_word=round(sum(lengths[name]) / words, 4),
            tokens_per_paragraph=spread_of(lengths[name]),
        )
        for name, (tokenizer, source) in tokenizers.items()
    )
    return TokenCountReport(
        paragraphs=paragraphs,
        words=words,
        characters=characters,
        tokenizers=counts,
        unavailable=unavailable,
        note=NOTE,
    )


def main() -> None:
    """Parse the command line, count the corpus and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=None,
        help=(
            "The classifier's model directory. Defaults to data.party_classifier in config.yaml."
        ),
    )
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    classifier_dir = args.classifier_dir or settings.data.party_classifier

    loaders = {
        CLASSIFIER_NAME: lambda: load_classifier_tokenizer(classifier_dir),
        XLM_ROBERTA_NAME: load_xlm_roberta_tokenizer,
    }
    available: dict[str, tuple[Tokenizer, str]] = {}
    unavailable: dict[str, str] = {}
    for name, load in loaders.items():
        try:
            available[name] = load()
            logger.info("Loaded %s", name)
        except TokenizerUnavailable as error:
            unavailable[name] = str(error)
            logger.warning("%s is unavailable: %s", name, error)
    if not available:
        raise SystemExit("No tokenizer could be loaded.")

    corpus = settings.data.search / SPEECHES_NAME
    report = count_corpus(corpus, available, unavailable)

    built = BuildReport.model_validate_json(
        (settings.tracked.reports / BUILD_REPORT_NAME).read_text(encoding="utf-8")
    )
    if (report.words, report.paragraphs) != (built.words, built.paragraphs):
        raise SystemExit(
            f"The counts differ from the build report: {report.words} words and "
            f"{report.paragraphs} paragraphs against {built.words} and {built.paragraphs}."
        )

    write_report(report, settings.tracked.reports / REPORT_NAME)
    for entry in report.tokenizers:
        logger.info(
            "%s: %d tokens, %.3f per word", entry.name, entry.total_tokens, entry.tokens_per_word
        )
    logger.info("Report written to %s", settings.tracked.reports / REPORT_NAME)


if __name__ == "__main__":
    main()
