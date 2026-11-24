"""Check LanceDB's full-text search against its documentation, on the real corpus.

LanceDB's documentation lists Swedish among the languages it stems and stops,
says the ranking is BM25, and does not say which ``k1`` and ``b`` it uses.
Nothing about it is assumed here; seven things are measured.

1. **Stemming.** Words of the corpus, the commonest and a random draw from its
   vocabulary, are indexed one per row with LanceDB's Swedish stemmer. A query
   for each word should find exactly the rows whose word has the same stem by
   Snowball's Swedish stemmer, which the BM25 index of :mod:`search.index.lexical`
   uses.
2. **Folding.** Whether folding letters to ASCII, LanceDB's default, makes words
   such as "får" and "far" one word, with the default and without it.
3. **Stop words.** Which of Snowball's Swedish stop words LanceDB's Swedish list
   drops, and which of the corpus's commonest other words it drops too.
4. **Long words.** How many words of the corpus are longer than 40 bytes, the
   length beyond which LanceDB drops a token by default, and whether a limit of
   ``None`` keeps them.
5. **BM25 on one word.** A sample of the passages is indexed by LanceDB and by
   :mod:`search.index.lexical`. For a query of one word a score is
   ``idf * f(tf, dl)``, so ``k1`` and ``b`` can be fitted over a grid.
6. **BM25 on questions.** Natural questions of the development split are scored
   both ways, with the fitted ``k1`` and ``b``: once with a repeated word counted
   once, as the tuned BM25 counts it, and once counted as often as it occurs.
7. **int8 vectors.** Whether LanceDB can search a column of int8 vectors.

Usage::

    python -m search_service.spike.probe_fts --search-dir DIR --reports-dir DIR --out FILE
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS
from pydantic import BaseModel, ConfigDict

from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_report
from search.eval.questions import read_question_set, set_file_name, split_of
from search.index.analyzer import Analyzer, stem_all, stopwords
from search.index.lexical import (
    LEXICAL_DIR_NAME,
    TERMS_NAME,
    Bm25Params,
    LexicalIndex,
    build_index,
    passage_texts,
    tokenize_corpus,
)
from search_service.spike.inputs import TUNING_NAME, read_bm25_choice
from search_service.spike.settings import FtsSettings, ProbeSettings, load_spike_settings

logger = logging.getLogger("spike.probe")

REPORT_NAME = "lancedb_fts_probe.json"

#: The index whose vocabulary holds the corpus's words as written, lower-cased.
RAW_ANALYZER = Analyzer(stem=False, stopwords=False)

#: A score agrees when it is this close to the other, relative to its size: float32's precision.
AGREES = 1e-5


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StemmingCheck(_Frozen):
    """Whether LanceDB's Swedish stems group the corpus's words as Snowball's do.

    Attributes:
        words: How many words were checked.
        common: How many of them are among the corpus's commonest.
        unmatched: Words whose own query found nothing.
        differing: Words whose query found other rows than Snowball's stem groups.
        examples: Up to 20 differing words, with what each query found and what Snowball groups.
    """

    words: int
    common: int
    unmatched: int
    differing: int
    examples: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]


class FoldingCheck(_Frozen):
    """Which words each query finds, with LanceDB's default folding and without.

    Attributes:
        rows: The indexed words, one per row.
        folded: For each query, the words found with folding to ASCII.
        unfolded: For each query, the words found without.
    """

    rows: tuple[str, ...]
    folded: dict[str, tuple[str, ...]]
    unfolded: dict[str, tuple[str, ...]]


class StopWordCheck(_Frozen):
    """LanceDB's Swedish stop words against Snowball's.

    Attributes:
        snowball_words: How many stop words Snowball's Swedish list holds.
        dropped_by_lancedb: How many of them LanceDB drops.
        kept_by_lancedb: Those it keeps.
        other_words_checked: How many of the corpus's commonest other words were checked.
        other_words_dropped: Those of them LanceDB drops too.
    """

    snowball_words: int
    dropped_by_lancedb: int
    kept_by_lancedb: tuple[str, ...]
    other_words_checked: int
    other_words_dropped: tuple[str, ...]


class LongWordCheck(_Frozen):
    """Words longer than LanceDB's default token limit.

    Attributes:
        limit_bytes: The default limit.
        corpus_words_over: How many distinct words of the corpus are longer.
        corpus_passages_over: How many passages hold one, counted per word.
        longest: The corpus's longest word.
        found_at_default: Whether a query for the longest word finds it at the default limit.
        found_without_limit: Whether it does with a limit of ``None``.
    """

    limit_bytes: int
    corpus_words_over: int
    corpus_passages_over: int
    longest: str
    found_at_default: bool
    found_without_limit: bool


class OneWordFit(_Frozen):
    """``k1`` and ``b`` fitted to LanceDB's scores for a query of one word.

    Attributes:
        word: The query.
        term: Its term after the analyzer.
        document_frequency: How many sampled passages hold it.
        hits: How many of LanceDB's hits were compared.
        k1: The best ``k1`` on the grid.
        b: The best ``b`` on the grid.
        max_deviation: The largest relative deviation of the implied idf at the best fit.
        implied_idf: The idf LanceDB's scores imply at the best fit.
        lucene_idf: Lucene's idf, ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
    """

    word: str
    term: str
    document_frequency: int
    hits: int
    k1: float
    b: float
    max_deviation: float
    implied_idf: float
    lucene_idf: float


class QuestionCheck(_Frozen):
    """LanceDB's scores on questions against :mod:`search.index.lexical`'s, at the fitted values.

    Attributes:
        questions: How many questions were scored.
        k1: The ``k1`` used.
        b: The ``b`` used.
        agree_counted_once: Questions whose every hit scores alike, a repeated word counted once.
        agree_counted_each_time: Questions whose every hit scores alike, a repeated word
            counted as often as it occurs.
        questions_with_a_repeated_term: Questions in which some term occurs twice or more.
    """

    questions: int
    k1: float
    b: float
    agree_counted_once: int
    agree_counted_each_time: int
    questions_with_a_repeated_term: int


class FtsProbeReport(_Frozen):
    """The check of LanceDB's full-text search.

    Attributes:
        lancedb: The version of LanceDB checked.
        stemmer: The version of PyStemmer the BM25 index uses.
        fts: The full-text index's settings in the stores.
        sampled_passages: How many passages the checks of BM25 indexed.
        stemming: The check of the stemmer.
        folding: The check of folding to ASCII.
        stop_words: The check of the stop words.
        long_words: The check of long words.
        one_word: The fit on queries of one word.
        questions: The check on questions.
        int8_vectors: What LanceDB answered to a search of int8 vectors.
    """

    lancedb: str
    stemmer: str
    fts: FtsSettings
    sampled_passages: int
    stemming: StemmingCheck
    folding: FoldingCheck
    stop_words: StopWordCheck
    long_words: LongWordCheck
    one_word: tuple[OneWordFit, ...]
    questions: QuestionCheck
    int8_vectors: str


def _fts(settings: FtsSettings, **changes: object) -> FTS:
    return FTS(**{**settings.model_dump(), **changes})


def _word_table(database: lancedb.DBConnection, name: str, words: list[str], config: FTS):
    table = database.create_table(name, pa.table({"id": list(range(len(words))), "text": words}))
    table.create_index("text", config=config)
    return table


def _found(table, query: str, limit: int) -> set[int]:
    rows = table.search(query, query_type="fts", fts_columns="text").select(["id"]).limit(limit)
    return {row["id"] for row in rows.to_list()}


def check_stemming(
    database, vocabulary: list[str], frequency: np.ndarray, probe: ProbeSettings, fts: FtsSettings
) -> StemmingCheck:
    """Group words by LanceDB's stems and by Snowball's, and compare."""
    rng = random.Random(probe.seed)
    common = {int(i) for i in np.argsort(-frequency, kind="stable")[: probe.common_tokens]}
    drawn = set(rng.sample(range(len(vocabulary)), min(probe.drawn_tokens, len(vocabulary))))
    words = sorted(vocabulary[i] for i in common | drawn)
    stems = stem_all(words)
    groups: dict[str, set[int]] = defaultdict(set)
    for row, stem in enumerate(stems):
        groups[stem].add(row)
    table = _word_table(database, "stemming", words, _fts(fts))
    unmatched, differing = [], []
    for row, word in enumerate(words):
        found = _found(table, word, len(words))
        expected = groups[stems[row]]
        if not found:
            unmatched.append(word)
        elif found != expected:
            differing.append(
                (
                    word,
                    tuple(sorted(words[i] for i in found))[:6],
                    tuple(sorted(words[i] for i in expected))[:6],
                )
            )
    return StemmingCheck(
        words=len(words),
        common=len(common),
        unmatched=len(unmatched),
        differing=len(differing),
        examples=tuple(differing[:20]),
    )


def check_folding(database, fts: FtsSettings) -> FoldingCheck:
    """Which words each query finds with folding to ASCII and without."""
    words = ["får", "far", "här", "har", "ön", "on", "säl", "sal"]
    folded = _word_table(database, "folded", words, _fts(fts, ascii_folding=True))
    unfolded = _word_table(database, "unfolded", words, _fts(fts, ascii_folding=False))
    return FoldingCheck(
        rows=tuple(words),
        folded={w: tuple(sorted(words[i] for i in _found(folded, w, 10))) for w in words},
        unfolded={w: tuple(sorted(words[i] for i in _found(unfolded, w, 10))) for w in words},
    )


def check_stop_words(
    database, vocabulary: list[str], frequency: np.ndarray, probe: ProbeSettings, fts: FtsSettings
) -> StopWordCheck:
    """Which words LanceDB's Swedish stop words drop: Snowball's, and the commonest others."""
    snowball = sorted(stopwords())
    common = [vocabulary[i] for i in np.argsort(-frequency, kind="stable")[: probe.common_tokens]]
    others = [word for word in common if word not in stopwords()]
    words = snowball + others
    config = _fts(fts, stem=False, remove_stop_words=True)
    table = _word_table(database, "stops", words, config)
    dropped = {word for word in words if not _found(table, word, 5)}
    return StopWordCheck(
        snowball_words=len(snowball),
        dropped_by_lancedb=sum(word in dropped for word in snowball),
        kept_by_lancedb=tuple(word for word in snowball if word not in dropped),
        other_words_checked=len(others),
        other_words_dropped=tuple(word for word in others if word in dropped),
    )


def check_long_words(
    database, vocabulary: list[str], frequency: np.ndarray, fts: FtsSettings
) -> LongWordCheck:
    """Count the corpus's words beyond LanceDB's default limit, and query the longest."""
    limit = FTS().max_token_length
    over = [i for i, word in enumerate(vocabulary) if len(word.encode("utf-8")) > limit]
    longest = max(vocabulary, key=lambda word: len(word.encode("utf-8")))
    words = [longest, "kort"]
    at_default = _word_table(database, "long_default", words, _fts(fts, max_token_length=limit))
    without = _word_table(database, "long_none", words, _fts(fts, max_token_length=None))
    return LongWordCheck(
        limit_bytes=limit,
        corpus_words_over=len(over),
        corpus_passages_over=int(sum(int(frequency[i]) for i in over)),
        longest=longest,
        found_at_default=bool(_found(at_default, longest, 5)),
        found_without_limit=bool(_found(without, longest, 5)),
    )


def _lucene_idf(passages: int, frequency: int) -> float:
    return math.log(1 + (passages - frequency + 0.5) / (frequency + 0.5))


def fit_one_word(table, index: LexicalIndex, word: str, probe: ProbeSettings) -> OneWordFit:
    """Fit ``k1`` and ``b`` to LanceDB's scores for a query of one word.

    For one term, a score is ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``.
    At the right ``k1`` and ``b`` the implied idf, the score divided by the rest, is
    the same for every hit.
    """
    term = index.analyzer.terms(word)[0]
    number = index.term_ids[term]
    start, end = int(index.term_ptr[number]), int(index.term_ptr[number + 1])
    docs = np.asarray(index.doc_ids[start:end]).tolist()
    counts = np.asarray(index.tfs[start:end]).tolist()
    tf_of = dict(zip(docs, counts, strict=True))
    rows = (
        table.search(word, query_type="fts", fts_columns="text")
        .select(["id"])
        .limit(probe.hits_per_query)
        .to_list()
    )
    ids = np.array([row["id"] for row in rows])
    scores = np.array([row["_score"] for row in rows], dtype=np.float64)
    tf = np.array([tf_of.get(int(i), 0) for i in ids], dtype=np.float64)
    length = np.asarray(index.doc_lengths)[ids].astype(np.float64) / index.mean_length
    best = None
    for k1 in ProbeSettings.values(probe.k1_grid):
        for b in ProbeSettings.values(probe.b_grid):
            implied = scores / (tf * (k1 + 1) / (tf + k1 * (1 - b + b * length)))
            deviation = float(np.max(np.abs(implied / np.median(implied) - 1)))
            if best is None or deviation < best[0]:
                best = (deviation, k1, b, float(np.median(implied)))
    if best is None:
        raise ValueError("The grids of k1 and b are empty.")
    return OneWordFit(
        word=word,
        term=term,
        document_frequency=end - start,
        hits=len(ids),
        k1=best[1],
        b=best[2],
        max_deviation=best[0],
        implied_idf=round(best[3], 6),
        lucene_idf=round(_lucene_idf(index.passages, end - start), 6),
    )


def counted_each_time(index: LexicalIndex, text: str, params: Bm25Params) -> np.ndarray:
    """BM25 of every passage with a repeated term of the query counted as often as it occurs."""
    scores = np.zeros(index.passages, dtype=np.float64)
    norm = params.k1 * (1 - params.b + params.b * np.asarray(index.doc_lengths) / index.mean_length)
    for term in index.analyzer.terms(text):
        number = index.term_ids.get(term)
        if number is None:
            continue
        start, end = int(index.term_ptr[number]), int(index.term_ptr[number + 1])
        docs = np.asarray(index.doc_ids[start:end])
        tf = np.asarray(index.tfs[start:end]).astype(np.float64)
        idf = _lucene_idf(index.passages, end - start)
        scores[docs] += idf * tf * (params.k1 + 1) / (tf + norm[docs])
    return scores


def check_questions(
    table, index: LexicalIndex, questions: list[str], params: Bm25Params
) -> QuestionCheck:
    """Score questions both ways and count those whose every hit agrees."""
    once = each = repeated = 0
    for text in questions:
        rows = table.search(text, query_type="fts", fts_columns="text").select(["id"]).limit(50)
        rows = rows.to_list()
        terms = index.analyzer.terms(text)
        repeated += len(terms) != len(set(terms))
        if not rows:
            continue
        ids = [row["id"] for row in rows]
        lance = np.array([row["_score"] for row in rows], dtype=np.float64)
        ours_once = index.scores(text, params)[ids]
        ours_each = counted_each_time(index, text, params)[ids]
        once += bool(np.all(np.abs(lance / ours_once - 1) < AGREES))
        each += bool(np.all(np.abs(lance / ours_each - 1) < AGREES))
    return QuestionCheck(
        questions=len(questions),
        k1=params.k1,
        b=params.b,
        agree_counted_once=once,
        agree_counted_each_time=each,
        questions_with_a_repeated_term=repeated,
    )


def check_int8(database) -> str:
    """Try a search of a column of int8 vectors, and return what LanceDB answers."""
    schema = pa.schema([pa.field("vector", pa.list_(pa.int8(), 8))])
    codes = pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(80, dtype=np.int8)), 8)
    table = database.create_table("int8", pa.table([codes], schema=schema))
    try:
        table.search(np.zeros(8, dtype=np.float32), vector_column_name="vector").limit(1).to_arrow()
    except Exception as error:  # noqa: BLE001 - the answer itself is what is reported
        return f"{type(error).__name__}: {error}"
    return "searched"


def run_probe(search_dir: Path, reports_dir: Path) -> FtsProbeReport:
    """Run every check.

    Args:
        search_dir: The search-data directory.
        reports_dir: The tracked reports, with the tuning report and the question sets.

    Returns:
        The report.
    """
    settings = load_spike_settings()
    probe, fts = settings.probe, settings.lance.fts
    choice = read_bm25_choice(reports_dir / TUNING_NAME)
    raw_dir = search_dir / LEXICAL_DIR_NAME / RAW_ANALYZER.name
    vocabulary = json.loads((raw_dir / TERMS_NAME).read_text(encoding="utf-8"))
    frequency = np.diff(np.load(raw_dir / "term_ptr.npy"))

    workdir = Path(tempfile.mkdtemp(prefix="lancedb-probe-"))
    try:
        database = lancedb.connect(workdir)
        logger.info("Stemming")
        stemming = check_stemming(database, vocabulary, frequency, probe, fts)
        folding = check_folding(database, fts)
        stops = check_stop_words(database, vocabulary, frequency, probe, fts)
        long_words = check_long_words(database, vocabulary, frequency, fts)

        logger.info("BM25")
        sample = [
            pair
            for number, pair in enumerate(
                passage_texts(search_dir / SPEECHES_NAME, search_dir / PASSAGES_NAME)
            )
            if number % probe.passage_step == 0
        ]
        index = build_index(tokenize_corpus(sample), choice.analyzer)
        table = database.create_table(
            "passages", pa.table({"id": list(range(len(sample))), "text": [t for _, t in sample]})
        )
        table.create_index("text", config=_fts(fts))
        fits = tuple(fit_one_word(table, index, word, probe) for word in probe.one_word_queries)
        k1, b = Counter((fit.k1, fit.b) for fit in fits).most_common(1)[0][0]
        questions_file = reports_dir / "questions" / set_file_name("natural")
        natural = split_of(read_question_set(questions_file), "dev")
        texts = [question.text for question in sorted(natural, key=lambda q: q.question_id)]
        questions = check_questions(table, index, texts[: probe.questions], Bm25Params(k1=k1, b=b))
        int8 = check_int8(database)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return FtsProbeReport(
        lancedb=importlib.metadata.version("lancedb"),
        stemmer=f"PyStemmer {importlib.metadata.version('PyStemmer')}",
        fts=fts,
        sampled_passages=len(sample),
        stemming=stemming,
        folding=folding,
        stop_words=stops,
        long_words=long_words,
        one_word=fits,
        questions=questions,
        int8_vectors=int8,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the checks and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Check LanceDB's full-text search on the corpus.")
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_report(run_probe(args.search_dir, args.reports_dir), args.out)
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
