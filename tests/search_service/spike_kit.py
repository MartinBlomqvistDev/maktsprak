"""A small corpus on disk for the spike's tests: speeches, passages, BM25 indexes, questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from search.eval.questions import (
    EvalQuestion,
    Provenance,
    QuestionFilter,
    set_file_name,
    write_question_set,
)
from search.index.analyzer import Analyzer
from search.index.lexical import (
    IndexManifest,
    build_index,
    file_sha256,
    lexical_dir,
    passage_ids_sha256,
    passage_texts,
    tokenize_corpus,
)
from search.models import Anforande
from search_service.spike.inputs import TUNING_NAME
from search_service.spike.settings import (
    SearchSettings,
    SpikeSettings,
    StoreSpec,
    VectorSettings,
    WorkloadSettings,
    load_spike_settings,
)
from tests.search.corpus_kit import speech, write_corpus

#: The analyzer the tuning report of the kit names, as the real one does.
CHOSEN = Analyzer(stem=True, stopwords=False)

#: The analyzer whose vocabulary the check of LanceDB's stemmer reads.
RAW = Analyzer(stem=False, stopwords=False)

PROVENANCE = Provenance(
    model="claude-opus-5-5", prompt_sha256="a" * 64, created=date(2026, 11, 22), generation=1
)


def _speech(
    dok_id: str, number: int, paragraphs: list[str], party: str, day: date, person: str | None
) -> Anforande:
    return speech(dok_id, number, paragraphs, party=party, day=day).model_copy(
        update={"person_id": person}
    )


#: Twelve speeches over three periods and all eight parties. Each is one passage, ``{anf_id}:0-1``.
SPEECHES: list[Anforande] = [
    _speech(
        "GR0950",
        1,
        ["Skolan behöver fler lärare och mindre klasser.", "Lärarna ska få högre lön."],
        "S",
        date(2004, 3, 2),
        "0101",
    ),
    _speech(
        "GR0950",
        2,
        ["Friskolor ger valfrihet för föräldrar.", "Kommunala skolor ska också utvecklas."],
        "M",
        date(2004, 3, 2),
        "0202",
    ),
    _speech(
        "GR0950",
        3,
        ["Skolan ska vara likvärdig i hela landet.", "Vinster i skolan ska förbjudas."],
        "V",
        date(2004, 3, 2),
        "0303",
    ),
    _speech(
        "H109050",
        1,
        ["Kärnkraftsavvecklingen kostar jobb i industrin.", "Elpriset stiger."],
        "M",
        date(2013, 11, 5),
        "0202",
    ),
    _speech(
        "H109050",
        2,
        ["Avveckling av kärnkraften är rätt väg.", "Sol och vind ger billig el."],
        "MP",
        date(2013, 11, 5),
        "0404",
    ),
    _speech(
        "H109050",
        3,
        ["Kärnkraften behövs för klimatets skull.", "Ny kärnkraft ska byggas."],
        "KD",
        date(2013, 11, 5),
        "0505",
    ),
    _speech(
        "H109050",
        4,
        ["Landsbygden behöver billig el.", "Kärnkraften ska inte subventioneras."],
        "C",
        date(2013, 11, 5),
        None,
    ),
    _speech(
        "HB0960",
        1,
        ["Sjukvården behöver kortare köer.", "Fler sjuksköterskor ska anställas i regionerna."],
        "SD",
        date(2023, 2, 14),
        "0606",
    ),
    _speech(
        "HB0960",
        2,
        ["Vårdköerna har blivit längre under regeringens tid.", "Det är ett misslyckande."],
        "V",
        date(2023, 2, 14),
        "0303",
    ),
    _speech(
        "HB0960",
        3,
        ["Valfrihet i vården ger kortare väntetider.", "Privata vårdgivare behövs."],
        "L",
        date(2023, 2, 14),
        "0707",
    ),
    _speech(
        "HB0960",
        4,
        ["Skatten på arbete ska sänkas.", "Jobbskatteavdraget har gett fler jobb."],
        "M",
        date(2023, 2, 14),
        "0808",
    ),
    _speech(
        "HB0960",
        5,
        ["Sjukvården ska styras av behov.", "Köerna måste kortas i hela landet."],
        "S",
        date(2023, 2, 14),
        "0101",
    ),
]


def _manifest(
    analyzer: Analyzer, index, speeches: Path, passages: Path, ids: list[str]
) -> IndexManifest:
    return IndexManifest(
        analyzer=analyzer,
        passages=len(ids),
        passage_ids_sha256=passage_ids_sha256(ids),
        passages_sha256=file_sha256(passages),
        speeches_sha256=file_sha256(speeches),
        terms=len(index.terms),
        postings=len(index.doc_ids),
        tokens=int(index.doc_lengths.sum()),
        mean_passage_terms=round(index.mean_length, 3),
    )


def _generated(
    name: str, number: int, split: str, text: str, passage: str, **extra
) -> EvalQuestion:
    return EvalQuestion(
        question_id=f"{name}-{number:04d}",
        question_set=name,
        split=split,
        text=text,
        gold=(passage.split(":")[0],),
        passage_id=passage,
        provenance=PROVENANCE,
        **extra,
    )


QUESTIONS: dict[str, list[EvalQuestion]] = {
    "natural": [
        _generated("natural", 1, "dev", "Ska det byggas ny kärnkraft?", "H109050-3:0-1"),
        _generated("natural", 2, "dev", "Behöver skolan fler lärare?", "GR0950-1:0-1"),
        _generated("natural", 3, "dev", "Hur kortas köerna i vården?", "HB0960-5:0-1"),
        _generated("natural", 4, "test", "Vad kostar avvecklingen?", "H109050-1:0-1"),
    ],
    "keyword": [
        _generated("keyword", 1, "dev", "kärnkraft klimat", "H109050-3:0-1"),
        _generated("keyword", 2, "test", "skatt arbete", "HB0960-4:0-1"),
    ],
    "filtered": [
        _generated(
            "filtered",
            1,
            "dev",
            "Ska friskolor finnas?",
            "GR0950-2:0-1",
            filter=QuestionFilter(party="M"),
        ),
        _generated(
            "filtered",
            2,
            "dev",
            "Är vårdköerna för långa?",
            "HB0960-2:0-1",
            filter=QuestionFilter(period="2018-2023"),
        ),
        _generated(
            "filtered",
            3,
            "test",
            "Ska kärnkraften subventioneras?",
            "H109050-4:0-1",
            filter=QuestionFilter(party="C"),
        ),
    ],
    "stance": [
        _generated(
            "stance",
            1,
            "dev",
            "Vilka argument har förts fram mot vinster i skolan?",
            "GR0950-3:0-1",
            kind="against",
        ),
    ],
    "interpellation": [
        EvalQuestion(
            question_id="interpellation-0001",
            question_set="interpellation",
            split="dev",
            text="vården i regionerna",
            gold=("HB0960-1", "HB0960-5"),
            session="2022/23",
        ),
        EvalQuestion(
            question_id="interpellation-0002",
            question_set="interpellation",
            split="dev",
            text="elpriset",
            gold=("H109050-1",),
            session="2013/14",
        ),
        EvalQuestion(
            question_id="interpellation-0003",
            question_set="interpellation",
            split="test",
            text="skolans likvärdighet",
            gold=("GR0950-3",),
            session="2003/04",
        ),
    ],
}


@dataclass(frozen=True)
class SpikeCorpus:
    """Where the kit's files are.

    Attributes:
        search_dir: The search-data directory: the corpus and its BM25 indexes.
        reports_dir: The tracked reports: the tuning report and the question sets.
        passage_ids: The passages, in index order.
        texts: Their texts, in the same order.
    """

    search_dir: Path
    reports_dir: Path
    passage_ids: list[str]
    texts: list[str]

    @property
    def tuning(self) -> Path:
        """The tuning report."""
        return self.reports_dir / TUNING_NAME


def write_spike_corpus(directory: Path) -> SpikeCorpus:
    """Write the kit's corpus, two BM25 indexes, a tuning report and question sets."""
    search_dir = directory / "search"
    reports_dir = directory / "reports"
    speeches_path, passages_path = write_corpus(search_dir, SPEECHES)
    pairs = list(passage_texts(speeches_path, passages_path))
    ids = [passage for passage, _ in pairs]
    for analyzer in (CHOSEN, RAW):
        index = build_index(tokenize_corpus(pairs), analyzer)
        manifest = _manifest(analyzer, index, speeches_path, passages_path, ids)
        index.save(lexical_dir(search_dir, analyzer), manifest)
    reports_dir.mkdir(parents=True)
    report = {"chosen": {"analyzer": CHOSEN.name, "k1": 1.2, "b": 0.9}}
    (reports_dir / TUNING_NAME).write_text(json.dumps(report), encoding="utf-8")
    for name, questions in QUESTIONS.items():
        write_question_set(reports_dir / "questions" / set_file_name(name), questions)
    return SpikeCorpus(search_dir, reports_dir, ids, [text for _, text in pairs])


#: The stores of the tests: small vectors, every format, both kinds.
TEST_STORES: tuple[StoreSpec, ...] = (
    StoreSpec(name="numpy-8-float32", kind="numpy", dim=8, format="float32"),
    StoreSpec(name="numpy-8-float16", kind="numpy", dim=8, format="float16"),
    StoreSpec(name="numpy-8-int8", kind="numpy", dim=8, format="int8"),
    StoreSpec(name="lance-8-float32", kind="lance", dim=8, format="float32"),
    StoreSpec(name="lance-8-float16", kind="lance", dim=8, format="float16"),
)


def write_index(directory: Path, ids: list[str], matrix: np.ndarray) -> Path:
    """A vector index as search/index/vector.py writes it, with the manifest fields read back."""
    directory.mkdir()
    np.save(directory / "vectors.npy", matrix)
    np.save(directory / "positions.npy", np.arange(len(ids), dtype=np.int64))
    manifest = {
        "rows": len(ids),
        "dimension": matrix.shape[1],
        "vectors_sha256": file_sha256(directory / "vectors.npy"),
        "positions_sha256": file_sha256(directory / "positions.npy"),
        "inputs": {"passage_ids_sha256": passage_ids_sha256(ids)},
    }
    (directory / "index.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def kit_settings() -> SpikeSettings:
    """The spike's settings, scaled down to the kit's twelve passages."""
    real = load_spike_settings()
    return real.model_copy(
        update={
            "vectors": VectorSettings(seed=7, chunk_rows=5),
            "search": SearchSettings(depth=6, hits=3, rrf_k=60, gather_below=0.5),
            "workload": WorkloadSettings(
                seed=3,
                sets=("natural", "keyword", "filtered", "stance"),
                warmup_set="interpellation",
                warmup_queries=2,
                filters={"party": 1.0, "period": 1.0, "speaker": 1.0, "party_period": 1.0},
                party_view_questions=2,
            ),
            "stores": TEST_STORES,
        }
    )
