"""Step 15: write what the page Fråga protokollet shows without asking the search service.

Three commands, each writing a file the site reads:

- ``examples`` answers the questions in ``search/page/examples.yaml`` with the
  chain the page runs. It writes one file per example under
  ``web/public/fraga/exempel/``, which the page fetches when an example is
  chosen, and the list of examples to ``web/src/data/fraga/examples.json``. Each
  question's vector is asked for once, one question per request, as the page
  asks for it. The question is then searched with the hybrid, once without a
  filter and once under each party, and each search's ten best passages go to
  Claude Opus 5.5 with the prompt, effort and output limit of the answer
  step's run on the test split from the hybrid's candidates. The assembler
  quotes what the model selects. The requests go as one message batch, at half
  the list price, with the parameters the page sends one at a time. A request
  that fails is sent once more, and when that fails too the baseline answers,
  as on the page. The run keeps its vectors, candidates, requests' log and
  answers under ``data/search/page/examples/``, and a second run reads them
  back, so nothing is paid twice.
- ``method`` writes the figures of the method section to
  ``web/src/data/fraga/method.json``, read from ``data/reports/search/``.
- ``speakers`` writes the list the page's speaker filter offers to
  ``web/public/fraga/talare.json``.
- ``dashes`` counts, in each of the Riksdag's datasets as downloaded, the
  records that hold an en dash, and writes
  ``data/reports/search/source_dashes.json``. The datasets up to 2023/24 hold
  none, where the Riksdag's page of the same speech has one.
- ``links`` requests the page of every speech the examples show, from
  data.riksdagen.se, and judges each against the speech with the deep-link
  check's own judgement: the page must name the speech and hold the opening
  words of each of its paragraphs in order, and hold neither neighbour. It
  writes ``data/reports/search/page_example_links.json``. Requests go two at a
  time at the most and at least a second apart, as in the deep-link check.

Usage::

    python search/pipeline/15_build_page.py examples --max-usd 3 --confirm
    python search/pipeline/15_build_page.py method
    python search/pipeline/15_build_page.py speakers
    python search/pipeline/15_build_page.py links
    python search/pipeline/15_build_page.py dashes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict

from search.answer.assemble import assemble
from search.answer.candidates import (
    CANDIDATES,
    build_candidates,
    fuse_candidates,
    load_speeches,
    rank_passages,
)
from search.answer.models import Answer, Candidate, Selected
from search.answer.selector import (
    BaselineSelector,
    ClaudeSelector,
    failed,
    fall_back,
    load_selector_prompt,
)
from search.cli import add_data_root_argument, configure_logging, settings_from
from search.config import REPO_ROOT, Settings
from search.corpus.links import (
    HostThrottle,
    PageResponse,
    RequestBudget,
    check_for,
    neighbours_of,
    totals_of,
)
from search.corpus.store import SPEECHES_NAME, write_report
from search.corpus.table import PassageTable
from search.eval.generation import append_log, read_log
from search.eval.llm import Completion, cost_usd, load_key, run_batch
from search.eval.runner import now
from search.index.embed import save_array
from search.index.lexical import Hit, file_sha256
from search.page.contract import (
    AnswerLine,
    IndexFacts,
    PartiesLine,
    SearchLine,
    SpeechFacts,
    answer_line,
    party_answer,
    search_line,
)
from search.page.method import method_figures
from search.page.speakers import SpeechRow, speakers
from search.parties import PARTY_ORDER

logger = logging.getLogger(__name__)

#: The example questions, in the order the page shows them.
EXAMPLES_FILE = REPO_ROOT / "search" / "page" / "examples.yaml"

#: Where the site reads its data.
WEB_DATA = REPO_ROOT / "web" / "src" / "data" / "fraga"
WEB_PUBLIC = REPO_ROOT / "web" / "public" / "fraga"
EXAMPLES_JSON = WEB_DATA / "examples.json"
EXAMPLE_DIR = WEB_PUBLIC / "exempel"
METHOD_JSON = WEB_DATA / "method.json"
SPEAKERS_JSON = WEB_PUBLIC / "talare.json"

#: The run of the answer step whose model, prompt, effort and threshold the page runs.
HYBRID_TEST_RUN_REPORT = "answer_test_run_hybrid.json"

#: The part of a request's id that marks a search without a party filter.
NO_FILTER = "alla"

#: The files of a run, under ``data/search/page/examples/``.
RUN_NAME = "run.json"
VECTORS_NAME = "query_vectors.npy"
CANDIDATES_NAME = "candidates.jsonl"
ANSWERS_NAME = "answers.jsonl"

STEP_14 = Path(__file__).with_name("14_run_answer_step.py")
LINK_CHECK = Path(__file__).with_name("check_deep_links.py")

#: The report of the check of the examples' links.
LINKS_REPORT = "page_example_links.json"

#: The report on how the Riksdag's datasets write the en dash.
DASHES_REPORT = "source_dashes.json"

#: The en dash, which the Riksdag's datasets up to 2023/24 write as a hyphen.
EN_DASH = "\u2013"


def load_script(name: str, path: Path) -> ModuleType:
    """Load one of the pipeline's scripts as a module, to reuse its functions."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Pydantic resolves the script's postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_step_14() -> ModuleType:
    """The answer step's script, whose checks and loaders the examples reuse."""
    return load_script("pipeline_14_run_answer_step", STEP_14)


class ExampleQuestion(BaseModel):
    """One example question.

    Attributes:
        id: The key the page knows it by.
        text: The question as a visitor would type it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str


class ExampleList(BaseModel):
    """The example questions, in the order the page shows them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: tuple[ExampleQuestion, ...]


class ExamplesRun(BaseModel):
    """What one run of the examples used, and what its requests cost.

    Attributes:
        questions_sha256: The hash of ``examples.yaml``'s text.
        embedding: The embedding model, by its name in the settings.
        rrf_k: The RRF constant of the hybrid.
        window: How many passages each of the hybrid's two rankings holds.
        tuning_sha256: The hash of the hybrid's tuning report.
        vectors_sha256: The hash of the corpus's vector matrix.
        model: The model that selects.
        round: The development round whose prompt runs.
        prompt_sha256: That prompt's hash.
        effort: The model's effort.
        max_tokens: The most output tokens of one answer.
        threshold: The baseline's threshold, from the hybrid's candidates.
        vector_waits_ms: How long each question waited for its vector.
        vector_cost_usd: What the vectors cost.
        query_vectors_sha256: The hash of the questions' vectors.
        batches: The message batches, by attempt.
        created: When the run began.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions_sha256: str
    embedding: str
    rrf_k: float
    window: int
    tuning_sha256: str
    vectors_sha256: str
    model: str
    round: str
    prompt_sha256: str
    effort: str
    max_tokens: int
    threshold: float
    vector_waits_ms: dict[str, int]
    vector_cost_usd: float
    query_vectors_sha256: str
    batches: dict[str, str] = {}
    created: datetime


class ExampleCandidates(BaseModel):
    """One search of one example question, and the candidates it gave.

    Attributes:
        custom_id: The request's id: the example's key and the party, or ``alla``.
        example: The example's key.
        party: The party the search was filtered on, or ``None``.
        question: The question.
        retrieval: ``hybrid``, or ``bm25`` when the vector came too late.
        candidates: The passages ranked highest, best first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str
    example: str
    party: str | None
    question: str
    retrieval: Literal["hybrid", "bm25"]
    candidates: tuple[Candidate, ...]


class ExampleAnswer(BaseModel):
    """One search's answer, as the assembler wrote it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str
    answer: Answer


class Example(BaseModel):
    """One example as the page shows it, in the lines ``/api/fraga`` answers with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    search: SearchLine
    answer: AnswerLine
    parties: PartiesLine


class ChainFacts(BaseModel):
    """What the examples were answered with, as the page states it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str
    provider: str
    rrf_k: float
    model: str
    prompt_sha256: str
    effort: str
    transport: str
    requests: int
    cost_usd: float


class ExampleEntry(BaseModel):
    """One example in the page's list: its key and its question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str


class ExamplesFile(BaseModel):
    """The page's list of examples, and what they were answered with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated: date
    chain: ChainFacts
    examples: tuple[ExampleEntry, ...]


def custom_id(example: str, party: str | None) -> str:
    """A request's id: the example's key, then the party or ``alla``."""
    return f"{example}--{party or NO_FILTER}"


def text_sha256(path: Path) -> str:
    """The SHA-256 of a text file, with Windows line ends read as Unix ones."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_examples(path: Path = EXAMPLES_FILE) -> ExampleList:
    """Read the example questions.

    Raises:
        ValueError: If two examples share a key.
    """
    examples = ExampleList.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    keys = [question.id for question in examples.questions]
    if len(set(keys)) != len(keys):
        raise ValueError("Two example questions share a key.")
    return examples


def speech_facts(speeches_path: Path, anf_ids: set[str]) -> dict[str, SpeechFacts]:
    """The session, protocol number and reply flag of some speeches, by key."""
    if not anf_ids:
        return {}
    rows = pq.read_table(
        speeches_path,
        columns=["anf_id", "rm", "dok_nummer", "is_reply"],
        filters=pc.field("anf_id").isin(sorted(anf_ids)),
    ).to_pylist()
    return {
        row["anf_id"]: SpeechFacts(
            rm=row["rm"], dok_nummer=int(row["dok_nummer"]), is_reply=bool(row["is_reply"])
        )
        for row in rows
    }


def index_facts(speeches_path: Path, table: PassageTable) -> IndexFacts:
    """The index's latest speech date and its sizes, read from the corpus it was built from."""
    dates = pq.read_table(speeches_path, columns=["date"]).column("date")
    return IndexFacts(
        last_date=pc.max(dates).as_py(),
        speeches=len(dates),
        passages=len(table),
    )


def write_lines(path: Path, entries: Sequence[BaseModel]) -> None:
    """Write entries as JSON Lines, one per line, with Unix line ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def search_hits(
    context: Mapping[str, object],
    text: str,
    vector: np.ndarray | None,
    allowed: np.ndarray | None,
) -> list[Hit]:
    """One search's ten best passages: the hybrid, or BM25 alone without a vector."""
    index = context["index"]
    params = context["params"]
    if vector is None:
        return rank_passages(index, text, params, allowed)
    window = context["window"]
    lexical = [hit.passage for hit in index.search(text, params, limit=window, allowed=allowed)]
    found = context["vector_index"].search(vector, window, allowed)
    return fuse_candidates(
        index, text, params, lexical, [hit.passage for hit in found], context["k"], CANDIDATES
    )


def run_examples(settings: Settings, cap_usd: float, env_file: Path | None) -> None:
    """Answer the example questions with the page's chain, and write the page's file.

    Raises:
        SystemExit: If the hybrid's configuration or the answer step's prompt
            is not the one recorded, or the batch could cost more than *cap_usd*.
    """
    step14 = load_step_14()
    step14.check_hybrid_registered()
    step14.check_tuned(settings)
    config = step14.hybrid_configuration(settings)
    record_path = settings.tracked.reports / HYBRID_TEST_RUN_REPORT
    held_out = step14.HybridTestRun.model_validate_json(record_path.read_text(encoding="utf-8"))
    prompt = load_selector_prompt(step14.round_dir(settings, held_out.round))
    if prompt.sha256 != held_out.prompt_sha256:
        raise SystemExit(f"The copy of {held_out.round}'s prompt no longer matches its hash.")

    examples = load_examples()
    directory = settings.data.search / "page" / "examples"
    run_path = directory / RUN_NAME
    questions_sha256 = text_sha256(EXAMPLES_FILE)

    index, _, table = step14.load_retrieval(settings)
    vector_index, vectors_sha256 = step14.load_hybrid_index(settings, config, table)
    params = step14.bm25_params(settings)
    texts = [question.text for question in examples.questions]

    if run_path.is_file():
        run = ExamplesRun.model_validate_json(run_path.read_text(encoding="utf-8"))
        if run.questions_sha256 != questions_sha256 or run.prompt_sha256 != prompt.sha256:
            raise SystemExit(f"{run_path} was made from other questions or another prompt.")
        vectors = np.load(directory / VECTORS_NAME, allow_pickle=False)
        if file_sha256(directory / VECTORS_NAME) != run.query_vectors_sha256:
            raise SystemExit(f"{directory / VECTORS_NAME} no longer matches its hash.")
    else:
        logger.info("Asking for %d question vectors one by one", len(texts))
        vectors, answered, usage, _ = step14.ask_for_vectors(
            settings, config.embedding, texts, env_file
        )
        directory.mkdir(parents=True, exist_ok=True)
        vectors_hash = save_array(directory / VECTORS_NAME, np.asarray(vectors, dtype=np.float32))
        run = ExamplesRun(
            questions_sha256=questions_sha256,
            embedding=config.embedding,
            rrf_k=config.k,
            window=config.window,
            tuning_sha256=config.tuning_sha256,
            vectors_sha256=vectors_sha256,
            model=held_out.model,
            round=held_out.round,
            prompt_sha256=held_out.prompt_sha256,
            effort=held_out.effort,
            max_tokens=held_out.max_tokens,
            threshold=held_out.threshold,
            vector_waits_ms={
                question.id: round(one.seconds * 1000)
                for question, one in zip(examples.questions, answered, strict=True)
            },
            vector_cost_usd=round(usage.cost_usd, 6),
            query_vectors_sha256=vectors_hash,
            created=now(),
        )
        write_report(run, run_path)

    context: dict[str, object] = {
        "index": index,
        "params": params,
        "vector_index": vector_index,
        "window": config.window,
        "k": config.k,
    }
    candidates_path = directory / CANDIDATES_NAME
    if candidates_path.is_file():
        searches = read_log(candidates_path, ExampleCandidates)
    else:
        planned: list[tuple[ExampleQuestion, str | None, Literal["hybrid", "bm25"], list[Hit]]] = []
        for row, question in enumerate(examples.questions):
            late = run.vector_waits_ms[question.id] > step14.VECTOR_WAIT_S * 1000
            vector = None if late else vectors[row]
            retrieval: Literal["hybrid", "bm25"] = "bm25" if late else "hybrid"
            for party in (None, *PARTY_ORDER):
                allowed = None if party is None else table.party_mask(party)
                hits = search_hits(context, question.text, vector, allowed)
                planned.append((question, party, retrieval, hits))
        speeches = load_speeches(
            settings.data.search / SPEECHES_NAME,
            {table.anf_ids[hit.passage] for *_, hits in planned for hit in hits},
        )
        searches = [
            ExampleCandidates(
                custom_id=custom_id(question.id, party),
                example=question.id,
                party=party,
                question=question.text,
                retrieval=retrieval,
                candidates=tuple(build_candidates(hits, table, speeches)),
            )
            for question, party, retrieval, hits in planned
        ]
        write_lines(candidates_path, searches)

    price = settings.generation.prices[run.model]
    selector = ClaudeSelector(prompt, run.model, run.effort, run.max_tokens, price)
    baseline = BaselineSelector(index, params, threshold=run.threshold)
    requests = {
        search.custom_id: selector.request(search.question, search.candidates)
        for search in searches
        if search.candidates
    }
    discount = settings.generation.batch_discount

    def spent() -> float:
        return sum(
            cost_usd(completion.usage, price, discount)
            for attempt in ("first", "again")
            for completion in read_log(directory / f"completions_{attempt}.jsonl", Completion)
        )

    def batch(attempt: str, todo: Mapping[str, Mapping[str, object]]) -> list[Completion]:
        nonlocal run
        log = directory / f"completions_{attempt}.jsonl"
        if log.is_file():
            return read_log(log, Completion)
        if attempt not in run.batches:
            _, costs = step14.round_costs(settings, run.model, todo)
            if spent() + costs[-1] > cap_usd:
                raise SystemExit(
                    f"Spent {spent():.2f} USD; this batch could cost up to {costs[-1]:.2f} USD, "
                    f"beyond the cap of {cap_usd:.2f} USD."
                )

        def remember(batch_id: str) -> None:
            nonlocal run
            run = ExamplesRun.model_validate(
                {**run.model_dump(), "batches": {**run.batches, attempt: batch_id}}
            )
            write_report(run, run_path)

        done = run_batch(todo, on_submitted=remember, batch_id=run.batches.get(attempt))
        ordered = [done[key] for key in todo if key in done]
        append_log(log, ordered)
        return ordered

    sent: dict[str, list[Completion]] = {}
    for completion in batch("first", requests):
        sent[completion.custom_id] = [completion]
    again = {
        key: params_
        for key, params_ in requests.items()
        if key not in sent or failed(sent[key][-1])
    }
    if again:
        for completion in batch("again", again):
            sent.setdefault(completion.custom_id, []).append(completion)

    speeches = load_speeches(
        settings.data.search / SPEECHES_NAME,
        {candidate.anf_id for search in searches for candidate in search.candidates},
    )
    answers = [
        ExampleAnswer(
            custom_id=search.custom_id,
            answer=assemble(
                selected_for(selector, baseline, search, sent.get(search.custom_id), discount),
                search.candidates,
                speeches,
            ),
        )
        for search in searches
    ]
    write_lines(directory / ANSWERS_NAME, answers)
    write_examples(settings, examples, run, searches, answers, table, spent())


def selected_for(
    selector: ClaudeSelector,
    baseline: BaselineSelector,
    search: ExampleCandidates,
    completions: Sequence[Completion] | None,
    discount: float,
) -> Selected:
    """One search's selection: the model's, or the baseline's after two failures."""
    if not search.candidates:
        return selector.select(search.question, search.candidates)
    selected = selector.read(
        completions or [Completion(custom_id=search.custom_id, error="no result")], discount
    )
    if selected.output is None:
        selected = fall_back(
            selected, baseline.select(search.question, search.candidates), "failed"
        )
    return selected


def write_examples(
    settings: Settings,
    examples: ExampleList,
    run: ExamplesRun,
    searches: Sequence[ExampleCandidates],
    answers: Sequence[ExampleAnswer],
    table: PassageTable,
    cost: float,
) -> None:
    """Write each example in the lines ``/api/fraga`` answers with, and the list of examples."""
    speeches_path = settings.data.search / SPEECHES_NAME
    facts = speech_facts(
        speeches_path, {candidate.anf_id for search in searches for candidate in search.candidates}
    )
    index = index_facts(speeches_path, table)
    by_id = {search.custom_id: search for search in searches}
    answer_of = {line.custom_id: line.answer for line in answers}
    candidate = settings.vectors.candidate(run.embedding)
    built = []
    for question in examples.questions:
        whole = by_id[custom_id(question.id, None)]
        wait = run.vector_waits_ms[question.id]
        waited = None if whole.retrieval == "bm25" else wait
        built.append(
            Example(
                id=question.id,
                question=question.text,
                search=search_line(whole.candidates, facts, whole.retrieval, waited, index),
                answer=answer_line(answer_of[whole.custom_id], facts),
                parties=PartiesLine(
                    retrieval=whole.retrieval,
                    vector_wait_ms=waited,
                    index=index,
                    parties=tuple(
                        party_answer(party, answer_of[custom_id(question.id, party)], facts)
                        for party in PARTY_ORDER
                    ),
                ),
            )
        )
    page = ExamplesFile(
        generated=run.created.date(),
        chain=ChainFacts(
            embedding_model=candidate.model,
            provider=str(candidate.provider),
            rrf_k=run.rrf_k,
            model=run.model,
            prompt_sha256=run.prompt_sha256,
            effort=run.effort,
            transport="message batch",
            requests=len([search for search in searches if search.candidates]),
            cost_usd=round(cost + run.vector_cost_usd, 4),
        ),
        examples=tuple(ExampleEntry(id=item.id, question=item.question) for item in built),
    )
    for item in built:
        write_json(item, EXAMPLE_DIR / f"{item.id}.json")
    write_json(page, EXAMPLES_JSON)
    logger.info(
        "%d examples, %d searches; the requests cost %.4f USD", len(built), len(searches), cost
    )


def write_json(model: BaseModel, path: Path) -> None:
    """Write a model as indented JSON with Unix line ends, keeping non-ASCII text as it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump(mode="json")
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def run_method(settings: Settings) -> None:
    """Write the method section's figures."""
    figures = method_figures(settings)
    write_json(figures, METHOD_JSON)
    logger.info(
        "Method figures from %d reports; the answer step's test report is %s",
        len(figures.sources),
        "in" if figures.answer_test.report is not None else "not written yet",
    )


def run_speakers(settings: Settings) -> None:
    """Write the speaker filter's list."""
    rows = pq.read_table(
        settings.data.search / SPEECHES_NAME,
        columns=["person_id", "speaker_raw", "party", "date"],
    ).to_pylist()
    entries = speakers((SpeechRow(**row) for row in rows), PARTY_ORDER)
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    SPEAKERS_JSON.write_text(
        json.dumps(
            [entry.model_dump(mode="json") for entry in entries],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("%d speakers", len(entries))


class LinkCheck(BaseModel):
    """One example link and how its page was judged.

    Attributes:
        anf_id: The speech.
        url: The address the examples show.
        computed: Whether that address is the one ``deep_link`` gives the speech.
        passed_with_folded_dashes: Whether the page passes once its en dashes
            are written as hyphens.
        check: The deep-link check's judgement of the page that came back.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anf_id: str
    url: str
    computed: bool
    passed_with_folded_dashes: bool
    check: dict[str, object]


class ExampleLinksReport(BaseModel):
    """The check of every link the page's examples show.

    Attributes:
        examples_sha256: The hash of the examples' list, which names the files read.
        speeches: How many distinct speeches the examples link to.
        totals: The deep-link check's counts over them.
        passed_with_folded_dashes: How many pages pass once their en dashes
            are written as hyphens.
        computed: How many addresses are the ones ``deep_link`` gives.
        xml_records: How many pages came back as the open-data XML record.
        failed: The speeches whose page did not pass.
        checks: Every link and its judgement.
        created: When the check ended.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    examples_sha256: str
    speeches: int
    totals: dict[str, int]
    passed_with_folded_dashes: int
    computed: int
    xml_records: int
    failed: tuple[str, ...]
    checks: tuple[LinkCheck, ...]
    created: datetime


def example_urls() -> dict[str, str]:
    """Every speech the examples link to, with the address the page shows."""
    listed = json.loads(EXAMPLES_JSON.read_text(encoding="utf-8"))
    urls: dict[str, str] = {}
    for entry in listed["examples"]:
        example = Example.model_validate_json(
            (EXAMPLE_DIR / f"{entry['id']}.json").read_text(encoding="utf-8")
        )
        quotes = [
            *example.answer.quotes,
            *(quote for party in example.parties.parties for quote in party.quotes),
        ]
        for item in (*example.search.hits, *quotes):
            urls.setdefault(item.anf_id, item.url)
    return dict(sorted(urls.items()))


def fold_dashes(html: str) -> str:
    """A page with its en dashes written as hyphens, as the datasets up to 2023/24 write them."""
    return html.replace(EN_DASH, "-")


def run_links(settings: Settings) -> None:
    """Request every example link's page and judge it against its speech.

    Each page is judged twice: as it came back, and with its en dashes written
    as hyphens. The Riksdag's datasets up to the 2023/24 session hold no en
    dash, where the page of the same speech has one, so a quotation from those
    sessions can differ from its page in that character alone. The second
    judgement shows which failures that accounts for; the first is the check.
    """
    checker = load_script("pipeline_check_deep_links", LINK_CHECK)
    urls = example_urls()
    corpus = settings.data.search / SPEECHES_NAME
    refs = [ref for ref in checker.load_speech_refs(corpus) if ref.anf_id in urls]
    if len(refs) != len(urls):
        raise SystemExit("An example links to a speech the corpus does not hold.")
    texts = checker.load_speech_texts(corpus, sorted({ref.dok_id for ref in refs}))
    attempts = 2
    budget = RequestBudget(len(refs) * attempts)
    throttle = HostThrottle(checker.MIN_INTERVAL_SECONDS)

    async def fetch() -> dict[str, PageResponse]:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            return await checker.fetch_pages(
                [ref.url for ref in refs],
                client=client,
                throttle=throttle,
                budget=budget,
                concurrency=checker.MAX_CONCURRENCY,
                attempts=attempts,
            )

    pages = asyncio.run(fetch())
    checks = []
    folded = []
    for ref in refs:
        response = pages[ref.url]
        neighbours = neighbours_of(ref, texts)
        checks.append(check_for(ref, texts[ref.anf_id], response, neighbours))
        if response.html is not None:
            response = response.model_copy(update={"html": fold_dashes(response.html)})
        folded.append(check_for(ref, texts[ref.anf_id], response, neighbours))
    entries = tuple(
        LinkCheck(
            anf_id=page.anf_id,
            url=urls[page.anf_id],
            computed=urls[page.anf_id] == page.url,
            passed_with_folded_dashes=other.passed,
            check=page.model_dump(mode="json"),
        )
        for page, other in zip(checks, folded, strict=True)
    )
    report = ExampleLinksReport(
        examples_sha256=text_sha256(EXAMPLES_JSON),
        speeches=len(entries),
        totals=totals_of(checks).model_dump(),
        passed_with_folded_dashes=sum(page.passed for page in folded),
        computed=sum(entry.computed for entry in entries),
        xml_records=sum(page.xml_record for page in checks),
        failed=tuple(page.anf_id for page in checks if not page.passed),
        checks=entries,
        created=now(),
    )
    write_report(report, settings.tracked.reports / LINKS_REPORT)
    logger.info(
        "%d speeches: %d pages passed, %d with the dashes folded, %d addresses as computed, "
        "%d XML records; failed: %s",
        report.speeches,
        report.totals["passed"],
        report.passed_with_folded_dashes,
        report.computed,
        report.xml_records,
        ", ".join(report.failed) or "none",
    )


class SessionDashes(BaseModel):
    """How one session's dataset writes the en dash.

    Attributes:
        session: The session, such as ``2021/22``.
        file: The dataset's file, as the manifest names it.
        records: How many records the dataset holds.
        with_en_dash: How many of them hold an en dash.
        spaced_hyphens: How many times a hyphen stands between two spaces.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: str
    file: str
    records: int
    with_en_dash: int
    spaced_hyphens: int


class DashesReport(BaseModel):
    """Which of the Riksdag's datasets hold an en dash at all.

    Attributes:
        manifest_sha256: The hash of the manifest that names the datasets.
        sessions: Each dataset, in the manifest's order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_sha256: str
    sessions: tuple[SessionDashes, ...]


def run_dashes(settings: Settings) -> None:
    """Count the records with an en dash in each of the Riksdag's datasets.

    The datasets up to the 2023/24 session were found to hold none, where the
    Riksdag's page of the same speech has one, so a quotation from those
    sessions can differ from its page in that character. This records that per
    session, read from the datasets as downloaded.
    """
    manifest_path = settings.tracked.reports / "anforanden_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dash = EN_DASH.encode("utf-8")
    sessions = []
    for entry in manifest["files"]:
        path = settings.data.raw_anforanden / entry["file"]
        records = with_dash = spaced = 0
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                data = archive.read(name)
                records += 1
                with_dash += dash in data
                spaced += data.count(b" - ")
        sessions.append(
            SessionDashes(
                session=entry["session"],
                file=entry["file"],
                records=records,
                with_en_dash=with_dash,
                spaced_hyphens=spaced,
            )
        )
    report = DashesReport(manifest_sha256=text_sha256(manifest_path), sessions=tuple(sessions))
    write_report(report, settings.tracked.reports / DASHES_REPORT)
    logger.info(
        "Sessions whose dataset holds no en dash: %s",
        ", ".join(item.session for item in sessions if not item.with_en_dash) or "none",
    )


def main() -> None:
    """Parse the command line and run the command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="The .env file that holds the API keys. Defaults to .env in the data root.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    examples = commands.add_parser("examples", help="Answer the example questions.")
    examples.add_argument("--max-usd", type=float, required=True, help="The run's cap in USD.")
    examples.add_argument("--confirm", action="store_true", help="Required: this calls the APIs.")
    commands.add_parser("method", help="Write the method section's figures.")
    commands.add_parser("speakers", help="Write the speaker filter's list.")
    commands.add_parser("links", help="Check every link the examples show.")
    commands.add_parser("dashes", help="Count the en dashes in each of the Riksdag's datasets.")
    args = parser.parse_args()

    configure_logging()
    settings = settings_from(args)
    if args.command == "examples":
        if not args.confirm:
            raise SystemExit("The examples call the APIs; pass --confirm.")
        load_key(args.env_file or settings.data.search.parents[1] / ".env")
        run_examples(settings, args.max_usd, args.env_file)
    elif args.command == "method":
        run_method(settings)
    elif args.command == "links":
        run_links(settings)
    elif args.command == "dashes":
        run_dashes(settings)
    else:
        run_speakers(settings)


if __name__ == "__main__":
    main()
