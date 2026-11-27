"""The search service's work: a question's vector, the hybrid's ten, and the answer.

The service is the system the test split measured, so the page runs it. A
question waits for its vector from the embedding API for as long as the package
allows; when none has come by then, it is answered from BM25 alone and the
answer says so. The hybrid ranks the passages as the check before the test
split built it, and its ten best are both the page's list of results and the
answer step's candidates, each with its BM25 score, by which the baseline
abstains. The selector picks paragraphs and the assembler quotes them; nothing
else writes the text of a quotation.

The view of the eight parties is outside what was measured, and runs the
baseline selector on each party's best passage, so it calls no model.

A question is put in Unicode's composed form (NFC) before anything reads it,
as the corpus is written, so that a pasted question whose å, ä and ö are
decomposed finds the same words. The answer step runs one question at a time
on an instance, apart from the searches, and within a deadline past which the
baseline answers.

Nothing here keeps or logs a question. The counters count, and the rate limit
keeps a salted hash of each caller in memory, which it forgets within two of
its windows after the caller's last request, or when the instance stops.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import secrets
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from search.answer.assemble import assemble
from search.answer.models import Answer, Candidate, Quote, Selected, Speech
from search.answer.selector import BaselineSelector, fall_back
from search.eval.generation import passage_span
from search.index.lexical import Bm25Params
from search.parties import PARTY_ORDER, VALID_PARTIES
from search_service.package import PackageManifest, SpeechTable, StalePackageError
from search_service.spike.contracts import SearchFilter, SearchRequest
from search_service.spike.fusion import rrf
from search_service.spike.numpy_store import NumpyStore
from search_service.spike.vectors import top_positions

logger = logging.getLogger("search_service")

Retrieval = Literal["hybrid", "bm25"]
Kind = Literal["search", "answer"]

#: The most characters a question may hold as sent, before it is trimmed and
#: held to the settings' ``question_chars``.
QUESTION_MAX_CHARS = 1000

#: The most digits a speaker's id may hold; the Riksdag's ids hold 10 to 13.
SPEAKER_MAX_CHARS = 20

#: The longest the answer step may take for one question, its wait for the
#: instance's one selector included; past it, the baseline answers.
ANSWER_DEADLINE_S = 45.0

#: The most callers the rate limit holds at once, about 50 MB. Past it, the
#: callers not heard from for longest are forgotten first, down to nine tenths.
MAX_CALLERS = 50_000


class InvalidRequest(ValueError):
    """A question or filter the page does not allow."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Filters(_Frozen):
    """The filters of the page, as the proxy sends them; each may be left out.

    Attributes:
        party: One of the eight parties.
        year_from: The first year allowed, inclusive.
        year_to: The last year allowed, inclusive.
        speaker: The Riksdag's id of one speaker.
    """

    party: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    speaker: str | None = Field(default=None, pattern=r"^[0-9]+$", max_length=SPEAKER_MAX_CHARS)


class Question(_Frozen):
    """One question to the service.

    Attributes:
        question: The question as written.
        filters: What it is narrowed to.
    """

    question: str = Field(max_length=QUESTION_MAX_CHARS)
    filters: Filters = Filters()


class Selector(Protocol):
    """What answers a question from its candidates."""

    def select(self, question: str, candidates: Sequence[Candidate]) -> Selected:
        """Answer one question."""
        ...


class ServiceStore(NumpyStore):
    """The numpy store, searched only with a question's own vector.

    The spike's store stands in a synthetic vector when it is given none; the
    service never does, and searches with BM25 alone instead.
    """

    def query(
        self, vector_seed: int, vector: np.ndarray | None, text: str | None = None
    ) -> np.ndarray:
        """The question's vector, which must be given.

        Raises:
            ValueError: If no vector is given.
        """
        if vector is None:
            raise ValueError("The service searches only with the question's own vector.")
        return super().query(vector_seed, vector, text)


@dataclass(frozen=True)
class Found:
    """The hybrid's best passages for one question, and how they were found.

    Attributes:
        retrieval: Both legs, or BM25 alone when the vector did not come.
        vector_wait_ms: How long the question waited for its vector, or
            ``None`` when none came.
        candidates: The best passages, best first, each with its BM25 score.
        speeches: Their speeches, by key, for the assembler.
        rows: Each candidate's speech, as a row of the speeches' table.
    """

    retrieval: Retrieval
    vector_wait_ms: int | None
    candidates: tuple[Candidate, ...]
    speeches: Mapping[str, Speech]
    rows: tuple[int, ...]


class RateLimiter:
    """How many searches and answers one caller may ask in a window, kept in memory.

    A caller is known by a salted hash of what the proxy names it by. The salt
    is drawn when the instance starts and never leaves it. A caller is
    forgotten at the first sweep after a window has passed since its last
    request, and sweeps run at least once a window while the instance is asked
    anything, so no caller is held for more than two windows after its last
    request. No more than ``max_callers`` are held at once.
    """

    def __init__(
        self,
        window_s: float,
        limits: Mapping[Kind, int],
        clock: Callable[[], float] = time.monotonic,
        max_callers: int = MAX_CALLERS,
    ) -> None:
        self.window_s = window_s
        self.limits = dict(limits)
        self.max_callers = max_callers
        self._clock = clock
        self._salt = secrets.token_bytes(16)
        self._seen: dict[tuple[str, Kind], deque[float]] = {}
        self._swept = clock()

    def _key(self, caller: str) -> str:
        return hashlib.sha256(self._salt + caller.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._seen)

    def _sweep(self, now: float) -> None:
        for key in [
            key
            for key, times in self._seen.items()
            if not times or times[-1] <= now - self.window_s
        ]:
            del self._seen[key]
        self._swept = now
        if len(self._seen) >= self.max_callers:
            excess = len(self._seen) - self.max_callers * 9 // 10
            for key in list(itertools.islice(self._seen, excess)):
                del self._seen[key]

    def tidy(self) -> None:
        """Forget the callers whose window has passed, once a window has gone by since a sweep."""
        now = self._clock()
        if now - self._swept >= self.window_s:
            self._sweep(now)

    def check(self, caller: str, kind: Kind) -> float | None:
        """Count one request, or say how long the caller must wait.

        Args:
            caller: What the proxy names the caller by.
            kind: A search or an answer.

        Returns:
            ``None`` when the request is allowed, else the seconds until it would be.
        """
        now = self._clock()
        if now - self._swept >= self.window_s or len(self._seen) >= self.max_callers:
            self._sweep(now)
        key = (self._key(caller), kind)
        times = self._seen.pop(key, None) or deque()
        self._seen[key] = times
        while times and times[0] <= now - self.window_s:
            times.popleft()
        if len(times) >= self.limits[kind]:
            return max(times[0] + self.window_s - now, 1.0)
        times.append(now)
        return None


def _day(value: date | str) -> str:
    return value if isinstance(value, str) else value.isoformat()


class SearchService:
    """The package, the question's model and the selector, behind the page's three questions.

    Attributes:
        manifest: The package's manifest.
        store: The numpy store.
        speeches: The speeches' table.
        selector: What answers a question: the language model within its
            budget, or the baseline.
        baseline: The baseline selector, which also answers the view of the parties.
        rates: The rate limit.
        question_chars: The most characters a question holds once trimmed.
        answer_deadline_s: The longest the answer step may take for one question.
        counters: How many requests of each kind, and what became of them.
    """

    def __init__(
        self,
        manifest: PackageManifest,
        store: ServiceStore,
        speeches: SpeechTable,
        embed: Callable[[str], Awaitable[np.ndarray]],
        selector: Selector,
        rates: RateLimiter,
        question_chars: int,
        answer_deadline_s: float = ANSWER_DEADLINE_S,
    ) -> None:
        self.manifest = manifest
        self.store = store
        self.speeches = speeches
        self._embed = embed
        self.params = Bm25Params(k1=manifest.retrieval.k1, b=manifest.retrieval.b)
        self.baseline = BaselineSelector(store.lexical, self.params, manifest.answer.threshold)
        self.selector = selector
        self.rates = rates
        self.question_chars = question_chars
        self.answer_deadline_s = answer_deadline_s
        self.counters: Counter[str] = Counter()
        # One selector at a time, on threads of its own: a slow model holds
        # this thread, never one of the searches'.
        self._answers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="answer")

    def close(self) -> None:
        """Let go of the answer step's thread, dropping what waits for it."""
        self._answers.shutdown(wait=False, cancel_futures=True)

    # The request -----------------------------------------------------------

    def read(self, body: Question, *, parties: bool = False) -> tuple[str, SearchFilter]:
        """The question trimmed, and its filter, if the page allows them.

        Args:
            body: The question and its filters.
            parties: Whether it asks for the view of the eight parties, which
                chooses the party and asks of no one speaker.

        Returns:
            The question in composed form, trimmed, and the filter.

        Raises:
            InvalidRequest: If the question is empty or too long, or a filter is
                outside what the page offers.
        """
        text = unicodedata.normalize("NFC", body.question).strip()
        if not 1 <= len(text) <= self.question_chars:
            raise InvalidRequest("The question is empty or too long.")
        filters = body.filters
        if filters.party is not None and filters.party not in VALID_PARTIES:
            raise InvalidRequest("The party is not one of the eight.")
        if parties and (filters.party is not None or filters.speaker is not None):
            raise InvalidRequest("The view of the parties chooses the party itself.")
        if filters.speaker is not None and set(filters.speaker) == {"0"}:
            raise InvalidRequest("An id of zeros is the Riksdag's placeholder, not a speaker.")
        first, last = self.manifest.first_date.year, self.manifest.last_date.year
        for year in (filters.year_from, filters.year_to):
            if year is not None and not first <= year <= last:
                raise InvalidRequest("A year is outside the corpus.")
        if (
            filters.year_from is not None
            and filters.year_to is not None
            and filters.year_from > filters.year_to
        ):
            raise InvalidRequest("The first year comes after the last.")
        return text, SearchFilter(
            party=filters.party,
            start=date(filters.year_from, 1, 1) if filters.year_from is not None else None,
            end=date(filters.year_to, 12, 31) if filters.year_to is not None else None,
            person_id=filters.speaker,
        )

    # The vector ------------------------------------------------------------

    async def vector(self, text: str) -> tuple[np.ndarray | None, int | None]:
        """The question's vector, if it comes within the package's wait.

        Returns:
            The vector and how long it took in milliseconds, or ``(None, None)``
            when it did not come in time or the API failed.
        """
        started = time.perf_counter()
        try:
            vector = await asyncio.wait_for(
                self._embed(text), timeout=self.manifest.retrieval.vector_wait_s
            )
        except TimeoutError:
            self.counters["vector_late"] += 1
            return None, None
        except Exception as error:  # any failure of the API means BM25 alone
            self.counters["vector_failed"] += 1
            logger.warning("The question's vector failed: %s", type(error).__name__)
            return None, None
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.store.dim,):
            self.counters["vector_failed"] += 1
            return None, None
        return vector, round((time.perf_counter() - started) * 1000)

    # The search ------------------------------------------------------------

    def _candidate(
        self, rank: int, position: int, score: float, speeches: dict[str, Speech]
    ) -> tuple[Candidate, int]:
        passage = self.store.passage_ids[position]
        anf_id, first, last = passage_span(passage)
        row = self.speeches.row_of(position)
        speech = speeches.get(anf_id) or self.speeches.speech(row)
        if speech.anf_id != anf_id or len(speech.paragraphs) <= last:
            raise StalePackageError(f"{passage} does not fit the speech the package holds.")
        speeches[anf_id] = speech
        candidate = Candidate(
            rank=rank,
            passage_id=passage,
            anf_id=anf_id,
            first=first,
            last=last,
            score=score,
            party=speech.party,
            speaker=speech.speaker,
            date=speech.date,
            title=speech.title,
            url=speech.url,
            paragraphs=speech.paragraphs[first : last + 1],
            text_sha256=speech.text_sha256,
        )
        return candidate, row

    def find(
        self, text: str, search_filter: SearchFilter, vector: np.ndarray | None, waited: int | None
    ) -> Found:
        """Rank the passages and turn the best into candidates.

        Args:
            text: The question.
            search_filter: Its filter, which both legs apply.
            vector: Its vector, or ``None`` for BM25 alone.
            waited: How long it waited for the vector, in milliseconds.

        Returns:
            The best passages with their speeches.
        """
        mode = "hybrid" if vector is not None else "lexical"
        ranking = self.store.rank(
            SearchRequest(text=text, vector_seed=0, filter=search_filter, mode=mode), vector
        )
        positions = [position for position, _ in ranking.ranked]
        speeches: dict[str, Speech] = {}
        candidates, rows = [], []
        if positions:
            # The lexical leg scored every passage; its scores label the candidates.
            scores = ranking.lexical_scores
            if scores is None:
                scores = self.store.lexical.scores(text, self.params)
            for rank, position in enumerate(positions, start=1):
                candidate, row = self._candidate(rank, position, float(scores[position]), speeches)
                candidates.append(candidate)
                rows.append(row)
        return Found(
            retrieval="hybrid" if vector is not None else "bm25",
            vector_wait_ms=waited,
            candidates=tuple(candidates),
            speeches=speeches,
            rows=tuple(rows),
        )

    def answer(self, text: str, found: Found) -> Answer:
        """Select from the candidates and assemble the quotations."""
        answer = assemble(
            self.selector.select(text, found.candidates), found.candidates, found.speeches
        )
        if answer.selected.fallback is not None:
            self.counters[f"fallback_{answer.selected.fallback}"] += 1
        return answer

    def baseline_answer(self, text: str, found: Found, reason: str) -> Answer:
        """The baseline's answer in place of the selector's, marked with *reason*."""
        selected = fall_back(
            Selected(selector="claude", output=None),
            self.baseline.select(text, found.candidates),
            reason,
        )
        self.counters[f"fallback_{reason}"] += 1
        return assemble(selected, found.candidates, found.speeches)

    def _answer_before(self, text: str, found: Found, deadline: float) -> Answer | None:
        if time.monotonic() >= deadline:
            return None  # waited past the deadline: the baseline has answered already
        return self.answer(text, found)

    async def answer_in_time(self, text: str, found: Found) -> Answer:
        """The selector's answer, on the answer step's own thread, within the deadline.

        A question that waits for the thread past the deadline is never sent to
        the model. One whose model has not answered by then is answered by the
        baseline, marked ``failed``; the model's request runs out on its thread,
        and the budget still counts what it costs.

        Returns:
            The answer.
        """
        deadline = time.monotonic() + self.answer_deadline_s
        work = asyncio.get_running_loop().run_in_executor(
            self._answers, self._answer_before, text, found, deadline
        )
        try:
            answer = await asyncio.wait_for(work, timeout=self.answer_deadline_s)
        except TimeoutError:
            answer = None
        if answer is None:
            self.counters["answer_late"] += 1
            return self.baseline_answer(text, found, "failed")
        return answer

    def party_best(
        self, text: str, search_filter: SearchFilter, vector: np.ndarray | None
    ) -> tuple[dict[str, int | None], np.ndarray]:
        """Each party's best passage under the filter, as :meth:`NumpyStore.party_view` ranks it.

        Without a vector, BM25 alone ranks each party's passages.

        Returns:
            The position of each party's best passage, or ``None``, in the
            site's order, and every passage's BM25 score.
        """
        store = self.store
        base = store.mask(search_filter)
        lexical_scores = store.lexical.scores(text, self.params)
        vector_scores = store.vectors.scores(vector) if vector is not None else None
        depth, best = store.settings.depth, {}
        for number, party in enumerate(PARTY_ORDER):
            allowed = store.parties == number
            if base is not None:
                allowed &= base
            rows = np.flatnonzero(allowed)
            matching = rows[lexical_scores[rows] > 0]
            lexical = top_positions(lexical_scores[matching], depth, matching).tolist()
            nearest = (
                top_positions(vector_scores[rows], depth, rows).tolist()
                if vector_scores is not None and len(rows)
                else []
            )
            ranked = rrf([leg for leg in (lexical, nearest) if leg], store.settings.rrf_k, 1)
            best[party] = ranked[0][0] if ranked else None
        return best, lexical_scores

    def parties(
        self, text: str, search_filter: SearchFilter, vector: np.ndarray | None
    ) -> list[tuple[str, Answer, dict[str, int]]]:
        """The baseline's answer from each party's best passage.

        The baseline reads only the first candidate, so each party's best
        passage is all it is given. A party without a passage under the filter
        gets the baseline's abstention.

        Returns:
            Each party in the site's order, with its answer and the speech row
            of each passage the answer may quote.
        """
        best, scores = self.party_best(text, search_filter, vector)
        out: list[tuple[str, Answer, dict[str, int]]] = []
        for party in PARTY_ORDER:
            position = best[party]
            speeches: dict[str, Speech] = {}
            candidates: list[Candidate] = []
            rows: dict[str, int] = {}
            if position is not None:
                candidate, row = self._candidate(1, position, float(scores[position]), speeches)
                candidates, rows = [candidate], {candidate.passage_id: row}
            answer = assemble(self.baseline.select(text, candidates), candidates, speeches)
            out.append((party, answer, rows))
        return out

    # The lines -------------------------------------------------------------

    def _index(self) -> dict[str, Any]:
        return {
            "last_date": self.manifest.last_date.isoformat(),
            "speeches": self.manifest.speeches,
            "passages": self.manifest.passages,
        }

    def _speech_fields(self, row: int) -> dict[str, Any]:
        fields = self.speeches.fields(row)
        return {
            "rm": fields["rm"],
            "dok_nummer": fields["dok_nummer"],
            "is_reply": fields["is_reply"],
        }

    def hit(self, candidate: Candidate, row: int) -> dict[str, Any]:
        """One hit of the list of results, as the page reads it."""
        fields = self._speech_fields(row)
        return {
            "rank": candidate.rank,
            "passage_id": candidate.passage_id,
            "anf_id": candidate.anf_id,
            "rm": fields["rm"],
            "dok_nummer": fields["dok_nummer"],
            "party": candidate.party,
            "speaker": candidate.speaker,
            "date": _day(candidate.date),
            "title": candidate.title,
            "is_reply": fields["is_reply"],
            "url": candidate.url,
            "paragraphs": list(candidate.paragraphs),
        }

    def quote(self, quote: Quote, row: int) -> dict[str, Any]:
        """One quotation, as the page reads it: a hit's fields with the paragraphs quoted."""
        fields = self._speech_fields(row)
        return {
            "rank": quote.rank,
            "passage_id": quote.passage_id,
            "anf_id": quote.anf_id,
            "rm": fields["rm"],
            "dok_nummer": fields["dok_nummer"],
            "party": quote.party,
            "speaker": quote.speaker,
            "date": _day(quote.date),
            "title": quote.title,
            "is_reply": fields["is_reply"],
            "url": quote.url,
            "paragraphs": list(quote.paragraphs),
            "first_paragraph": quote.first_paragraph,
            "last_paragraph": quote.last_paragraph,
        }

    def search_line(self, found: Found) -> dict[str, Any]:
        """The line with the list of results."""
        return {
            "type": "search",
            "retrieval": found.retrieval,
            "vector_wait_ms": found.vector_wait_ms,
            "index": self._index(),
            "hits": [
                self.hit(candidate, row)
                for candidate, row in zip(found.candidates, found.rows, strict=True)
            ],
        }

    def answer_fields(self, answer: Answer, rows: Mapping[str, int]) -> dict[str, Any]:
        """An answer as the page reads it: who selected, why, and the quotations."""
        selected = answer.selected
        return {
            "selector": selected.selector,
            "model": selected.model if selected.selector == "claude" else None,
            "fallback": selected.fallback,
            "answerable": answer.answerable,
            "quotes": [self.quote(quote, rows[quote.passage_id]) for quote in answer.quotes],
        }

    def answer_line(self, answer: Answer, found: Found) -> dict[str, Any]:
        """The line with the answer step's quotations."""
        rows = {
            candidate.passage_id: row
            for candidate, row in zip(found.candidates, found.rows, strict=True)
        }
        return {"type": "answer", **self.answer_fields(answer, rows)}

    def parties_line(
        self,
        parties: Sequence[tuple[str, Answer, Mapping[str, int]]],
        retrieval: Retrieval,
        waited: int | None,
    ) -> dict[str, Any]:
        """The line with the view of the eight parties: one answer per party."""
        return {
            "type": "parties",
            "retrieval": retrieval,
            "vector_wait_ms": waited,
            "index": self._index(),
            "parties": [
                {"party": party, **self.answer_fields(answer, rows)}
                for party, answer, rows in parties
            ],
        }
