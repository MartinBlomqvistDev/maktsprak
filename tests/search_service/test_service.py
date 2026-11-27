"""Tests for search_service/service.py, on a package of the kit's corpus."""

from __future__ import annotations

import asyncio
import json
import time
import unicodedata
from datetime import date

import numpy as np
import pytest

from search.answer.models import Selected
from search.answer.selector import ABSTENTION, selection_output
from search.eval.generation import passage_span
from search.parties import PARTY_ORDER
from search_service.service import Filters, InvalidRequest, Question, RateLimiter
from search_service.spike.contracts import PartyViewRequest, SearchFilter, SearchRequest
from tests.search_service.package_kit import FakeSelector, build_kit_package, kit_service
from tests.search_service.spike_kit import SPEECHES, _speech, write_spike_corpus

#: A question with words of several of the kit's passages.
QUESTION = "skolan skolor jobb el sjukvården vårdköerna"

HIT_FIELDS = {
    "rank",
    "passage_id",
    "anf_id",
    "rm",
    "dok_nummer",
    "party",
    "speaker",
    "date",
    "title",
    "is_reply",
    "url",
    "paragraphs",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = write_spike_corpus(tmp_path_factory.mktemp("kit"))
    return build_kit_package(corpus, tmp_path_factory.mktemp("built"))


@pytest.fixture
def service(built):
    directory, manifest, _, matrix, _ = built
    return kit_service(directory, manifest, {QUESTION: matrix[4]})


def ask(question=QUESTION, **filters):
    return Question(question=question, filters=Filters(**filters))


class TestRead:
    def test_the_question_is_trimmed(self, service):
        text, search_filter = service.read(ask(f"  {QUESTION} "))
        assert text == QUESTION and search_filter.empty

    def test_a_decomposed_question_is_read_composed(self, service):
        composed = "Kärnkraften på Ringhals och vårdköerna"
        decomposed = unicodedata.normalize("NFD", composed)
        assert decomposed != composed
        text, _ = service.read(ask(decomposed))
        assert text == composed
        found = service.find(text, SearchFilter(), None, None)
        assert found.candidates == service.find(composed, SearchFilter(), None, None).candidates

    @pytest.mark.parametrize("question", ["", "   ", "x" * 401])
    def test_an_empty_or_long_question_is_refused(self, service, question):
        with pytest.raises(InvalidRequest):
            service.read(ask(question))

    def test_years_become_the_first_and_last_day(self, service, built):
        manifest = built[1]
        first, last = manifest.first_date.year, manifest.last_date.year
        _, search_filter = service.read(ask(party="S", year_from=first, year_to=last))
        assert (search_filter.party, search_filter.start, search_filter.end) == (
            "S",
            date(first, 1, 1),
            date(last, 12, 31),
        )

    @pytest.mark.parametrize(
        "filters",
        [
            {"party": "FP"},
            {"year_from": 1990},
            {"year_to": 2100},
            {"year_from": 2023, "year_to": 2004},
        ],
    )
    def test_a_filter_outside_the_page_is_refused(self, service, filters):
        with pytest.raises(InvalidRequest):
            service.read(ask(**filters))

    @pytest.mark.parametrize("filters", [{"party": "S"}, {"speaker": "123"}])
    def test_the_view_of_the_parties_names_no_party_or_speaker(self, service, filters):
        with pytest.raises(InvalidRequest):
            service.read(ask(**filters), parties=True)

    def test_the_riksdags_placeholder_id_is_no_speaker(self, service):
        with pytest.raises(InvalidRequest, match="placeholder"):
            service.read(ask(speaker="0000000000000"))
        _, search_filter = service.read(ask(speaker="0397205342021"))
        assert search_filter.person_id == "0397205342021"


class TestVector:
    def test_the_vector_comes_with_its_wait(self, service, built):
        vector, waited = asyncio.run(service.vector(QUESTION))
        np.testing.assert_array_equal(vector, built[3][4])
        assert isinstance(waited, int) and waited >= 0

    def test_a_vector_later_than_the_wait_is_not_waited_for(self, built):
        directory, manifest, _, matrix, _ = built
        slow = kit_service(directory, manifest, {QUESTION: matrix[4]}, delay_s=0.5, wait_s=0.05)
        assert asyncio.run(slow.vector(QUESTION)) == (None, None)
        assert slow.counters["vector_late"] == 1

    def test_a_failed_api_or_a_wrong_vector_means_none(self, built):
        directory, manifest, _, matrix, _ = built
        service = kit_service(directory, manifest, {"kort": matrix[0][:4]})
        assert asyncio.run(service.vector("okänd")) == (None, None)
        assert asyncio.run(service.vector("kort")) == (None, None)
        assert service.counters["vector_failed"] == 2


class TestFind:
    def test_the_hybrid_ranks_as_the_store_and_carries_bm25_scores(self, service, built):
        vector = built[3][4]
        found = service.find(QUESTION, SearchFilter(), vector, 12)
        ranked = service.store.rank(SearchRequest(text=QUESTION, vector_seed=0), vector)
        scores = service.store.lexical.scores(QUESTION, service.params)
        assert [c.passage_id for c in found.candidates] == [
            service.store.passage_ids[position] for position, _ in ranked.ranked
        ]
        assert [c.score for c in found.candidates] == [
            float(scores[position]) for position, _ in ranked.ranked
        ]
        assert (found.retrieval, found.vector_wait_ms) == ("hybrid", 12)

    def test_without_a_vector_bm25_alone_ranks(self, service):
        found = service.find(QUESTION, SearchFilter(), None, None)
        hits = service.store.lexical.search(QUESTION, service.params, limit=3)
        assert [c.passage_id for c in found.candidates] == [
            service.store.passage_ids[hit.passage] for hit in hits
        ]
        assert (found.retrieval, found.vector_wait_ms) == ("bm25", None)

    def test_the_store_never_searches_without_the_questions_vector(self, service):
        with pytest.raises(ValueError, match="own vector"):
            service.store.query(0, None, QUESTION)

    def test_a_candidate_is_its_passage_of_its_speech(self, service, built):
        found = service.find(QUESTION, SearchFilter(), built[3][4], 1)
        for candidate, row in zip(found.candidates, found.rows, strict=True):
            speech = found.speeches[candidate.anf_id]
            assert speech.paragraphs[candidate.first : candidate.last + 1] == candidate.paragraphs
            assert service.speeches.fields(row)["anf_id"] == candidate.anf_id

    def test_a_filter_applies_to_both_legs(self, service, built):
        found = service.find(QUESTION, SearchFilter(party="M"), built[3][4], 1)
        assert found.candidates and {c.party for c in found.candidates} == {"M"}

    def test_the_corpus_is_scored_once_a_search(self, service, built, monkeypatch):
        lexical = service.store.lexical
        calls = []
        scores = lexical.scores
        monkeypatch.setattr(lexical, "scores", lambda *args: calls.append(1) or scores(*args))
        service.find(QUESTION, SearchFilter(), built[3][4], 1)
        service.find(QUESTION, SearchFilter(party="S"), None, None)
        assert len(calls) == 2

    @pytest.mark.parametrize("party", [None, "S", "M"])
    @pytest.mark.parametrize("question", [QUESTION, "skolan", "kärnkraften el", "ingenting alls"])
    def test_the_lexical_leg_ranks_as_the_tuned_bm25_search(self, service, question, party):
        store = service.store
        allowed = store.mask(SearchFilter(party=party))
        leg, _ = store._lexical_leg(question, allowed)
        hits = store.lexical.search(
            question, service.params, limit=store.settings.depth, allowed=allowed
        )
        assert leg == [hit.passage for hit in hits]


#: A speech of three long paragraphs, which the kit packs into three passages.
LONG = _speech(
    "H209999",
    4,
    [
        f"Stycke {number} om vindkraften och elnäten. " + " ".join([f"ord{number}"] * 150)
        for number in (1, 2, 3)
    ],
    "C",
    date(2021, 5, 4),
    "0909",
)


@pytest.fixture(scope="module")
def long_built(tmp_path_factory):
    corpus = write_spike_corpus(tmp_path_factory.mktemp("long"), [*SPEECHES, LONG])
    return build_kit_package(corpus, tmp_path_factory.mktemp("long-built"))


class TestSpeechOfSeveralPassages:
    def later_passage(self, ids):
        passages = [passage for passage in ids if passage.startswith(f"{LONG.anf_id}:")]
        assert len(passages) >= 2
        return next(passage for passage in passages if passage_span(passage)[1] > 0)

    def test_the_table_joins_the_speech_from_its_passages(self, long_built):
        directory, manifest, ids, _, _ = long_built
        service = kit_service(directory, manifest, {})
        position = ids.index(self.later_passage(ids))
        speech = service.speeches.speech(service.speeches.row_of(position))
        assert (speech.anf_id, list(speech.paragraphs)) == (LONG.anf_id, list(LONG.paragraphs))

    def test_a_quotation_from_a_later_passage_is_the_speechs_own_paragraph(self, long_built):
        directory, manifest, ids, _, _ = long_built
        later = self.later_passage(ids)
        _, first, _ = passage_span(later)
        selector = FakeSelector(selection_output(later, 1, 1))
        service = kit_service(directory, manifest, {}, selector=selector)
        found = service.find("vindkraften elnäten", SearchFilter(party="C"), None, None)
        assert later in [candidate.passage_id for candidate in found.candidates]
        answer = service.answer("vindkraften elnäten", found)
        (quote,) = answer.quotes
        assert list(quote.paragraphs) == [LONG.paragraphs[first]]
        line = service.answer_line(answer, found)
        assert line["quotes"][0]["paragraphs"] == [LONG.paragraphs[first]]

    def test_a_range_past_its_passage_quotes_nothing(self, long_built):
        directory, manifest, ids, _, _ = long_built
        later = self.later_passage(ids)
        _, first, last = passage_span(later)
        beyond = selection_output(later, 1, last - first + 2)
        service = kit_service(directory, manifest, {}, selector=FakeSelector(beyond))
        found = service.find("vindkraften elnäten", SearchFilter(party="C"), None, None)
        assert list(service.answer("vindkraften elnäten", found).quotes) == []


class TestAnswerInTime:
    def test_past_the_deadline_the_baseline_answers(self, built):
        directory, manifest, _, matrix, _ = built
        selector = FakeSelector(delay_s=0.6)
        service = kit_service(directory, manifest, {}, selector=selector, answer_deadline_s=0.2)
        found = service.find(QUESTION, SearchFilter(), matrix[4], 1)
        answer = asyncio.run(service.answer_in_time(QUESTION, found))
        assert (answer.selected.selector, answer.selected.fallback) == ("baseline", "failed")
        assert service.counters["answer_late"] == 1
        service.close()

    def test_a_question_that_waits_past_the_deadline_is_never_sent(self, built):
        directory, manifest, _, matrix, _ = built
        selector = FakeSelector(delay_s=0.5)
        service = kit_service(directory, manifest, {}, selector=selector, answer_deadline_s=0.2)
        found = service.find(QUESTION, SearchFilter(), matrix[4], 1)

        async def two():
            return await asyncio.gather(
                service.answer_in_time(QUESTION, found), service.answer_in_time(QUESTION, found)
            )

        answers = asyncio.run(two())
        time.sleep(0.6)
        assert [answer.selected.fallback for answer in answers] == ["failed", "failed"]
        assert len(selector.asked) == 1
        service.close()

    def test_within_the_deadline_the_selector_answers(self, service, built):
        found = service.find(QUESTION, SearchFilter(), built[3][4], 1)
        answer = asyncio.run(service.answer_in_time(QUESTION, found))
        assert (answer.selected.selector, answer.selected.fallback) == ("claude", None)


class TestLines:
    def test_the_search_line_is_the_pages_contract(self, service, built):
        found = service.find(QUESTION, SearchFilter(), built[3][4], 7)
        line = json.loads(json.dumps(service.search_line(found)))
        assert set(line) == {"type", "retrieval", "vector_wait_ms", "index", "hits"}
        assert line["index"] == {
            "last_date": built[1].last_date.isoformat(),
            "speeches": built[1].speeches,
            "passages": built[1].passages,
        }
        hit = line["hits"][0]
        assert set(hit) == HIT_FIELDS
        assert hit["rank"] == 1 and isinstance(hit["is_reply"], bool)
        assert date.fromisoformat(hit["date"]) and isinstance(hit["dok_nummer"], int)

    def test_the_answer_line_quotes_with_the_hits_fields(self, service, built):
        found = service.find(QUESTION, SearchFilter(), built[3][4], 7)
        line = service.answer_line(service.answer(QUESTION, found), found)
        assert (line["type"], line["selector"], line["model"], line["fallback"]) == (
            "answer",
            "claude",
            "claude-opus-5-5",
            None,
        )
        quote = line["quotes"][0]
        assert set(quote) == HIT_FIELDS | {"first_paragraph", "last_paragraph"}
        assert quote["paragraphs"] == [found.candidates[0].paragraphs[0]]

    def test_a_selection_outside_the_candidates_quotes_nothing(self, built):
        directory, manifest, _, matrix, _ = built
        wrong = selection_output("XX0000-1:0-0", 1, 1)
        service = kit_service(directory, manifest, {}, selector=FakeSelector(wrong))
        found = service.find(QUESTION, SearchFilter(), matrix[4], 1)
        line = service.answer_line(service.answer(QUESTION, found), found)
        assert (line["answerable"], line["quotes"]) == (True, [])

    def test_the_baseline_names_no_model_and_a_fallback_is_counted(self, built):
        directory, manifest, _, matrix, _ = built

        class Budgeted(FakeSelector):
            def select(self, question, candidates):
                return Selected(selector="baseline", output=ABSTENTION, fallback="budget")

        service = kit_service(directory, manifest, {}, selector=Budgeted())
        found = service.find(QUESTION, SearchFilter(), matrix[4], 1)
        line = service.answer_line(service.answer(QUESTION, found), found)
        assert (line["selector"], line["model"], line["fallback"], line["quotes"]) == (
            "baseline",
            None,
            "budget",
            [],
        )
        assert service.counters["fallback_budget"] == 1


class TestParties:
    def test_each_party_gets_the_passage_the_stores_view_gives_it(self, service, built):
        vector = built[3][4]
        best, _ = service.party_best(QUESTION, SearchFilter(), vector)
        view = service.store.party_view(PartyViewRequest(text=QUESTION, vector_seed=0), vector).hits
        for party in PARTY_ORDER:
            expected = view[party].passage_id if view[party] is not None else None
            got = service.store.passage_ids[best[party]] if best[party] is not None else None
            assert got == expected

    def test_each_partys_passage_is_the_hybrids_best_under_its_party(self, service, built):
        vector = built[3][4]
        best, _ = service.party_best(QUESTION, SearchFilter(), vector)
        for party in PARTY_ORDER:
            request = SearchRequest(text=QUESTION, vector_seed=0, filter=SearchFilter(party=party))
            ranked = service.store.rank(request, vector).ranked
            assert best[party] == (ranked[0][0] if ranked else None)

    def test_the_line_holds_one_answer_per_party_in_the_sites_order(self, service):
        result = service.parties("skolan", SearchFilter(), None)
        line = service.parties_line(result, "bm25", None)
        assert set(line) == {"type", "retrieval", "vector_wait_ms", "index", "parties"}
        assert [entry["party"] for entry in line["parties"]] == list(PARTY_ORDER)
        for entry in line["parties"]:
            assert set(entry) == {"party", "selector", "model", "fallback", "answerable", "quotes"}
            assert (entry["selector"], entry["model"], entry["fallback"]) == (
                "baseline",
                None,
                None,
            )
            assert entry["answerable"] == bool(entry["quotes"]) and len(entry["quotes"]) <= 1
            assert all(quote["party"] == entry["party"] for quote in entry["quotes"])
        assert any(entry["quotes"] for entry in line["parties"])

    def test_a_party_whose_best_passage_is_below_the_threshold_abstains(self, service):
        result = service.parties(QUESTION, SearchFilter(), None)
        assert all(answer.answerable is False and not answer.quotes for _, answer, _ in result)

    def test_without_a_vector_bm25_alone_ranks_each_party(self, service):
        best, scores = service.party_best(QUESTION, SearchFilter(), None)
        assert all(scores[position] > 0 for position in best.values() if position is not None)


class TestRateLimiter:
    def test_a_caller_waits_once_the_window_is_full(self):
        now = [0.0]
        rates = RateLimiter(60, {"search": 2, "answer": 1}, clock=lambda: now[0])
        assert rates.check("a", "search") is None and rates.check("a", "search") is None
        assert rates.check("a", "search") == pytest.approx(60.0)
        assert rates.check("b", "search") is None
        assert rates.check("a", "answer") is None
        now[0] = 61.0
        assert rates.check("a", "search") is None

    def test_a_caller_is_forgotten_within_two_windows(self):
        now = [0.0]
        rates = RateLimiter(60, {"search": 2, "answer": 1}, clock=lambda: now[0])
        rates.check("a", "search")
        now[0] = 70.0
        rates.check("b", "search")
        now[0] = 121.0
        rates.check("c", "search")
        assert len(rates) == 2  # a is forgotten; b's and c's windows have not passed
        now[0] = 182.0
        rates.tidy()
        assert len(rates) == 0

    def test_no_more_than_the_most_callers_are_held(self):
        rates = RateLimiter(60, {"search": 2, "answer": 1}, max_callers=10)
        for number in range(25):
            rates.check(f"caller-{number}", "search")
        assert len(rates) <= 10
        rates.check("caller-24", "search")
        assert rates.check("caller-24", "search") is not None  # the latest is still counted

    def test_a_caller_is_kept_as_a_salted_hash(self):
        rates = RateLimiter(60, {"search": 1, "answer": 1})
        rates.check("203.0.113.9", "search")
        assert all("203.0.113.9" not in key for key, _ in rates._seen)
        other = RateLimiter(60, {"search": 1, "answer": 1})
        other.check("203.0.113.9", "search")
        assert set(rates._seen) != set(other._seen)
