"""Tests for search_service/spike/workload.py, on the kit's questions."""

from __future__ import annotations

from datetime import date

import pytest

from search.eval.questions import QuestionFilter
from search.parties import PARTY_ORDER
from search_service.spike.inputs import CorpusFiles, read_passage_columns
from search_service.spike.workload import (
    build_workload,
    own_filter,
    period_filter,
    vector_seed,
)
from tests.search_service.spike_kit import QUESTIONS, SPEECHES, kit_settings, write_spike_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_spike_corpus(tmp_path_factory.mktemp("spike"))


@pytest.fixture(scope="module")
def columns(corpus):
    return read_passage_columns(CorpusFiles.under(corpus.search_dir, corpus.tuning).passages)


@pytest.fixture(scope="module")
def workload(corpus, columns):
    return build_workload(kit_settings().workload, corpus.reports_dir / "questions", columns)


def split_of_question(question_id: str) -> str:
    return next(q.split for qs in QUESTIONS.values() for q in qs if q.question_id == question_id)


class TestWorkload:
    def test_it_holds_the_development_split_only(self, workload):
        every = workload.queries + workload.warmup + workload.party_view
        assert every
        assert all(split_of_question(query.query_id) == "dev" for query in every)

    def test_every_dev_question_of_the_sets_is_sent_once(self, workload):
        expected = {
            q.question_id
            for name in kit_settings().workload.sets
            for q in QUESTIONS[name]
            if q.split == "dev"
        }
        sent = [query.query_id for query in workload.queries]
        assert sorted(sent) == sorted(expected)

    def test_a_filtered_question_keeps_its_own_filter(self, workload):
        query = next(q for q in workload.queries if q.query_id == "filtered-0001")
        assert query.filter_kind == "own"
        assert query.filter.party == "M"

    def test_every_other_question_gets_a_filter_that_allows_a_passage(self, workload):
        for query in workload.queries:
            if query.filter_kind == "own":
                continue
            assert not query.filter.empty
            assert query.filter.party in (None, *PARTY_ORDER)
            if query.filter_kind == "speaker":
                assert query.filter.person_id in {s.person_id for s in SPEECHES}
            if query.filter_kind == "party_period":
                assert any(
                    s.party == query.filter.party
                    and query.filter.start <= s.date <= query.filter.end
                    for s in SPEECHES
                )

    def test_the_same_seed_gives_the_same_workload(self, corpus, columns, workload):
        again = build_workload(kit_settings().workload, corpus.reports_dir / "questions", columns)
        assert again == workload

    def test_the_warmup_and_the_party_view_have_their_sizes(self, workload):
        assert len(workload.warmup) == 2
        assert all(q.question_set == "interpellation" for q in workload.warmup)
        assert len(workload.party_view) == 2
        assert all(q.filter.empty and q.question_set == "natural" for q in workload.party_view)


class TestFilters:
    def test_a_period_runs_from_new_year_to_new_year_eve(self):
        middle = period_filter(2010, 2026)
        assert (middle.start, middle.end) == (date(2010, 1, 1), date(2017, 12, 31))
        last = period_filter(2018, 2026)
        assert (last.start, last.end) == (date(2018, 1, 1), date(2026, 12, 31))

    def test_a_questions_own_period_becomes_dates(self):
        question = QUESTIONS["filtered"][1]
        assert question.filter == QuestionFilter(period="2018-2023")
        search_filter = own_filter(question)
        assert (search_filter.start, search_filter.end) == (date(2018, 1, 1), date(2023, 12, 31))

    def test_a_question_without_a_filter_has_no_own_filter(self):
        with pytest.raises(ValueError, match="no filter"):
            own_filter(QUESTIONS["natural"][0])

    def test_a_vector_seed_is_stable_and_differs_by_question(self):
        assert vector_seed("natural-0001", 1) == vector_seed("natural-0001", 1)
        assert vector_seed("natural-0001", 1) != vector_seed("natural-0002", 1)
