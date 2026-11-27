"""The service's lines against the page's contract, search/page/contract.py.

The contract comes with the page; until both are in one tree these tests are
skipped, and from then on any line the service writes outside the contract
fails here before it reaches the page.
"""

from __future__ import annotations

import json

import pytest

contract = pytest.importorskip("search.page.contract")

from search.answer.models import Selected  # noqa: E402
from search.answer.selector import ABSTENTION  # noqa: E402
from search_service.spike.contracts import SearchFilter  # noqa: E402
from tests.search_service.package_kit import (  # noqa: E402
    FakeSelector,
    build_kit_package,
    kit_service,
)
from tests.search_service.spike_kit import write_spike_corpus  # noqa: E402

QUESTION = "skolan skolor jobb el sjukvården vårdköerna"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = write_spike_corpus(tmp_path_factory.mktemp("kit"))
    return build_kit_package(corpus, tmp_path_factory.mktemp("built"))


def through_json(line):
    """The line as the page reads it, after JSON."""
    return json.loads(json.dumps(line, ensure_ascii=False))


@pytest.mark.parametrize("vector", [True, False])
def test_the_search_line_is_the_contracts(built, vector):
    directory, manifest, _, matrix, _ = built
    service = kit_service(directory, manifest, {})
    found = service.find(QUESTION, SearchFilter(), matrix[4] if vector else None, 3)
    line = contract.SearchLine.model_validate(through_json(service.search_line(found)))
    assert line.hits and line.retrieval == ("hybrid" if vector else "bm25")


def test_the_answer_line_is_the_contracts(built):
    directory, manifest, _, matrix, _ = built
    service = kit_service(directory, manifest, {})
    found = service.find(QUESTION, SearchFilter(), matrix[4], 3)
    line = service.answer_line(service.answer(QUESTION, found), found)
    assert contract.AnswerLine.model_validate(through_json(line)).quotes


def test_the_baseline_in_place_of_the_model_is_the_contracts(built):
    directory, manifest, _, matrix, _ = built

    class Budgeted(FakeSelector):
        def select(self, question, candidates):
            return Selected(selector="baseline", output=ABSTENTION, fallback="budget")

    service = kit_service(directory, manifest, {}, selector=Budgeted())
    found = service.find(QUESTION, SearchFilter(), matrix[4], 3)
    line = service.answer_line(service.answer(QUESTION, found), found)
    assert contract.AnswerLine.model_validate(through_json(line)).fallback == "budget"


def test_the_parties_line_is_the_contracts(built):
    directory, manifest, *_ = built
    service = kit_service(directory, manifest, {})
    line = service.parties_line(service.parties("skolan", SearchFilter(), None), "bm25", None)
    assert len(contract.PartiesLine.model_validate(through_json(line)).parties) == 8
