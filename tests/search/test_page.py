"""Tests for the page's static data: the contract, the speaker list and the method figures.

The method figures are read from the tracked reports, so the test that compares
them with the committed ``method.json`` fails as soon as a report changes and
the page's file has not been written again.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from search.answer.models import Answer, Candidate, Rejected, Selected
from search.answer.models import Quote as AssembledQuote
from search.config import REPO_ROOT, load_settings
from search.page.contract import (
    ErrorLine,
    PartiesLine,
    PartyAnswer,
    Quote,
    SpeechFacts,
    answer_line,
    hit_of,
    party_answer,
)
from search.page.method import method_figures
from search.page.speakers import (
    SpeechRow,
    bare_name,
    display_name,
    is_capitals,
    is_placeholder,
    speakers,
    title_case,
)
from search.parties import PARTY_ORDER

FACTS = {"HC09139-36": SpeechFacts(rm="2024/25", dok_nummer=139, is_reply=False)}
URL = "https://data.riksdagen.se/anforande/HC09139-36/html"


def load_step():
    """Import step 15, whose name is not a module name, registered before it runs."""
    path = REPO_ROOT / "search" / "pipeline" / "15_build_page.py"
    spec = importlib.util.spec_from_file_location("build_page", path)
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


step = load_step()


def candidate(rank: int = 1) -> Candidate:
    return Candidate(
        rank=rank,
        passage_id="HC09139-36:0-2",
        anf_id="HC09139-36",
        first=0,
        last=2,
        score=3.5,
        party="L",
        speaker="Arbetsmarknadsministern Johan Britz (L)",
        date=date(2025, 9, 4),
        title="Svar på interpellation 2024/25:734 om arbetsvillkor inom gigekonomin",
        url=URL,
        paragraphs=("Fru talman!", "Andra stycket.", "Tredje stycket."),
        text_sha256="0" * 64,
    )


def assembled(first: int, last: int, party: str = "L") -> AssembledQuote:
    paragraphs = ("Fru talman!", "Andra stycket.", "Tredje stycket.")[first - 1 : last]
    return AssembledQuote(
        passage_id="HC09139-36:0-2",
        rank=1,
        first_paragraph=first,
        last_paragraph=last,
        anf_id="HC09139-36",
        party=party,
        speaker="Arbetsmarknadsministern Johan Britz (L)",
        date=date(2025, 9, 4),
        title=None,
        url=URL,
        paragraphs=paragraphs,
    )


def answer(selector: str = "claude", quotes=(), fallback=None) -> Answer:
    return Answer(
        selected=Selected(
            selector=selector,
            output="{}",
            fallback=fallback,
            model="claude-opus-5-5",
        ),
        answerable=bool(quotes),
        quotes=tuple(quotes),
        rejected=(Rejected(position=None, reason="malformed"),) if not quotes else (),
    )


# --- The contract ---------------------------------------------------------------


def test_a_hit_carries_the_candidates_paragraphs_and_the_speechs_facts():
    hit = hit_of(candidate(), FACTS)
    assert hit.paragraphs == candidate().paragraphs
    assert (hit.rm, hit.dok_nummer, hit.is_reply) == ("2024/25", 139, False)
    assert hit.url == URL


def test_an_answer_line_keeps_the_assemblers_paragraphs():
    line = answer_line(answer(quotes=[assembled(2, 3)]), FACTS)
    assert line.quotes[0].paragraphs == ("Andra stycket.", "Tredje stycket.")
    assert (line.selector, line.model) == ("claude", "claude-opus-5-5")


def test_the_baseline_is_shown_without_a_model():
    line = answer_line(
        answer(selector="baseline", quotes=[assembled(1, 1)], fallback="budget"), FACTS
    )
    assert (line.selector, line.model, line.fallback) == ("baseline", None, "budget")


def test_a_quote_must_fill_its_range():
    fields = hit_of(candidate(), FACTS).model_dump()
    with pytest.raises(ValidationError):
        Quote(**{**fields, "first_paragraph": 1, "last_paragraph": 2})


def test_the_parties_line_holds_every_party_in_order():
    parties = tuple(party_answer(party, answer(), FACTS) for party in PARTY_ORDER)
    index = {"last_date": "2026-11-18", "speeches": 1, "passages": 1}
    line = PartiesLine(retrieval="hybrid", vector_wait_ms=1200, index=index, parties=parties)
    assert [item.party for item in line.parties] == list(PARTY_ORDER)
    with pytest.raises(ValidationError):
        PartiesLine(retrieval="hybrid", vector_wait_ms=None, index=index, parties=parties[::-1])


def test_a_party_holds_only_its_own_quotations():
    with pytest.raises(ValidationError):
        PartyAnswer(
            party="S",
            selector="claude",
            model=None,
            fallback=None,
            answerable=True,
            quotes=(quote_of_l(),),
        )


def quote_of_l() -> Quote:
    return answer_line(answer(quotes=[assembled(1, 1)]), FACTS).quotes[0]


# --- The speakers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("published", "name"),
    [
        ("Statsminister GÖRAN PERSSON (s) replik", "GÖRAN PERSSON"),
        ("Arbetsmarknadsministern Johan Britz (L)", "Johan Britz"),
        ("Utbildnings- och kulturminister LEIF PAGROTSKY (s)", "LEIF PAGROTSKY"),
        ("Minister för civilt försvar CARL-OSKAR BOHLIN (M)", "CARL-OSKAR BOHLIN"),
        ("Förste vice talmannen Kerstin Lundgren (C)", "Kerstin Lundgren"),
        ("GÖRAN PERSSON i Stjärnhov (s)", "GÖRAN PERSSON i Stjärnhov"),
        ("Carl B Hamilton (L) replik", "Carl B Hamilton"),
    ],
)
def test_the_bare_name_drops_title_party_and_reply(published, name):
    assert bare_name(published) == name


def test_capitals_are_told_from_mixed_case():
    assert is_capitals("GÖRAN PERSSON i Stjärnhov")
    assert not is_capitals("Carl B Hamilton")


def test_title_case_keeps_particles_and_hyphens():
    assert title_case("ANN-CHRISTIN AHLBERG") == "Ann-Christin Ahlberg"
    assert title_case("ANNA VON SYDOW") == "Anna von Sydow"
    assert title_case("GÖRAN PERSSON i Stjärnhov") == "Göran Persson i Stjärnhov"


def test_a_name_in_mixed_case_is_preferred():
    forms = {"STAFFAN DANIELSSON (c)": 390, "Staffan Danielsson (C)": 143}
    assert display_name(forms) == "Staffan Danielsson"


def test_a_name_only_in_capitals_is_put_in_title_case():
    assert display_name({"Statsminister GÖRAN PERSSON (s)": 643}) == "Göran Persson"


def test_placeholder_ids_name_nobody():
    assert is_placeholder("0000000000000")
    assert is_placeholder(None)
    assert not is_placeholder("0397205342021")


def test_the_list_has_one_entry_per_person_sorted_by_name():
    rows = [
        SpeechRow(person_id="2", speaker_raw="Bo Ek (S)", party="S", date=date(2010, 1, 1)),
        SpeechRow(person_id="2", speaker_raw="BO EK (s)", party="S", date=date(2005, 1, 1)),
        SpeechRow(person_id="1", speaker_raw="Åsa Al (M)", party="M", date=date(2020, 1, 1)),
        SpeechRow(person_id="1", speaker_raw="Åsa Al (SD)", party="SD", date=date(2024, 1, 1)),
        SpeechRow(
            person_id="0000000000000", speaker_raw="X Y (V)", party="V", date=date(2004, 1, 1)
        ),
        SpeechRow(person_id=None, speaker_raw="Z W (V)", party="V", date=date(2004, 1, 1)),
    ]
    entries = speakers(rows, PARTY_ORDER)
    assert [entry.id for entry in entries] == ["2", "1"]
    assert (entries[0].name, entries[0].first, entries[0].last, entries[0].speeches) == (
        "Bo Ek",
        2005,
        2010,
        2,
    )
    assert entries[1].parties == ("M", "SD")


# --- The files the site reads ----------------------------------------------------


def test_method_json_matches_the_reports():
    figures = method_figures(load_settings())
    committed = json.loads(step.METHOD_JSON.read_text(encoding="utf-8"))
    assert committed == json.loads(figures.model_dump_json())


def test_every_example_file_follows_the_contract():
    listed = json.loads(step.EXAMPLES_JSON.read_text(encoding="utf-8"))
    assert listed["examples"], "the page needs at least one example"
    for entry in listed["examples"]:
        example = step.Example.model_validate_json(
            (step.EXAMPLE_DIR / f"{entry['id']}.json").read_text(encoding="utf-8")
        )
        assert (example.id, example.question) == (entry["id"], entry["question"])
        assert example.search.hits, f"{entry['id']} has no results"


def test_the_examples_are_the_listed_questions():
    examples = step.load_examples()
    listed = json.loads(step.EXAMPLES_JSON.read_text(encoding="utf-8"))
    assert [(item.id, item.text) for item in examples.questions] == [
        (entry["id"], entry["question"]) for entry in listed["examples"]
    ]


def test_the_speaker_list_names_nobody_twice_and_no_placeholder():
    entries = json.loads(step.SPEAKERS_JSON.read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    assert not any(is_placeholder(person) for person in ids)


def test_a_request_id_names_the_example_and_the_filter():
    assert step.custom_id("nato", None) == "nato--alla"
    assert step.custom_id("nato", "KD") == "nato--KD"


def test_paths_point_into_the_site():
    web = Path(REPO_ROOT) / "web"
    for path in (step.EXAMPLES_JSON, step.METHOD_JSON, step.SPEAKERS_JSON, step.EXAMPLE_DIR):
        assert web in path.parents


def test_the_error_line_is_the_one_the_service_sends():
    line = ErrorLine.model_validate({"type": "error", "error": "answer_failed"})
    assert line.model_dump() == {"type": "error", "error": "answer_failed"}
    with pytest.raises(ValidationError):
        ErrorLine.model_validate({"type": "error", "error": "something_else"})
