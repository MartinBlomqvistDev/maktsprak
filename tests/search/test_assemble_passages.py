"""Tests for the assembler on a speech packed into several passages.

The other corpora of the tests pack each speech into one passage, which starts
at the speech's first paragraph. Most quotations in a real answer come from a
passage that starts later, so these tests quote from such passages: the
passage's offset in its speech, a range that would run on into the next
passage, and selections from different passages of one speech.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from search.answer.assemble import assemble
from search.answer.candidates import build_candidates, load_speeches
from search.answer.models import Candidate, Selected, Speech
from search.corpus.passages import pack
from search.corpus.store import PASSAGES_NAME, SPEECHES_NAME, write_passages, write_speeches
from search.corpus.table import load_passage_table
from search.index.lexical import Hit
from search.models import passage_id
from tests.search.corpus_kit import speech

#: Six paragraphs of five words each. Packed towards ten words, they make three
#: passages of two paragraphs each.
LONG = speech(
    "H909100",
    1,
    [
        "Skatten på arbete ska sänkas.",
        "Jobbskatteavdraget har gett fler jobb.",
        "Kärnkraften behövs för industrins elförsörjning.",
        "Vattenkraften räcker inte till ensam.",
        "Skolan behöver fler behöriga lärare.",
        "Lärarna ska få högre lön.",
    ],
    party="M",
)
FIRST, MIDDLE, LAST = (passage_id(LONG.anf_id, first, first + 1) for first in (0, 2, 4))


class Kit:
    """The speech on disk, packed, and its three passages as one question's candidates."""

    def __init__(self, directory: Path) -> None:
        speeches_path, passages_path = directory / SPEECHES_NAME, directory / PASSAGES_NAME
        write_speeches(speeches_path, [[LONG]])
        write_passages(passages_path, [list(pack(LONG, 10))])
        table = load_passage_table(passages_path)
        assert list(table.passage_ids) == [FIRST, MIDDLE, LAST]
        self.speeches: dict[str, Speech] = load_speeches(speeches_path, [LONG.anf_id])
        hits = [
            Hit(passage=table.passage_ids.index(key), score=1.0) for key in (MIDDLE, LAST, FIRST)
        ]
        self.candidates: list[Candidate] = build_candidates(hits, table, self.speeches)

    def answer(self, *selections: tuple[str, int, int], candidates=None, speeches=None):
        output = {
            "answerable": True,
            "selections": [
                {"passage_id": passage, "first_paragraph": first, "last_paragraph": last}
                for passage, first, last in selections
            ],
        }
        return assemble(
            Selected(selector="claude", output=json.dumps(output)),
            self.candidates if candidates is None else candidates,
            self.speeches if speeches is None else speeches,
        )


@pytest.fixture
def kit(tmp_path: Path) -> Kit:
    return Kit(tmp_path)


def spanning(speech: Speech, first: int, last: int, rank: int) -> Candidate:
    """A candidate for paragraphs *first* to *last* of *speech*, as another packing would cut it."""
    return Candidate(
        rank=rank,
        passage_id=passage_id(speech.anf_id, first, last),
        anf_id=speech.anf_id,
        first=first,
        last=last,
        score=1.0,
        party=speech.party,
        speaker=speech.speaker,
        date=speech.date,
        title=speech.title,
        url=speech.url,
        paragraphs=speech.paragraphs[first : last + 1],
        text_sha256=speech.text_sha256,
    )


def test_each_candidate_holds_its_own_paragraphs_of_the_speech(kit):
    by_id = {candidate.passage_id: candidate for candidate in kit.candidates}
    assert (by_id[MIDDLE].first, by_id[MIDDLE].last) == (2, 3)
    assert by_id[MIDDLE].paragraphs == LONG.paragraphs[2:4]
    assert by_id[LAST].paragraphs == LONG.paragraphs[4:6]


def test_a_quotation_is_read_at_the_passages_offset_in_its_speech(kit):
    answer = kit.answer((MIDDLE, 1, 2), (LAST, 2, 2))
    assert answer.rejected == ()
    middle, last = answer.quotes
    assert middle.passage_id == MIDDLE
    assert (middle.first_paragraph, middle.last_paragraph) == (1, 2)
    assert middle.paragraphs == LONG.paragraphs[2:4]
    assert last.paragraphs == (LONG.paragraphs[5],)


@pytest.mark.parametrize(
    ("passage", "first", "last"), [(MIDDLE, 2, 3), (MIDDLE, 1, 3), (FIRST, 2, 3)]
)
def test_a_range_that_runs_on_into_the_next_passage_is_rejected(kit, passage, first, last):
    answer = kit.answer((passage, first, last))
    assert answer.quotes == ()
    assert [item.reason for item in answer.rejected] == ["outside_passage"]


def test_the_same_numbers_in_two_passages_of_one_speech_are_no_overlap(kit):
    answer = kit.answer((FIRST, 1, 1), (MIDDLE, 1, 1), (LAST, 1, 1))
    assert answer.rejected == ()
    assert [quote.paragraphs for quote in answer.quotes] == [
        (LONG.paragraphs[0],),
        (LONG.paragraphs[2],),
        (LONG.paragraphs[4],),
    ]
    adjacent = kit.answer((FIRST, 2, 2), (MIDDLE, 1, 1))
    assert adjacent.rejected == () and len(adjacent.quotes) == 2


def test_an_overlap_is_found_in_the_speech_across_passages(kit):
    wide = spanning(kit.speeches[LONG.anf_id], 0, 3, rank=4)
    candidates = [*kit.candidates, wide]
    answer = kit.answer((wide.passage_id, 3, 4), (MIDDLE, 1, 1), candidates=candidates)
    assert [quote.paragraphs for quote in answer.quotes] == [LONG.paragraphs[2:4]]
    assert [(item.position, item.reason) for item in answer.rejected] == [(1, "overlap")]
    within = kit.answer((MIDDLE, 1, 2), (MIDDLE, 2, 2))
    assert [(item.position, item.reason) for item in within.rejected] == [(1, "overlap")]


def test_a_later_passage_of_a_speech_that_has_since_shrunk_is_stale(kit):
    read = kit.speeches[LONG.anf_id]
    shorter = {LONG.anf_id: read.model_copy(update={"paragraphs": read.paragraphs[:4]})}
    answer = kit.answer((LAST, 1, 1), speeches=shorter)
    assert answer.quotes == ()
    assert [item.reason for item in answer.rejected] == ["stale_text"]
