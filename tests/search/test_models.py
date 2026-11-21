"""Tests for search/models.py."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from search.models import (
    Anforande,
    Passage,
    deep_link,
    passage_id,
    session_bounds,
    session_label,
    session_start_year,
    text_sha256,
)

PARAGRAPHS = ("Fru talman! Tack för svaret.", "Jag yrkar bifall till reservation 3.")


def speech(**changes: object) -> Anforande:
    """Build a valid speech, with any field replaced by *changes*."""
    fields: dict[str, object] = {
        "anf_id": "H209100-7",
        "dok_id": "H209100",
        "rm": "2014/15",
        "dok_nummer": 100,
        "anf_nummer": 7,
        "date": date(2015, 5, 12),
        "speaker_raw": "Anna Andersson (S)",
        "person_id": "0123456789012",
        "party": "S",
        "is_reply": False,
        "title": "Svar på interpellation 2014/15:1 om skolan",
        "activity": "interpellationsdebatt",
        "rel_dok_id": "H21001",
        "layout": "html",
        "paragraphs": PARAGRAPHS,
        "notes": ("(Applåder)",),
        "text_sha256": text_sha256(PARAGRAPHS),
        "url": deep_link("H209100", 7),
    }
    fields.update(changes)
    return Anforande(**fields)


def with_paragraphs(*paragraphs: str, **changes: object) -> Anforande:
    """Build a valid speech whose paragraphs are *paragraphs*, hash included."""
    return speech(paragraphs=paragraphs, text_sha256=text_sha256(paragraphs), **changes)


class TestAValidSpeech:
    def test_can_be_built(self):
        assert speech().anf_id == "H209100-7"

    def test_cannot_be_mutated(self):
        with pytest.raises(ValidationError):
            speech().party = "M"  # type: ignore[misc]

    def test_takes_no_unknown_field(self):
        with pytest.raises(ValidationError):
            speech(summary="Talaren är nöjd.")

    def test_sverigedemokraterna_from_2010_is_allowed(self):
        assert speech(party="SD", rm="2010/11", date=date(2010, 10, 5)).party == "SD"

    def test_a_speech_may_have_no_notes(self):
        assert speech(notes=()).notes == ()

    def test_inner_spaces_and_typographic_quotes_are_kept(self):
        text = "Herr talman!  Jag yrkar ”bifall”."
        assert with_paragraphs(text).paragraphs == (text,)


class TestInvariants:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"anf_id": "H209100-8"}, "key does not match"),
            ({"party": "FP"}, "not one of the eight parties"),
            ({"party": "TALMANNEN"}, "not one of the eight parties"),
            ({"paragraphs": (), "text_sha256": text_sha256(())}, "at least one paragraph"),
            ({"text_sha256": "0" * 64}, "hash does not match"),
            (
                {"party": "SD", "rm": "2009/10", "date": date(2009, 10, 5)},
                "Sverigedemokraterna before 2010",
            ),
            ({"date": date(2016, 5, 12)}, "outside session"),
            ({"date": date(2014, 8, 31)}, "outside session"),
            ({"url": "https://example.org/H209100#anf7"}, "address does not match"),
            (
                {
                    "url": "https://www.riksdagen.se/sv/dokument-och-lagar/dokument/protokoll/"
                    "protokoll_H209100/#anf7"
                },
                "address does not match",
            ),
            (
                {"url": "https://data.riksdagen.se/anforande/H209100-8/html"},
                "address does not match",
            ),
            ({"rm": "2014"}, "Not a session label"),
            ({"rm": "2014/99"}, "Not a session label"),
            ({"rm": "2014-15"}, "Not a session label"),
            ({"rm": ""}, "Not a session label"),
        ],
    )
    def test_a_speech_that_breaks_a_rule_cannot_exist(self, changes, message):
        with pytest.raises(ValidationError, match=message):
            speech(**changes)


class TestSpeaker:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "  ",
            "\t",
            " Anna Andersson (S)",
            "Anna Andersson (S) ",
            "Anna Andersson (S)\n",
            "Anna Andersson\n(S)",
            "\u200b",
            "\xad\ufeff",
        ],
    )
    def test_a_speaker_that_is_blank_split_or_untrimmed_is_refused(self, bad):
        with pytest.raises(ValidationError, match="the speaker must hold a visible character"):
            speech(speaker_raw=bad)

    def test_a_speaker_on_one_trimmed_line_is_kept_as_given(self):
        assert speech().speaker_raw == "Anna Andersson (S)"

    def test_a_speaker_with_a_title_and_inner_spaces_is_kept_as_given(self):
        speaker = "Statsrådet  Anna Andersson (S)"
        assert speech(speaker_raw=speaker).speaker_raw == speaker


class TestParagraphs:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "\xa0",
            "\t",
            "  Tack.",
            "Tack. ",
            "Tack.\n",
            "Tack.\nJag yrkar bifall.",
            "Tack.\r\nJag yrkar bifall.",
            "Tack.\rJag yrkar bifall.",
            "Tack.\u2028Jag yrkar bifall.",
            "Tack.\x0cJag yrkar bifall.",
            "\u200b",
            "\ufeff",
            "\xad",
            "\u200b\u2060\xad",
        ],
    )
    def test_a_paragraph_that_is_blank_split_or_untrimmed_is_refused(self, bad):
        with pytest.raises(ValidationError, match="paragraph 1 must hold a visible character"):
            with_paragraphs("Tack.", bad)

    def test_a_split_paragraph_cannot_pass_as_two_paragraphs(self):
        joined = ("Tack.\nJag yrkar bifall.",)
        split = ("Tack.", "Jag yrkar bifall.")
        assert text_sha256(joined) == text_sha256(split)
        assert with_paragraphs(*split).paragraphs == split
        with pytest.raises(ValidationError):
            with_paragraphs(*joined)

    def test_invisible_characters_inside_a_paragraph_are_kept(self):
        text = "Riks\xaddagen\u200b sa ja."
        assert with_paragraphs("Tack.", text).paragraphs == ("Tack.", text)

    @pytest.mark.parametrize(
        "bad",
        ["", " ", " (Applåder)", "(Applåder) ", "(Applåder)\n(Skratt)", "(Applåder)\r", "\u200b"],
    )
    def test_a_note_follows_the_same_rule(self, bad):
        with pytest.raises(ValidationError, match="note 1 must hold a visible character"):
            speech(notes=("(Applåder)", bad))


class TestHelpers:
    def test_the_hash_covers_the_paragraphs_in_order(self):
        assert text_sha256(("a", "b")) != text_sha256(("b", "a"))
        assert text_sha256(("a", "b")) == text_sha256(["a", "b"])

    def test_the_deep_link_is_the_page_that_holds_the_speech(self):
        assert deep_link("HC09139", 36) == "https://data.riksdagen.se/anforande/HC09139-36/html"

    def test_the_deep_link_is_built_from_the_speech_key(self):
        assert deep_link("GR092", 3) == "https://data.riksdagen.se/anforande/GR092-3/html"

    def test_a_session_runs_from_september_to_the_september_after(self):
        assert session_bounds("2014/15") == (date(2014, 9, 1), date(2015, 9, 30))

    def test_a_session_is_named_after_the_year_it_opens(self):
        assert session_label(2002) == "2002/03"
        assert session_label(2099) == "2099/00"
        assert session_start_year("2002/03") == 2002

    @pytest.mark.parametrize("label", ["2002", "2002/04", "02/03", "2002/2003", "1999/00", ""])
    def test_a_malformed_session_label_raises(self, label):
        with pytest.raises(ValueError):
            session_start_year(label)

    def test_a_session_before_2000_is_refused_with_the_year_it_was_given(self):
        with pytest.raises(ValueError, match="opening in 1999 is before 2000/01"):
            session_label(1999)

    def test_session_bounds_refuses_a_malformed_label(self):
        with pytest.raises(ValueError, match="Not a session label"):
            session_bounds("2014")


def passage(**changes: object) -> Passage:
    """Build a valid passage, with any field replaced by *changes*."""
    fields: dict[str, object] = {
        "passage_id": "H209100-7:0-2",
        "anf_id": "H209100-7",
        "first": 0,
        "last": 2,
        "words": 187,
        "party": "S",
        "date": date(2015, 5, 12),
        "person_id": "0123456789012",
    }
    fields.update(changes)
    return Passage(**fields)


class TestAValidPassage:
    def test_can_be_built(self):
        assert passage().passage_id == "H209100-7:0-2"

    def test_a_single_paragraph_passage_has_the_same_first_and_last(self):
        built = passage(passage_id="H209100-7:4-4", first=4, last=4, words=1)
        assert (built.first, built.last) == (4, 4)

    def test_the_speaker_may_be_unknown(self):
        assert passage(person_id=None).person_id is None

    def test_cannot_be_mutated(self):
        with pytest.raises(ValidationError):
            passage().words = 1  # type: ignore[misc]

    def test_takes_no_unknown_field(self):
        with pytest.raises(ValidationError):
            passage(text="Fru talman!")

    def test_the_key_is_the_speech_and_the_paragraph_range(self):
        assert passage_id("HC09139-36", 0, 2) == "HC09139-36:0-2"


class TestPassageInvariants:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"passage_id": "H209100-7:0-3"}, "key does not match"),
            ({"passage_id": "H209100-8:0-2"}, "key does not match"),
            ({"anf_id": "H209100-8"}, "key does not match"),
            ({"passage_id": "H209100-7:2-0", "first": 2, "last": 0}, "first paragraph comes after"),
            ({"passage_id": "H209100-7:-1-2", "first": -1}, "cannot be negative"),
            ({"passage_id": "H209100-7:0--1", "last": -1}, "cannot be negative"),
            ({"words": 0}, "at least one word"),
            ({"words": -4}, "at least one word"),
            ({"party": "FP"}, "not one of the eight parties"),
            ({"party": "TALMANNEN"}, "not one of the eight parties"),
        ],
    )
    def test_a_passage_that_breaks_a_rule_cannot_exist(self, changes, message):
        with pytest.raises(ValidationError, match=message):
            passage(**changes)
