"""Tests for search/eval/unanswerable.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from search.eval.unanswerable import SCHEMAS, UnanswerableRequest, accepted, check

SPEAKERS = ["anna andersson (s)", "arbetsmarknadsministern johan britz (l)"]


def request(kind: str, questions: list) -> UnanswerableRequest:
    return UnanswerableRequest(
        custom_id=f"unanswerable-{kind}",
        kind=kind,
        asked=len(questions),
        prompt_sha256="0" * 64,
        response=json.dumps({"questions": questions}, ensure_ascii=False),
        created=datetime(2026, 11, 21, tzinfo=timezone.utc),
    )


class TestCheck:
    def test_an_off_topic_question_is_checked_for_form_only(self):
        assert check("off_topic", "Hur kokar man ägg?", SPEAKERS) == ("Hur kokar man ägg?", None)
        assert check("off_topic", "Hur kokar man ägg", SPEAKERS)[1] == "no question mark"
        assert check("off_topic", {"question": "x?"}, SPEAKERS)[1] == "unreadable"

    def test_a_named_member_must_be_unknown_and_named_in_full(self):
        good = {"name": "Karin Holmgård", "question": "Vad har Karin Holmgård sagt om skolan?"}
        assert check("unknown_speaker", good, SPEAKERS)[1] is None
        real = {"name": "Johan Britz", "question": "Vad har Johan Britz sagt om skolan?"}
        assert "speaker's name" in check("unknown_speaker", real, SPEAKERS)[1]
        absent = {"name": "Karin Holm", "question": "Vad har hon sagt om skolan?"}
        assert "missing" in check("unknown_speaker", absent, SPEAKERS)[1]

    def test_sverigedemokraterna_before_2010_needs_the_party_and_a_year_from_2003_to_2009(self):
        good = {"year": 2006, "question": "Vad sa Sverigedemokraterna om skatten 2006?"}
        assert check("sd_before_2010", good, SPEAKERS)[1] is None
        late = {"year": 2012, "question": "Vad sa Sverigedemokraterna om skatten 2012?"}
        assert "2003 to 2009" in check("sd_before_2010", late, SPEAKERS)[1]
        unnamed = {"year": 2006, "question": "Vad sa partiet om skatten 2006?"}
        assert "not named" in check("sd_before_2010", unnamed, SPEAKERS)[1]

    def test_after_the_index_needs_a_year_from_2028_and_no_other(self):
        good = {"year": 2030, "question": "Vad sades i riksdagen 2030 om valet?"}
        assert check("after_index", good, SPEAKERS)[1] is None
        mixed = {"year": 2030, "question": "Hur skilde sig 2030 från 2022?"}
        assert "2028 to 2035" in check("after_index", mixed, SPEAKERS)[1]

    @pytest.mark.parametrize("kind", ["off_topic", "unknown_speaker", "sd_before_2010"])
    def test_every_kind_has_a_schema_that_forbids_extra_fields(self, kind):
        assert SCHEMAS[kind]["additionalProperties"] is False


class TestAccepted:
    def test_the_first_passing_questions_are_kept_once_each(self):
        requests = [
            request(
                "off_topic",
                [
                    "Hur kokar man ägg?",
                    "hur kokar man ägg?",
                    "Utan frågetecken",
                    "Vad heter katten?",
                ],
            ),
            request("after_index", [{"year": 2030, "question": "Vad sades 2030?"}]),
        ]
        kept, refused = accepted("off_topic", requests, 5, SPEAKERS)
        assert [text for text, _ in kept] == ["Hur kokar man ägg?", "Vad heter katten?"]
        assert refused == {"no question mark": 1, "repeats an earlier question": 1}

    def test_no_more_than_asked_for_are_kept(self):
        requests = [request("off_topic", ["A?", "B?", "C?"])]
        kept, _ = accepted("off_topic", requests, 2, SPEAKERS)
        assert [text for text, _ in kept] == ["A?", "B?"]

    def test_a_broken_answer_keeps_nothing(self):
        broken = request("off_topic", []).model_copy(update={"response": "{"})
        assert accepted("off_topic", [broken], 5, SPEAKERS) == ([], {})
