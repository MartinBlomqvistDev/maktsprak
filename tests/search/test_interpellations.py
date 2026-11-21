"""Tests for search/eval/interpellations.py."""

from __future__ import annotations

from collections import Counter
from datetime import date

from search.eval.interpellations import (
    Debate,
    draw_sessions,
    normalise_heading,
    read_debates,
    sessions_by_period,
    split_debates,
    topic_of,
)
from tests.search.corpus_kit import speech, write_corpus

HEADING = "Svar på interpellation 2013/14:434 om stöd till kvinnojourer"


class TestHeadings:
    def test_the_marks_of_a_continued_debate_are_stripped(self):
        assert normalise_heading(f"(forts. från 9 §) {HEADING}") == HEADING
        assert normalise_heading(f"(forts.) {HEADING}") == HEADING
        assert normalise_heading(f"{HEADING} (forts.)") == HEADING
        assert normalise_heading(None) is None
        assert normalise_heading("  ") is None

    def test_the_title_follows_the_first_om(self):
        assert topic_of(HEADING) == "Stöd till kvinnojourer"
        joint = "Svar på interpellationerna 2012/13:50 och 52 om bostäder och om villor"
        assert topic_of(joint) == "Bostäder och om villor"

    def test_a_heading_without_om_or_without_an_answer_has_no_title(self):
        assert topic_of("Svar på interpellation 2012/13:211 Vart är vi på väg?") is None
        assert topic_of("Debatt om budgeten") is None


class TestSessions:
    def test_a_session_across_two_periods_belongs_to_none(self):
        periods = sessions_by_period("2003/04", "2025/26")
        assert periods[2003][0] == "2003/04" and periods[2003][-1] == "2008/09"
        assert "2009/10" not in [s for sessions in periods.values() for s in sessions]
        assert "2017/18" not in periods[2010] and periods[2018][0] == "2018/19"

    def test_the_draw_takes_one_session_per_period_and_follows_the_seed(self):
        periods = sessions_by_period("2003/04", "2025/26")
        drawn = draw_sessions(periods, 1, seed=20261121)
        assert len(drawn) == 3
        assert drawn == draw_sessions(periods, 1, seed=20261121)
        assert all(any(s in sessions for sessions in periods.values()) for s in drawn)


class TestDebates:
    def test_two_of_every_five_debates_per_session_go_to_development(self):
        debates = [
            Debate(heading=f"h{n}", session="2007/08", topic="T", speeches=(f"a{n}-1",))
            for n in range(10)
        ] + [
            Debate(heading=f"g{n}", session="2011/12", topic="T", speeches=(f"b{n}-1",))
            for n in range(5)
        ]
        splits = split_debates(debates, seed=1)
        assert Counter(splits[:10])["dev"] == 4
        assert Counter(splits[10:])["dev"] == 2

    def test_a_continued_debate_is_one_debate(self, tmp_path):
        day = date(2014, 3, 4)
        activity = "interpellationsdebatt"
        speeches = [
            speech("H109090", 1, ["Svar."], day=day, title=HEADING, activity=activity),
            speech("H109090", 2, ["Fråga."], day=day, title=HEADING, activity=activity),
            speech("H109090", 3, ["Mer."], day=day, title=f"(forts.) {HEADING}", activity=activity),
            speech(
                "H109090",
                4,
                ["Annat."],
                day=day,
                title="Svar på interpellation 2013/14:12 utan titel",
                activity=activity,
            ),
            speech("H109090", 5, ["Debatt."], day=day, title=HEADING, activity="ärendedebatt"),
        ]
        speeches_path, _ = write_corpus(tmp_path, speeches)
        debates, left_out = read_debates(speeches_path, ["2013/14"])
        assert len(debates) == 1
        assert debates[0].speeches == ("H109090-1", "H109090-2", "H109090-3")
        assert debates[0].topic == "Stöd till kvinnojourer"
        assert left_out == {"no title of an interpellation in the heading": 1}
