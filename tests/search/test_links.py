"""Tests for search/corpus/links.py. Nothing here reaches the network or reads the clock."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from search.config import load_settings
from search.corpus.fetch import sessions_between
from search.corpus.links import (
    BUDGET_EXHAUSTED,
    EXCERPT_CHARS,
    OPENING_CHARS,
    BudgetExhausted,
    DeepLinkReport,
    HostThrottle,
    PageCheck,
    PageJudgement,
    PageResponse,
    RequestBudget,
    SpeechRef,
    SpeechText,
    check_for,
    draw_sample,
    excerpt_of,
    fetch_page,
    is_xml_record,
    judge_page,
    neighbours_of,
    opening_words,
    page_text,
    report_of,
    squeeze,
    totals_by_layout,
    totals_by_session,
    totals_of,
    verdict_of,
)
from search.models import deep_link

HOST = "https://data.riksdagen.se"
ADDRESS = f"{HOST}/anforande/HC093-36/html"
SPEAKER = "Arbetsmarknadsministern Johan Britz (L)"
OTHER_SPEAKER = "Eva Lindh (S)"
PARAGRAPHS = (
    "Fru talman! Det är svårt att säga om det kommer att räcka. Men utgångspunkten är, som jag "
    "som sagt tror att vi är helt överens om, att arbetstagare i Sverige ska ha goda villkor.",
    "Utgångspunkten är, som jag sa, att arbetstagare i Sverige ska ha goda arbetsvillkor.",
    "Med detta vill jag återigen tacka Eva Lindh för den här interpellationen.",
)
NEIGHBOUR_PARAGRAPHS = (
    "Fru talman! Jag får tacka för svaret och för att vi är överens om det mesta.",
    "Det återstår en sak.",
)
PREVIOUS_PARAGRAPHS = (
    "Fru talman! Eva Lindh har frågat arbetsmarknadsministern vilka åtgärder han tänker vidta.",
    "Jag vill börja med att tacka för frågan.",
)


def page(
    number: int = 36,
    speaker: str = SPEAKER,
    paragraphs: tuple[str, ...] = PARAGRAPHS,
    text_div: str | None = None,
) -> str:
    """Build the page of one speech as the Riksdag serves it, shortened."""
    body = text_div if text_div is not None else "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        '<div class="anforande">\n  <style>\n.text {\nwhite-space:pre-wrap;\n}\n</style>\n'
        "  <h4>Protokoll 2024/25:139 Torsdagen den 4 september</h4>"
        "interpellationsdebatt / Svar p&aring; interpellation 2024/25:734<br />\n\n\n"
        f'Anf. {number} {speaker}<br /><div class="text">{body}</div></div>'
    )


def text_of(
    number: int = 36,
    speaker: str = SPEAKER,
    paragraphs: tuple[str, ...] = PARAGRAPHS,
    dok_id: str = "HC093",
) -> SpeechText:
    """Build what a page is compared with, from a speech's paragraphs."""
    return SpeechText(
        anf_id=f"{dok_id}-{number}",
        anf_nummer=number,
        speaker=speaker,
        paragraphs=paragraphs,
    )


def neighbour_text() -> SpeechText:
    """The speech that follows the one above in its protocol."""
    return text_of(37, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)


def previous_text() -> SpeechText:
    """The speech that comes before the one above in its protocol."""
    return text_of(35, OTHER_SPEAKER, PREVIOUS_PARAGRAPHS)


def both_neighbours() -> tuple[SpeechText, SpeechText]:
    """The speeches before and after speech 36, in that order."""
    return (previous_text(), neighbour_text())


def page_with_neighbour() -> str:
    """A page that holds speech 36 and then speech 37, as a page of the whole protocol would."""
    return page() + page(37, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)


def page_with_previous() -> str:
    """A page that holds speech 35 and then speech 36, as a page of the whole protocol would."""
    return page(35, OTHER_SPEAKER, PREVIOUS_PARAGRAPHS) + page()


def xml_record() -> str:
    """The API's XML record of speech 36, shortened, as the server sometimes answers with."""
    return (
        "<anforande><dok_id>HC093</dok_id><anforande_nummer>36</anforande_nummer>"
        f"<talare>{SPEAKER}</talare><anforandetext>"
        + "".join(f"<p>{p}</p>" for p in PARAGRAPHS)
        + "</anforandetext></anforande>"
    )


def ref(
    number: int = 36,
    dok_id: str = "HC093",
    rm: str = "2024/25",
    day: date = date(2024, 9, 11),
) -> SpeechRef:
    """Build a sampled speech."""
    return SpeechRef(
        anf_id=f"{dok_id}-{number}",
        dok_id=dok_id,
        anf_nummer=number,
        rm=rm,
        date=day,
        layout="html",
    )


def corpus(sessions: list[str], protocols: int, speeches: int = 3) -> list[SpeechRef]:
    """Build a fake corpus of speeches: every protocol has the same number of speeches."""
    return [
        ref(number, f"H{index}{session[2:4]}{protocol:03d}", session, date(int(session[:4]), 10, 1))
        for index, session in enumerate(sessions)
        for protocol in range(protocols)
        for number in range(1, speeches + 1)
    ]


class TestSpeechRef:
    def test_the_address_is_computed_from_the_protocol_and_the_number(self):
        speech = ref(36)
        assert speech.url == deep_link("HC093", 36) == ADDRESS

    def test_the_address_is_on_the_open_data_host(self):
        assert ref(5, "GR092").url == f"{HOST}/anforande/GR092-5/html"

    def test_there_is_no_url_to_set(self):
        with pytest.raises(ValidationError):
            SpeechRef(
                anf_id="HC093-36",
                dok_id="HC093",
                anf_nummer=36,
                rm="2024/25",
                date=date(2024, 9, 11),
                layout="html",
                url="https://www.riksdagen.se/sv/dokument-och-lagar/dokument/protokoll/x/#anf36",
            )


class TestSqueeze:
    def test_removes_every_kind_of_whitespace(self):
        assert squeeze(" Fru\ttalman!\n\r Tack \u00a0för\u2003ordet. ") == "Frutalman!Tackförordet."

    def test_removes_soft_hyphens_and_zero_width_spaces(self):
        assert squeeze("Riks\u00addagen\u200b sa ja.") == "Riksdagensaja."

    def test_keeps_case_and_punctuation(self):
        assert squeeze("Anf. 3 (S)") == "Anf.3(S)"


class TestPageText:
    def test_removes_tags_and_decodes_entities(self):
        assert page_text("<p>Svar p&aring; <b>fr&aring;gan</b>.</p>") == "Svar på frågan."

    def test_leaves_out_styles_and_scripts(self):
        html = "<style>.text { color: red }</style><script>var x = 1;</script><p>Hej.</p>"
        assert page_text(html) == "Hej."

    def test_a_self_closing_script_does_not_hide_what_follows(self):
        assert page_text("<script src='a.js'/><p>Hej.</p>") == "Hej."

    def test_a_self_closing_tag_inside_a_hidden_region_neither_opens_nor_closes_one(self):
        html = "<template><style/>hidden</template><p>Hej.</p>"
        assert page_text(html) == "Hej."

    def test_an_empty_page_has_no_text(self):
        assert page_text("") == ""


class TestOpeningWords:
    def test_takes_the_first_characters_of_the_paragraph_without_whitespace(self):
        paragraph = "Fru talman! " + "ord " * 40
        words = opening_words(paragraph)
        assert " " not in words
        assert words == squeeze(paragraph[:OPENING_CHARS])

    def test_a_short_paragraph_is_taken_whole(self):
        assert opening_words("Tack,  fru talman.") == "Tack,frutalman."


def found(judgement: PageJudgement | PageCheck) -> tuple[bool, bool, bool]:
    """The three judgements, in order: the name, the first opening, every opening."""
    return (
        judgement.names_speech,
        judgement.first_opening_found,
        judgement.every_opening_in_order,
    )


class TestJudgePage:
    def test_the_page_of_the_speech_passes(self):
        judgement = judge_page(page(), text_of())
        assert found(judgement) == (True, True, True)
        assert judgement.is_match

    def test_the_page_of_the_neighbouring_speech_does_not_pass(self):
        judgement = judge_page(page(), neighbour_text())
        assert not judgement.is_match
        assert not judgement.names_speech
        assert not judgement.first_opening_found

    def test_a_whitespace_only_paragraph_in_the_speech_makes_the_judgement_fail(self):
        speech = text_of(paragraphs=(PARAGRAPHS[0], " \n\t ", PARAGRAPHS[1]))
        judgement = judge_page(page(), speech)
        assert found(judgement) == (True, True, False)
        assert not judgement.is_match

    def test_the_neighbouring_speech_passes_on_its_own_page(self):
        html = page(37, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)
        assert judge_page(html, neighbour_text()).is_match
        assert not judge_page(html, text_of()).is_match

    def test_the_same_speaker_under_another_number_is_not_the_speech(self):
        judgement = judge_page(page(35), text_of(36))
        assert not judgement.names_speech
        assert judgement.first_opening_found

    def test_a_number_that_starts_the_page_number_is_not_the_speech(self):
        assert not judge_page(page(36), text_of(3)).names_speech

    def test_another_speaker_under_the_same_number_is_not_the_speech(self):
        assert not judge_page(page(36, OTHER_SPEAKER), text_of(36)).names_speech

    def test_the_text_of_another_speech_under_the_right_name_does_not_pass(self):
        html = page(36, SPEAKER, NEIGHBOUR_PARAGRAPHS)
        judgement = judge_page(html, text_of())
        assert judgement.names_speech
        assert not judgement.first_opening_found
        assert not judgement.is_match

    def test_a_page_that_lacks_the_last_paragraph_does_not_pass(self):
        judgement = judge_page(page(paragraphs=PARAGRAPHS[:2]), text_of())
        assert judgement.first_opening_found
        assert not judgement.every_opening_in_order
        assert not judgement.is_match

    def test_a_page_that_lacks_the_last_paragraph_whose_words_open_the_one_before_fails(self):
        closing = "Jag yrkar bifall till reservation 2."
        paragraphs = (PARAGRAPHS[0], closing + " Det är viktigt för alla.", closing)
        speech = text_of(paragraphs=paragraphs)
        assert judge_page(page(paragraphs=paragraphs), speech).is_match
        judgement = judge_page(page(paragraphs=paragraphs[:2]), speech)
        assert judgement.first_opening_found
        assert not judgement.every_opening_in_order
        assert not judgement.is_match

    def test_a_page_that_lacks_the_last_paragraph_whose_words_occur_earlier_fails(self):
        closing = "Jag yrkar bifall till reservation 2."
        paragraphs = (PARAGRAPHS[0] + " " + closing, PARAGRAPHS[1], closing)
        speech = text_of(paragraphs=paragraphs)
        assert judge_page(page(paragraphs=paragraphs), speech).is_match
        assert not judge_page(page(paragraphs=paragraphs[:2]), speech).is_match

    def test_a_page_that_lacks_a_middle_paragraph_does_not_pass(self):
        judgement = judge_page(page(paragraphs=(PARAGRAPHS[0], PARAGRAPHS[2])), text_of())
        assert judgement.first_opening_found
        assert not judgement.every_opening_in_order
        assert not judgement.is_match

    def test_a_page_with_two_paragraphs_swapped_does_not_pass(self):
        swapped = (PARAGRAPHS[0], PARAGRAPHS[2], PARAGRAPHS[1])
        judgement = judge_page(page(paragraphs=swapped), text_of())
        assert judgement.first_opening_found
        assert not judgement.every_opening_in_order
        assert not judgement.is_match

    def test_text_dropped_after_the_opening_words_of_a_paragraph_is_not_seen(self):
        shortened = (PARAGRAPHS[0][:OPENING_CHARS], *PARAGRAPHS[1:])
        assert judge_page(page(paragraphs=shortened), text_of()).is_match

    def test_the_last_paragraph_must_come_after_the_first(self):
        judgement = judge_page(page(paragraphs=PARAGRAPHS[::-1]), text_of())
        assert judgement.first_opening_found
        assert not judgement.every_opening_in_order

    def test_a_speech_of_one_paragraph_passes_on_that_paragraph(self):
        paragraphs = ("Tack, fru talman.",)
        assert judge_page(page(paragraphs=paragraphs), text_of(paragraphs=paragraphs)).is_match

    def test_two_paragraphs_that_begin_alike_are_found_as_two(self):
        paragraphs = ("Tack.", "Tack.")
        speech = text_of(paragraphs=paragraphs)
        assert judge_page(page(paragraphs=paragraphs), speech).every_opening_in_order
        assert not judge_page(page(paragraphs=("Tack.",)), speech).every_opening_in_order

    def test_only_the_opening_characters_are_compared(self):
        opening = PARAGRAPHS[0][:OPENING_CHARS]
        html = page(paragraphs=(opening + " och sedan något helt annat.", *PARAGRAPHS[1:]))
        assert judge_page(html, text_of()).is_match

    def test_a_difference_inside_the_opening_characters_is_not_a_match(self):
        altered = PARAGRAPHS[0].replace("svårt", "enkelt", 1)
        judgement = judge_page(page(paragraphs=(altered, *PARAGRAPHS[1:])), text_of())
        assert not judgement.first_opening_found
        assert not judgement.every_opening_in_order

    def test_a_paragraph_that_the_corpus_put_together_from_lines_is_found(self):
        lines = (
            "\nFru talman! Det är svårt att säga\nom det kommer att räcka. Men \nutgångspunkten är"
        )
        paragraph = "Fru talman! Det är svårt att säga om det kommer att räcka. Men utgångspunkten"
        judgement = judge_page(page(text_div=lines), text_of(paragraphs=(paragraph,)))
        assert judgement.is_match

    def test_whitespace_entities_soft_hyphens_and_markup_do_not_matter(self):
        corpus_paragraph = "Riksdagen är ett skäl <till> att tacka."
        html = page(
            text_div="<p>Riks&shy;dagen\u200b  är ett <b>skäl</b>\n&lt;till&gt; att&nbsp;tacka.</p>"
        )
        speech = text_of(paragraphs=(corpus_paragraph,))
        assert judge_page(html, speech).is_match

    def test_text_in_a_script_or_a_style_is_not_on_the_page(self):
        html = (
            f"<style>Anf. 36 {SPEAKER}</style><script>{PARAGRAPHS[0]}</script>"
            f"<p>{PARAGRAPHS[2]}</p>"
        )
        judgement = judge_page(html, text_of())
        assert not judgement.names_speech
        assert not judgement.first_opening_found

    def test_an_empty_page_shows_nothing(self):
        judgement = judge_page("", text_of())
        assert found(judgement) == (False, False, False)

    def test_a_page_that_is_not_html_shows_nothing(self):
        assert not judge_page("Not Found", text_of()).is_match

    def test_the_xml_record_of_the_speech_has_the_text_but_not_the_heading(self):
        record = (
            "<anforande><dok_id>HC093</dok_id><anforande_nummer>36</anforande_nummer>"
            f"<talare>{SPEAKER}</talare><anforandetext>"
            + "".join(f"<p>{p}</p>" for p in PARAGRAPHS)
            + "</anforandetext></anforande>"
        )
        judgement = judge_page(record + record, text_of())
        assert found(judgement) == (False, True, True)
        assert not judgement.is_match


class TestPageJudgement:
    def test_every_paragraph_cannot_be_in_order_when_the_first_is_absent(self):
        with pytest.raises(ValidationError, match="cannot all be found"):
            PageJudgement(names_speech=True, first_opening_found=False, every_opening_in_order=True)

    def test_only_a_page_that_passes_all_three_is_a_match(self):
        assert PageJudgement(
            names_speech=True, first_opening_found=True, every_opening_in_order=True
        ).is_match
        assert not PageJudgement(
            names_speech=False, first_opening_found=True, every_opening_in_order=True
        ).is_match
        assert not PageJudgement(
            names_speech=True, first_opening_found=True, every_opening_in_order=False
        ).is_match


def neighbour_ids(speech: SpeechRef, texts: dict[str, SpeechText]) -> list[str]:
    """The keys of the neighbours that :func:`neighbours_of` returns, in its order."""
    return [n.anf_id for n in neighbours_of(speech, texts)]


class TestNeighboursOf:
    def texts(self, *numbers: int, dok_id: str = "HC093") -> dict[str, SpeechText]:
        return {f"{dok_id}-{n}": text_of(n, dok_id=dok_id) for n in numbers}

    def test_the_speech_before_and_the_speech_after_are_both_returned(self):
        assert neighbour_ids(ref(36), self.texts(35, 36, 37)) == ["HC093-35", "HC093-37"]

    def test_the_first_speech_of_a_protocol_has_only_the_one_after(self):
        assert neighbour_ids(ref(1), self.texts(1, 2, 3)) == ["HC093-2"]

    def test_the_last_speech_of_a_protocol_has_only_the_one_before(self):
        assert neighbour_ids(ref(3), self.texts(1, 2, 3)) == ["HC093-2"]

    def test_a_protocol_of_one_speech_has_no_neighbour(self):
        assert neighbours_of(ref(1), self.texts(1)) == ()

    def test_a_speech_with_the_same_number_in_another_protocol_is_not_a_neighbour(self):
        texts = {**self.texts(36), **self.texts(35, 37, dok_id="HC094")}
        assert neighbours_of(ref(36), texts) == ()


class TestDrawSample:
    SESSIONS = sessions_between("2003/04", "2026/27")

    def test_two_per_session_and_one_more_from_the_two_latest_makes_fifty(self):
        sample = draw_sample(corpus(self.SESSIONS, protocols=5), seed=1)
        counts = Counter(speech.rm for speech in sample)
        assert len(self.SESSIONS) == 24
        assert len(sample) == 50
        assert counts["2025/26"] == counts["2026/27"] == 3
        assert {counts[session] for session in self.SESSIONS[:-2]} == {2}

    def test_speeches_come_from_distinct_protocols(self):
        sample = draw_sample(corpus(self.SESSIONS, protocols=5), seed=1)
        assert len({speech.dok_id for speech in sample}) == len(sample)

    def test_the_same_seed_gives_the_same_sample(self):
        speeches = corpus(self.SESSIONS, protocols=5)
        assert draw_sample(speeches, seed=7) == draw_sample(speeches, seed=7)

    def test_the_order_of_the_input_does_not_matter(self):
        speeches = corpus(self.SESSIONS, protocols=5)
        shuffled = list(speeches)
        random.Random(0).shuffle(shuffled)
        assert draw_sample(shuffled, seed=7) == draw_sample(speeches, seed=7)

    def test_a_different_seed_gives_a_different_sample(self):
        speeches = corpus(self.SESSIONS, protocols=20)
        assert draw_sample(speeches, seed=1) != draw_sample(speeches, seed=2)

    def test_protocols_are_ranked_by_the_sha256_of_the_seed_and_the_id(self):
        speeches = corpus(["2024/25"], protocols=6)
        ranked = sorted(
            {speech.dok_id for speech in speeches},
            key=lambda dok_id: hashlib.sha256(f"2026|{dok_id}".encode()).hexdigest(),
        )
        sample = draw_sample(speeches, seed=2026, per_session=2, extra_recent_sessions=0)
        assert {speech.dok_id for speech in sample} == set(ranked[:2])

    def test_the_draw_is_pinned_so_that_it_cannot_change_between_versions_of_python(self):
        speeches = corpus(["2024/25"], protocols=6)
        sample = draw_sample(speeches, seed=2026, per_session=2, extra_recent_sessions=0)
        assert [speech.anf_id for speech in sample] == ["H024004-1", "H024001-2"]

    def test_the_sample_is_ordered_by_session(self):
        sample = draw_sample(corpus(self.SESSIONS, protocols=5), seed=1)
        assert [speech.rm for speech in sample] == sorted(speech.rm for speech in sample)

    def test_a_session_with_few_protocols_gives_all_it_has(self):
        sample = draw_sample(corpus(["2024/25"], protocols=1), seed=1, per_session=2)
        assert len(sample) == 1

    def test_more_extra_sessions_than_sessions_gives_every_session_one_more(self):
        sample = draw_sample(
            corpus(["2023/24", "2024/25"], protocols=5),
            seed=1,
            per_session=1,
            extra_recent_sessions=9,
        )
        assert len(sample) == 4

    def test_no_extra_sessions_gives_exactly_the_per_session_count(self):
        sample = draw_sample(
            corpus(self.SESSIONS, protocols=5), seed=1, per_session=1, extra_recent_sessions=0
        )
        assert len(sample) == 24

    def test_nothing_to_draw_from_gives_an_empty_sample(self):
        assert draw_sample([], seed=1) == []

    def test_each_session_must_give_at_least_one(self):
        with pytest.raises(ValueError, match="at least one"):
            draw_sample([], seed=1, per_session=0)

    def test_extra_sessions_cannot_be_negative(self):
        with pytest.raises(ValueError, match="negative"):
            draw_sample([], seed=1, extra_recent_sessions=-1)


class FakeClock:
    """A clock that only moves when it is told to, or when something sleeps."""

    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestHostThrottle:
    def test_the_first_request_goes_at_once(self):
        clock = FakeClock()
        asyncio.run(HostThrottle(1.0, sleep=clock.sleep, now=clock.time).wait())
        assert clock.slept == []

    def test_a_request_right_after_another_waits_out_the_interval(self):
        clock = FakeClock()

        async def two() -> None:
            throttle = HostThrottle(1.0, sleep=clock.sleep, now=clock.time)
            await throttle.wait()
            await throttle.wait()

        asyncio.run(two())
        assert clock.slept == [1.0]

    def test_only_the_rest_of_the_interval_is_waited(self):
        clock = FakeClock()

        async def two() -> None:
            throttle = HostThrottle(1.0, sleep=clock.sleep, now=clock.time)
            await throttle.wait()
            clock.now += 0.4
            await throttle.wait()

        asyncio.run(two())
        assert clock.slept == [pytest.approx(0.6)]

    def test_a_request_after_the_interval_does_not_wait(self):
        clock = FakeClock()

        async def two() -> None:
            throttle = HostThrottle(1.0, sleep=clock.sleep, now=clock.time)
            await throttle.wait()
            clock.now += 5
            await throttle.wait()

        asyncio.run(two())
        assert clock.slept == []

    def test_requests_made_at_once_still_go_a_full_interval_apart(self):
        clock = FakeClock()
        sent: list[float] = []

        async def request(throttle: HostThrottle) -> None:
            await throttle.wait()
            sent.append(clock.now)

        async def three_at_once() -> None:
            throttle = HostThrottle(1.0, sleep=clock.sleep, now=clock.time)
            await asyncio.gather(*(request(throttle) for _ in range(3)))

        asyncio.run(three_at_once())
        assert sent == [100.0, 101.0, 102.0]


class TestRequestBudget:
    def test_counts_every_request_up_to_the_limit(self):
        budget = RequestBudget(2)
        budget.spend()
        budget.spend()
        assert budget.used == 2

    def test_stops_at_the_limit(self):
        budget = RequestBudget(1)
        budget.spend()
        with pytest.raises(BudgetExhausted, match=BUDGET_EXHAUSTED):
            budget.spend()
        assert budget.used == 1


class FakeHost:
    """A stand-in for the Riksdag's host: answers by address and counts the requests."""

    def __init__(self, answers: dict[str, list[httpx.Response] | httpx.Response]) -> None:
        self.answers = answers
        self.requests: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests[url] += 1
        answer = self.answers.get(url, httpx.Response(404))
        if isinstance(answer, list):
            index = min(self.requests[url], len(answer)) - 1
            answer = answer[index]
        if isinstance(answer, Exception):
            raise answer
        return answer


def get(
    host: FakeHost, url: str = ADDRESS, limit: int = 50, attempts: int = 3
) -> tuple[PageResponse, RequestBudget, FakeClock]:
    """Fetch a page from the fake host with no real waiting."""
    clock = FakeClock()
    budget = RequestBudget(limit)

    async def run() -> PageResponse:
        throttle = HostThrottle(0.0, sleep=clock.sleep, now=clock.time)
        async with httpx.AsyncClient(transport=httpx.MockTransport(host)) as client:
            return await fetch_page(
                client,
                url,
                throttle=throttle,
                budget=budget,
                attempts=attempts,
                backoff_seconds=2.0,
                sleep=clock.sleep,
            )

    return asyncio.run(run()), budget, clock


class TestFetchPage:
    def test_returns_the_page(self):
        host = FakeHost({ADDRESS: httpx.Response(200, text="<p>hej</p>")})
        response, budget, _ = get(host)
        assert (response.status, response.html) == (200, "<p>hej</p>")
        assert response.error is None
        assert budget.used == 1

    def test_a_redirect_is_not_followed(self):
        other = f"{HOST}/anforande/HC093-37/html"
        host = FakeHost(
            {ADDRESS: httpx.Response(301, headers={"location": other}), other: httpx.Response(200)}
        )
        response, budget, _ = get(host)
        assert (response.status, response.html, response.error) == (301, None, "HTTP 301")
        assert sum(host.requests.values()) == 1
        assert budget.used == 1

    def test_a_missing_page_is_not_retried(self):
        host = FakeHost({})
        response, budget, clock = get(host)
        assert (response.status, response.html, response.error) == (404, None, "HTTP 404")
        assert budget.used == 1
        assert clock.slept == []

    def test_a_server_error_is_retried_after_a_growing_pause(self):
        host = FakeHost(
            {
                ADDRESS: [
                    httpx.Response(503),
                    httpx.Response(502),
                    httpx.Response(200, text="hej"),
                ]
            }
        )
        response, budget, clock = get(host)
        assert response.status == 200
        assert budget.used == 3
        assert clock.slept == [2.0, 4.0]

    def test_a_throttling_answer_is_retried(self):
        host = FakeHost({ADDRESS: [httpx.Response(429), httpx.Response(200, text="hej")]})
        response, _, _ = get(host)
        assert response.status == 200

    def test_gives_up_after_the_last_attempt(self):
        host = FakeHost({ADDRESS: httpx.Response(503)})
        response, budget, _ = get(host, attempts=3)
        assert (response.status, response.error) == (503, "HTTP 503")
        assert budget.used == 3

    def test_a_dropped_connection_is_retried_and_then_named(self):
        host = FakeHost({ADDRESS: httpx.ConnectError("refused")})
        response, budget, _ = get(host, attempts=2)
        assert response.status is None
        assert response.error == "ConnectError"
        assert budget.used == 2

    def test_stops_when_the_budget_is_spent_and_keeps_the_answer_that_came_back(self):
        host = FakeHost({ADDRESS: httpx.Response(503)})
        response, budget, _ = get(host, limit=1)
        assert response.status == 503
        assert response.error == f"HTTP 503; {BUDGET_EXHAUSTED}"
        assert response.html is None
        assert budget.used == 1
        assert sum(host.requests.values()) == 1

    def test_a_budget_spent_before_the_first_attempt_sends_nothing(self):
        host = FakeHost({ADDRESS: httpx.Response(200, text="hej")})
        response, budget, _ = get(host, limit=0)
        assert (response.status, response.html, response.error) == (None, None, BUDGET_EXHAUSTED)
        assert budget.used == 0
        assert sum(host.requests.values()) == 0

    def test_a_body_that_cannot_be_decoded_is_named_and_does_not_stop_the_run(self):
        corrupt = [
            httpx.Response(
                200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(b"not gzip")
            )
            for _ in range(2)
        ]
        host = FakeHost({ADDRESS: corrupt})
        response, budget, _ = get(host, attempts=2)
        assert (response.status, response.html, response.error) == (None, None, "DecodingError")
        assert budget.used == 2


def fetched(html: str | None = None) -> PageResponse:
    """A page that came back with status 200, the page of speech 36 unless told otherwise."""
    return PageResponse(status=200, html=html if html is not None else page())


class TestCheckFor:
    def test_judges_a_page_against_its_speech(self):
        check = check_for(ref(), text_of(), fetched(), both_neighbours())
        assert (check.anf_id, check.session, check.status, check.url) == (
            "HC093-36",
            "2024/25",
            200,
            ADDRESS,
        )
        assert found(check) == (True,) * 3
        assert check.passed
        assert check.error is None
        assert check.page_excerpt is None

    def test_the_page_is_judged_against_the_speech_before_and_the_speech_after(self):
        check = check_for(ref(), text_of(), fetched(), both_neighbours())
        assert check.neighbour_anf_ids == ("HC093-35", "HC093-37")
        assert check.neighbour_rejected is True

    def test_a_judgement_that_passes_a_neighbour_too_is_not_rejected(self):
        check = check_for(ref(), text_of(), fetched(), (neighbour_text(), text_of()))
        assert check.neighbour_rejected is False
        assert not check.passed

    def test_a_speech_without_a_neighbour_has_no_second_judgement(self):
        check = check_for(ref(), text_of(), fetched())
        assert (check.neighbour_anf_ids, check.neighbour_rejected) == ((), None)

    def test_a_speech_alone_in_its_protocol_can_pass(self):
        assert check_for(ref(), text_of(), fetched()).passed

    def test_a_page_that_holds_the_following_speech_too_does_not_pass(self):
        check = check_for(ref(), text_of(), fetched(page_with_neighbour()), both_neighbours())
        assert found(check) == (True, True, True)
        assert check.neighbour_rejected is False
        assert not check.passed
        assert check.page_excerpt is not None

    def test_a_page_that_holds_the_previous_speech_too_does_not_pass(self):
        check = check_for(ref(), text_of(), fetched(page_with_previous()), both_neighbours())
        assert found(check) == (True, True, True)
        assert check.neighbour_rejected is False
        assert not check.passed

    def test_the_first_speech_of_a_protocol_is_judged_against_the_one_after(self):
        first = text_of(1)
        following = text_of(2, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)
        html = page(1) + page(2, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)
        check = check_for(ref(1), first, fetched(html), (following,))
        assert check.neighbour_anf_ids == ("HC093-2",)
        assert (check.neighbour_rejected, check.passed) == (False, False)
        assert check_for(ref(1), first, fetched(page(1)), (following,)).passed

    def test_the_last_speech_of_a_protocol_is_judged_against_the_one_before(self):
        last = text_of(3)
        previous = text_of(2, OTHER_SPEAKER, PREVIOUS_PARAGRAPHS)
        html = page(2, OTHER_SPEAKER, PREVIOUS_PARAGRAPHS) + page(3)
        check = check_for(ref(3), last, fetched(html), (previous,))
        assert check.neighbour_anf_ids == ("HC093-2",)
        assert (check.neighbour_rejected, check.passed) == (False, False)
        assert check_for(ref(3), last, fetched(page(3)), (previous,)).passed

    def test_a_page_with_the_neighbours_heading_but_only_its_own_speech_passes(self):
        html = page() + f"Anf. 37 {OTHER_SPEAKER}<br />"
        check = check_for(ref(), text_of(), fetched(html), both_neighbours())
        assert judge_page(html, neighbour_text()).names_speech
        assert check.neighbour_rejected is True
        assert check.passed

    def test_the_page_of_another_speech_does_not_pass(self):
        html = page(37, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)
        check = check_for(ref(), text_of(), fetched(html), both_neighbours())
        assert not check.passed
        assert check.neighbour_rejected is False
        assert check.page_excerpt is not None
        assert "Anf. 37" in check.page_excerpt

    def test_a_page_that_did_not_come_back_shows_nothing(self):
        response = PageResponse(status=404, error="HTTP 404")
        check = check_for(ref(), text_of(), response, both_neighbours())
        assert (check.status, check.error) == (404, "HTTP 404")
        assert found(check) == (False,) * 3
        assert not check.passed
        assert check.neighbour_rejected is None
        assert check.xml_record is False
        assert check.page_excerpt is None

    def test_records_whether_the_page_holds_the_fields_of_the_xml_record(self):
        assert check_for(ref(), text_of(), fetched(xml_record())).xml_record
        assert not check_for(ref(), text_of(), fetched()).xml_record

    def test_a_body_is_an_xml_record_only_when_it_holds_every_field(self):
        assert is_xml_record(xml_record())
        assert not is_xml_record(xml_record().replace("<anforandetext>", "<text>"))
        assert not is_xml_record(page())

    def test_a_status_other_than_200_cannot_pass(self):
        with pytest.raises(ValidationError, match="passed must be"):
            PageCheck(**{**checked(ref()).model_dump(), "status": 301, "passed": True})

    def test_a_passing_page_cannot_be_recorded_as_failing(self):
        with pytest.raises(ValidationError, match="passed must be"):
            PageCheck(**{**checked(ref()).model_dump(), "passed": False})

    def test_a_page_that_holds_a_neighbouring_speech_cannot_be_recorded_as_passing(self):
        entry = checked(ref(), page_with_previous(), neighbours=both_neighbours()).model_dump()
        assert entry["neighbour_rejected"] is False
        with pytest.raises(ValidationError, match="passed must be"):
            PageCheck(**{**entry, "passed": True})

    def test_the_excerpt_is_the_start_of_the_page_text(self):
        html = page(text_div="<p>" + "ord " * 200 + "</p>")
        excerpt = excerpt_of(html)
        assert len(excerpt) == EXCERPT_CHARS
        assert excerpt.startswith("Protokoll 2024/25:139 Torsdagen den 4 september")


def checked(
    speech: SpeechRef,
    html: str | None = None,
    text: SpeechText | None = None,
    neighbours: tuple[SpeechText, ...] = (),
) -> PageCheck:
    """Check one speech against a fetched page, the page of speech 36 unless given."""
    return check_for(speech, text or text_of(speech.anf_nummer), fetched(html), neighbours)


class TestTotals:
    def test_counts_what_was_found(self):
        checks = [
            checked(ref(36), neighbours=both_neighbours()),
            checked(ref(36), page(paragraphs=PARAGRAPHS[:2]), neighbours=both_neighbours()),
            checked(ref(37), page(), neighbours=(text_of(36),)),
            checked(ref(36, "HC094")),
        ]
        totals = totals_of(checks)
        assert totals.sampled == 4
        assert totals.fetched == 4
        assert totals.names_speech == 3
        assert totals.first_opening_found == 4
        assert totals.every_opening_in_order == 3
        assert totals.passed == 2
        assert totals.neighbour_checked == 3
        assert totals.neighbour_rejected == 2
        assert totals.passed_neighbour_checked == 1
        assert totals.never_requested == 0
        assert totals.retry_cut_by_budget == 0

    def test_counts_the_pages_never_requested_because_the_budget_ran_out(self):
        checks = [
            check_for(ref(36), text_of(), PageResponse(error=BUDGET_EXHAUSTED)),
            check_for(
                ref(5, "HC094"),
                text_of(5),
                PageResponse(status=503, error=f"HTTP 503; {BUDGET_EXHAUSTED}"),
            ),
            checked(ref(36, "HC095")),
        ]
        totals = totals_of(checks)
        assert (totals.sampled, totals.fetched, totals.never_requested) == (3, 1, 1)
        assert totals.retry_cut_by_budget == 1

    def test_a_page_that_did_not_come_back_is_sampled_but_not_fetched(self):
        missing = check_for(ref(), text_of(), PageResponse(status=404, error="HTTP 404"))
        totals = totals_of([missing])
        assert (totals.sampled, totals.fetched, totals.passed) == (1, 0, 0)

    def test_totals_are_given_per_session_in_chronological_order(self):
        checks = [
            checked(ref(36, "HC093", "2024/25")),
            checked(ref(37, "HC094", "2003/04", date(2003, 10, 1))),
            checked(ref(36, "HC095", "2024/25")),
        ]
        by_session = totals_by_session(checks)
        assert list(by_session) == ["2003/04", "2024/25"]
        assert (by_session["2003/04"].sampled, by_session["2024/25"].sampled) == (1, 2)

    def test_totals_are_given_per_layout(self):
        lines = SpeechRef(
            anf_id="GR092-3",
            dok_id="GR092",
            anf_nummer=3,
            rm="2003/04",
            date=date(2003, 9, 16),
            layout="lines",
        )
        by_layout = totals_by_layout([checked(ref()), checked(lines), checked(ref(36, "HC094"))])
        assert list(by_layout) == ["html", "lines"]
        assert (by_layout["html"].sampled, by_layout["lines"].sampled) == (2, 1)
        assert by_layout["html"].passed == 2
        assert by_layout["lines"].passed == 0

    def test_an_empty_set_counts_nothing(self):
        totals = totals_of([])
        assert (totals.sampled, totals.fetched, totals.passed, totals.neighbour_checked) == (0,) * 4


class TestVerdict:
    def test_says_when_nothing_was_fetched(self):
        missing = check_for(ref(), text_of(), PageResponse(status=404, error="HTTP 404"))
        verdict = verdict_of(totals_of([missing]))
        assert "not checked" in verdict
        assert "budget" not in verdict

    def test_states_every_count(self):
        checks = [
            checked(ref(36), neighbours=both_neighbours()),
            checked(ref(36), page(paragraphs=PARAGRAPHS[:2]), neighbours=both_neighbours()),
            checked(ref(36), page_with_previous(), neighbours=both_neighbours()),
            checked(ref(36, "HC094")),
        ]
        verdict = verdict_of(totals_of(checks))
        assert verdict.startswith("Speeches sampled: 4. Pages that came back with status 200: 4.")
        assert "pages that name the speech and its speaker: 4;" in verdict
        assert "the opening words of the speech's first paragraph: 4;" in verdict
        assert "the opening words of every paragraph of the speech in order: 3." in verdict
        assert (
            "Pages judged against at least one neighbouring speech: 3; of these, pages that "
            "hold none of their neighbours: 2." in verdict
        )
        assert (
            "Pages that pass every judgement: 2; of these, pages judged against at least one "
            "neighbouring speech: 1." in verdict
        )
        assert "budget" not in verdict

    def test_leaves_out_the_neighbour_judgement_when_no_page_had_one(self):
        verdict = verdict_of(totals_of([checked(ref(36))]))
        assert "Pages judged against" not in verdict
        assert "Pages that pass every judgement: 1;" in verdict

    def test_says_how_many_pages_the_budget_left_unrequested(self):
        checks = [
            checked(ref(36)),
            check_for(ref(5, "HC094"), text_of(5), PageResponse(error=BUDGET_EXHAUSTED)),
        ]
        verdict = verdict_of(totals_of(checks))
        assert verdict.endswith(
            "Sampled pages never requested, because the request budget ran out: 1."
        )

    def test_says_how_many_pages_the_budget_left_unrequested_when_none_came_back(self):
        unrequested = check_for(ref(), text_of(), PageResponse(error=BUDGET_EXHAUSTED))
        verdict = verdict_of(totals_of([unrequested]))
        assert "not checked" in verdict
        assert "never requested, because the request budget ran out: 1." in verdict

    def test_says_how_many_retries_the_budget_cut_off(self):
        checks = [
            checked(ref(36)),
            check_for(
                ref(5, "HC094"),
                text_of(5),
                PageResponse(status=503, error=f"HTTP 503; {BUDGET_EXHAUSTED}"),
            ),
        ]
        verdict = verdict_of(totals_of(checks))
        assert verdict.endswith(
            "Sampled pages that failed and were not requested again, because the request "
            "budget ran out: 1."
        )
        assert "never requested" not in verdict


class TestReport:
    def checks(self):
        return [
            checked(ref(36), neighbours=both_neighbours()),
            checked(ref(37, "HC094", "2003/04", date(2003, 10, 1)), page()),
        ]

    def build(self, checks: list[PageCheck] | None = None) -> DeepLinkReport:
        budget = RequestBudget(10)
        budget.spend()
        return report_of(
            checks if checks is not None else self.checks(),
            inputs_sha256="ab" * 32,
            corpus_speeches=301,
            seed=5,
            per_session=1,
            extra_recent_sessions=0,
            budget=budget,
        )

    def test_carries_the_seed_the_request_count_and_one_entry_per_speech(self):
        report = self.build()
        assert (report.seed, report.request_limit, report.requests_made) == (5, 10, 1)
        assert [entry.anf_id for entry in report.entries] == ["HC093-36", "HC094-37"]
        assert report.totals.sampled == 2
        assert list(report.by_session) == ["2003/04", "2024/25"]
        assert list(report.by_layout) == ["html"]

    def test_names_the_corpus_the_sample_was_drawn_from(self):
        report = self.build()
        assert (report.inputs_sha256, report.corpus_speeches) == ("ab" * 32, 301)

    def test_records_the_address_that_was_fetched_in_every_entry(self):
        urls = [entry.url for entry in self.build().entries]
        assert urls == [deep_link("HC093", 36), deep_link("HC094", 37)]

    def test_says_how_a_page_is_judged_and_what_the_check_does_not_cover(self):
        report = self.build()
        assert f"first {OPENING_CHARS} characters of every paragraph" in report.judgement
        assert "in order" in report.judgement
        assert "the speech before and the speech after" in report.judgement
        assert "nothing was rendered" in report.limitation
        assert "One response per address was examined" in report.limitation
        assert "requested again, up to the number of attempts" in report.limitation
        assert "inside a paragraph, after its opening words" in report.limitation
        assert "for the first paragraph, anywhere earlier on the page" in report.limitation
        assert "for the last, anywhere later" in report.limitation
        assert "a second copy of the speech" in report.limitation
        assert "a neighbour present only in part" in report.limitation
        assert "XML record" in report.limitation
        assert "Entries of this run" not in report.limitation

    def test_names_the_entries_whose_page_is_the_xml_record_only_when_there_are_some(self):
        record = checked(ref(36, "HC095"), xml_record(), neighbours=both_neighbours())
        assert found(record) == (False, True, True)
        report = self.build([*self.checks(), record])
        assert report.limitation.endswith("as the API's XML record does: HC095-36.")
        assert "<anforande_nummer> and <anforandetext>" in report.limitation

    def test_does_not_name_a_page_that_names_the_speech(self):
        fields = "<anforande_nummer>36</anforande_nummer><anforandetext></anforandetext>"
        record = checked(ref(36, "HC095"), page() + fields)
        assert record.xml_record and record.names_speech
        assert "Entries of this run" not in self.build([record]).limitation

    def test_does_not_name_an_xml_record_that_lacks_the_speech(self):
        empty = xml_record().replace(PARAGRAPHS[0][:20], "")
        record = checked(ref(36, "HC095"), empty)
        assert record.xml_record and not record.first_opening_found
        assert "Entries of this run" not in self.build([record]).limitation

    def test_keeps_a_short_account_of_the_two_forms_that_do_not_reach_the_speech(self):
        forms = self.build().other_address_forms
        assert len(forms) == 2
        protocol_page, full_text_page = forms
        assert "On 23 of the 24 pages fetched" in protocol_page.result
        assert "1 of the 24" in protocol_page.result
        assert "2024/25:139" in protocol_page.result
        assert "2026-11-20" in protocol_page.result
        assert "2026-11-20" in full_text_page.result
        assert "4 of 8" in full_text_page.result
        assert "before 2014/15" in full_text_page.result
        assert "protokoll_<dok_id>/#anf<number>" in protocol_page.address
        assert "protokoll_url_www" in protocol_page.address
        assert "/html/#anf<number>" in full_text_page.address
        assert all(form.how_established for form in forms)

    def test_is_deterministic_and_holds_no_time_or_path(self):
        first, second = self.build(), self.build()
        assert first.model_dump_json() == second.model_dump_json()
        text = first.model_dump_json()
        assert "\\\\" not in text
        assert ":/" not in text.replace("https://", "")

    def test_holds_no_dash_the_house_style_forbids(self):
        text = self.build().model_dump_json(indent=2)
        assert chr(0x2014) not in text
        assert chr(0x2013) not in text

    def test_round_trips_through_json(self):
        report = self.build()
        assert DeepLinkReport.model_validate(json.loads(report.model_dump_json())) == report

    def test_cannot_be_changed(self):
        with pytest.raises(ValidationError):
            self.build().seed = 6


def load_check_script():
    """Import search/pipeline/check_deep_links.py, whose name is not a module name."""
    path = Path(__file__).resolve().parents[2] / "search" / "pipeline" / "check_deep_links.py"
    spec = importlib.util.spec_from_file_location("check_deep_links", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckScript:
    """The script's stages, against a fake host and a small synthetic corpus."""

    script = load_check_script()

    def texts(self) -> dict[str, SpeechText]:
        return {
            "HC093-36": text_of(36),
            "HC093-37": neighbour_text(),
            "HC094-5": text_of(5, dok_id="HC094"),
        }

    def run_check(
        self,
        host: FakeHost,
        sample: list[SpeechRef],
        limit: int = 50,
        pages_directory: Path | None = None,
    ):
        budget = RequestBudget(limit)
        clock = FakeClock()

        async def run() -> list[PageCheck]:
            throttle = HostThrottle(0.0, sleep=clock.sleep, now=clock.time)
            async with httpx.AsyncClient(transport=httpx.MockTransport(host)) as client:
                return await self.script.check_speeches(
                    sample,
                    self.texts(),
                    client=client,
                    throttle=throttle,
                    budget=budget,
                    concurrency=2,
                    attempts=1,
                    pages_directory=pages_directory,
                )

        return asyncio.run(run()), budget

    def test_fetches_each_speech_from_its_computed_address_once(self):
        other = page(5, SPEAKER, PARAGRAPHS)
        host = FakeHost(
            {
                deep_link("HC093", 36): httpx.Response(200, text=page()),
                deep_link("HC094", 5): httpx.Response(200, text=other),
            }
        )
        checks, budget = self.run_check(host, [ref(36), ref(5, "HC094")])
        assert sorted(host.requests) == [deep_link("HC093", 36), deep_link("HC094", 5)]
        assert set(host.requests.values()) == {1}
        assert budget.used == 2
        assert [check.passed for check in checks] == [True, True]

    def test_every_request_goes_to_the_open_data_host(self):
        host = FakeHost({deep_link("HC093", 36): httpx.Response(200, text=page())})
        self.run_check(host, [ref(36)])
        assert {url.split("/")[2] for url in host.requests} == {"data.riksdagen.se"}

    def test_the_neighbouring_speech_is_read_from_the_corpus_and_rejected(self):
        host = FakeHost({deep_link("HC093", 36): httpx.Response(200, text=page())})
        checks, _ = self.run_check(host, [ref(36)])
        assert (checks[0].neighbour_anf_ids, checks[0].neighbour_rejected) == (("HC093-37",), True)

    def test_a_page_of_another_speech_fails(self):
        wrong = page(37, OTHER_SPEAKER, NEIGHBOUR_PARAGRAPHS)
        host = FakeHost({deep_link("HC093", 36): httpx.Response(200, text=wrong)})
        checks, _ = self.run_check(host, [ref(36)])
        assert not checks[0].passed

    def test_a_missing_page_fails_and_does_not_stop_the_others(self):
        host = FakeHost({deep_link("HC094", 5): httpx.Response(200, text=page(5))})
        checks, _ = self.run_check(host, [ref(36), ref(5, "HC094")])
        assert [(c.anf_id, c.status, c.passed) for c in checks] == [
            ("HC093-36", 404, False),
            ("HC094-5", 200, True),
        ]

    def test_stops_asking_when_the_request_limit_is_reached(self):
        host = FakeHost(
            {
                deep_link("HC093", 36): httpx.Response(200, text=page()),
                deep_link("HC094", 5): httpx.Response(200, text=page(5)),
            }
        )
        checks, budget = self.run_check(host, [ref(36), ref(5, "HC094")], limit=1)
        assert budget.used == 1
        assert sum(host.requests.values()) == 1
        assert [check.error for check in checks].count(BUDGET_EXHAUSTED) == 1

    def test_saves_the_pages_that_came_back_when_asked_to(self, tmp_path):
        host = FakeHost({deep_link("HC093", 36): httpx.Response(200, text=page())})
        self.run_check(host, [ref(36), ref(5, "HC094")], pages_directory=tmp_path / "pages")
        assert [path.name for path in (tmp_path / "pages").iterdir()] == ["HC093-36.html"]
        assert (tmp_path / "pages" / "HC093-36.html").read_text(encoding="utf-8") == page()

    def write_corpus(self, path: Path) -> None:
        """Write a corpus file whose url column holds an address the script must not read."""
        rows = [
            ("HC093-36", "HC093", 36, "2024/25", date(2024, 9, 4), SPEAKER, PARAGRAPHS),
            ("HC093-37", "HC093", 37, "2024/25", date(2024, 9, 4), OTHER_SPEAKER, ("Tack.",)),
            ("HC094-5", "HC094", 5, "2024/25", date(2024, 9, 5), SPEAKER, PARAGRAPHS[:2]),
        ]
        old = (
            "https://www.riksdagen.se/sv/dokument-och-lagar/dokument/protokoll/protokoll_{}/#anf{}"
        )
        table = pa.table(
            {
                "anf_id": [r[0] for r in rows],
                "dok_id": [r[1] for r in rows],
                "anf_nummer": pa.array([r[2] for r in rows], pa.int32()),
                "rm": [r[3] for r in rows],
                "date": pa.array([r[4] for r in rows], pa.date32()),
                "speaker_raw": [r[5] for r in rows],
                "layout": ["html"] * 3,
                "paragraphs": [list(r[6]) for r in rows],
                "url": [old.format(r[1], r[2]) for r in rows],
            }
        )
        pq.write_table(table, path)

    def test_reads_the_identity_of_every_speech_and_computes_the_address(self, tmp_path):
        self.write_corpus(tmp_path / "anforanden.parquet")
        speeches = self.script.load_speech_refs(tmp_path / "anforanden.parquet")
        assert [speech.anf_id for speech in speeches] == ["HC093-36", "HC093-37", "HC094-5"]
        assert speeches[0].date == date(2024, 9, 4)
        assert speeches[0].url == ADDRESS

    def test_reads_the_speaker_and_every_paragraph_of_the_given_protocols(self, tmp_path):
        self.write_corpus(tmp_path / "anforanden.parquet")
        texts = self.script.load_speech_texts(tmp_path / "anforanden.parquet", ["HC093"])
        assert sorted(texts) == ["HC093-36", "HC093-37"]
        assert texts["HC093-36"] == text_of(36)
        assert texts["HC093-36"].paragraphs == PARAGRAPHS
        assert texts["HC093-37"].paragraphs == ("Tack.",)

    def test_refuses_a_corpus_that_its_build_report_does_not_describe(self, tmp_path):
        self.write_corpus(tmp_path / "anforanden.parquet")
        build = SimpleNamespace(speeches=4, inputs_sha256="ab" * 32)
        with pytest.raises(SystemExit, match="holds 3 speeches and its build report 4"):
            asyncio.run(self.script.run(SimpleNamespace(), tmp_path / "anforanden.parquet", build))

    def test_the_report_names_the_corpus_of_its_build_report(self, tmp_path):
        self.write_corpus(tmp_path / "anforanden.parquet")
        build = SimpleNamespace(speeches=3, inputs_sha256="cd" * 32)
        args = SimpleNamespace(
            seed=7,
            per_session=2,
            extra_recent_sessions=0,
            max_requests=10,
            min_interval=0.0,
            timeout=5.0,
            concurrency=1,
            attempts=1,
            save_pages=None,
        )
        host = FakeHost(
            {
                deep_link("HC093", 36): httpx.Response(200, text=page()),
                deep_link("HC093", 37): httpx.Response(200, text=page(37, OTHER_SPEAKER)),
                deep_link("HC094", 5): httpx.Response(200, text=page(5)),
            }
        )
        report = asyncio.run(
            self.script.run(
                args, tmp_path / "anforanden.parquet", build, transport=httpx.MockTransport(host)
            )
        )
        assert (report.inputs_sha256, report.corpus_speeches) == ("cd" * 32, 3)
        assert (report.seed, report.totals.sampled, report.requests_made) == (7, 2, 2)

    @pytest.fixture
    def parsed_arguments(self, tmp_path, monkeypatch):
        """Run the script's main with a given command line; return the arguments it parsed."""
        seen: list = []

        async def fake_run(args, corpus, build):
            seen.append(args)
            return SimpleNamespace(verdict="", requests_made=0, request_limit=0)

        monkeypatch.setattr(self.script, "run", fake_run)
        monkeypatch.setattr(self.script, "load_build_report", lambda path: None)
        monkeypatch.setattr(self.script, "write_report", lambda report, path: None)

        def run_with(*options: str):
            argv = ["check_deep_links.py", "--data-root", str(tmp_path), *options]
            monkeypatch.setattr(sys, "argv", argv)
            self.script.main()
            return seen[-1]

        return run_with

    def test_the_seed_and_request_limit_default_to_the_links_settings(self, parsed_arguments):
        args = parsed_arguments()
        links = load_settings().links
        assert (args.seed, args.max_requests) == (links.seed, links.max_requests)

    def test_the_seed_and_request_limit_can_be_given_on_the_command_line(self, parsed_arguments):
        args = parsed_arguments("--seed", "7", "--max-requests", "5")
        assert (args.seed, args.max_requests) == (7, 5)
