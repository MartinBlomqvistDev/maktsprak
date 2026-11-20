"""Tests for search/corpus/parse.py."""

from __future__ import annotations

import pytest

from search.corpus.parse import (
    detect_layout,
    html_paragraphs,
    line_paragraphs,
    margin_heading,
    normalise,
    read_paragraphs,
    split_speech_from_record,
)

NO_HEADINGS: frozenset[str] = frozenset()


class TestLayout:
    def test_paragraph_elements_mean_html(self):
        assert detect_layout("<p>Fru talman!</p>") == "html"

    def test_carriage_returns_mean_one_paragraph_per_line(self):
        assert detect_layout("\r\nFru talman! Jag tackar för svaret.\r\n") == "lines"

    def test_bare_line_feeds_mean_the_printed_column(self):
        assert (
            detect_layout("Fru talman! Många barnfamil-\njer har haft det jobbigt.\n") == "wrapped"
        )

    def test_the_wrapped_layout_is_not_read(self):
        assert read_paragraphs("Många barnfamil-\njer har haft det jobbigt.\n") is None


class TestHtml:
    def test_each_element_is_a_paragraph(self):
        assert html_paragraphs("<p>Fru talman!</p><p>Det är svårt att säga.</p>") == [
            "Fru talman!",
            "Det är svårt att säga.",
        ]

    def test_inline_tags_leave_no_trace_and_add_no_space(self):
        text = "<p>Det är <em>inte</em> sant, och o<strong>möjligt</strong>.</p>"
        assert html_paragraphs(text) == ["Det är inte sant, och omöjligt."]

    def test_entities_are_decoded(self):
        assert html_paragraphs("<p>Forskning &amp; utveckling</p>") == ["Forskning & utveckling"]

    def test_empty_elements_are_dropped(self):
        assert html_paragraphs("<p></p><p> </p><p>Herr talman!</p>") == ["Herr talman!"]

    def test_whitespace_is_collapsed(self):
        assert html_paragraphs("<p>  Herr   talman!\n Tack.  </p>") == ["Herr talman! Tack."]

    def test_text_outside_the_elements_raises(self):
        with pytest.raises(ValueError, match="outside the paragraph elements"):
            html_paragraphs("<p>Herr talman!</p>Detta står utanför.")


class TestLines:
    def test_each_line_is_a_paragraph(self):
        text = "\r\nFru talman! Tack för svaret.\r\nJag har en fråga till.\r\n"
        assert line_paragraphs(text) == ["Fru talman! Tack för svaret.", "Jag har en fråga till."]

    def test_an_emphasised_run_is_joined_back_as_it_stands(self):
        text = "\r\nMan får inte ha bara \r\nen\r\n kund eller \r\nen\r\n uppdragsgivare.\r\n"
        assert line_paragraphs(text) == ["Man får inte ha bara en kund eller en uppdragsgivare."]

    def test_closing_punctuation_on_its_own_line_is_joined_without_a_space(self):
        assert line_paragraphs("\r\nDe var mycket \r\nservice-minded\r\n.\r\n") == [
            "De var mycket service-minded."
        ]

    def test_a_trailing_space_after_a_finished_sentence_is_a_paragraph_break(self):
        text = "\r\nDet är billigare att importera kolkraft. \r\nDet tror jag att vi kan lösa.\r\n"
        assert line_paragraphs(text) == [
            "Det är billigare att importera kolkraft.",
            "Det tror jag att vi kan lösa.",
        ]

    def test_an_opening_quotation_mark_starts_a_new_paragraph(self):
        text = '\r\nHan sade:\r\n"Men vi kan inte avskaffa fängelserna."\r\n'
        assert line_paragraphs(text) == ["Han sade:", '"Men vi kan inte avskaffa fängelserna."']

    def test_lines_with_no_space_between_them_are_never_glued(self):
        assert line_paragraphs("\r\nseparate but\r\nequal\r\n") == ["separate but", "equal"]


class TestSpeechAndRecord:
    def test_plain_speech_is_kept_whole(self):
        paragraphs = ["Fru talman! Tack för svaret.", "Jag yrkar bifall till reservation 3."]
        split = split_speech_from_record(paragraphs, NO_HEADINGS)
        assert split.paragraphs == tuple(paragraphs)
        assert split.notes == ()
        assert split.furniture == 0

    def test_applause_becomes_a_note_and_the_speech_goes_on(self):
        split = split_speech_from_record(
            ["Vi vann valet.", "(Applåder)", "Nu börjar arbetet."], NO_HEADINGS
        )
        assert split.paragraphs == ("Vi vann valet.", "Nu börjar arbetet.")
        assert split.notes == ("(Applåder)",)

    def test_an_interjection_by_somebody_else_is_never_the_speakers(self):
        split = split_speech_from_record(
            ["Det går bra.", "(MATS ODELL (kd): Det hjälper ju inte!)", "Jo, det gör det."],
            NO_HEADINGS,
        )
        assert split.paragraphs == ("Det går bra.", "Jo, det gör det.")
        assert split.notes == ("(MATS ODELL (kd): Det hjälper ju inte!)",)

    @pytest.mark.parametrize(
        "closing",
        [
            "Överläggningen var härmed avslutad.",
            "Interpellationsdebatten var härmed avslutad.",
            "Den utrikespolitiska debatten var härmed avslutad.",
            "Överläggningen av härmed avslutad.",
            "Överläggningen var härmed avslutad",
            "Härmed var överläggningen avslutad.",
            "I detta anförande instämde Johan Pehrson (fp).",
            "Andre vice talmannen konstaterade att interpellanten inte var närvarande i kammaren.",
            "Talmannen meddelade att en promemoria hade delats ut till kammarens ledamöter.",
            "Kammaren biföll konstitutionsutskottets förslag och valde därmed",
            "Kammaren valde i enlighet med konstitutionsutskottets förslag till",
            "Då Åsa Domeij, som framställt interpellationen, anmält att hon var förhindrad att "
            "närvara vid sammanträdet medgav tredje vice talmannen att Ingegerd Saarinen i "
            "stället fick delta i överläggningen.",
        ],
    )
    def test_the_record_closing_a_speech_takes_everything_after_it(self, closing):
        split = split_speech_from_record(
            ["Tack för debatten!", closing, "(Beslut fattades under § 14.)", "Votering:"],
            NO_HEADINGS,
        )
        assert split.paragraphs == ("Tack för debatten!",)
        assert split.notes == (closing, "(Beslut fattades under § 14.)", "Votering:")

    @pytest.mark.parametrize(
        "speech",
        [
            "Härmed är mitt anförande avslutat.",
            "Talmannen har ju uppmanat oss att fatta oss kort.",
            "Kammaren beslutade i fjol om ett tillkännagivande till regeringen.",
            "Då kan talmannen säga: Den nuvarande regeringen bör sitta kvar.",
            "Jag vill precis som talmannen påpekade säga att jag svarar för regeringen.",
            "Det är (som bekant) inte första gången.",
        ],
    )
    def test_speech_that_resembles_the_record_stays_speech(self, speech):
        split = split_speech_from_record([speech, "Tack."], NO_HEADINGS)
        assert split.paragraphs == (speech, "Tack.")
        assert split.notes == ()

    def test_a_long_paragraph_that_ends_like_a_closing_note_stays_speech(self):
        long = (
            "Jag vill säga att "
            + "vi har arbetat länge med detta och " * 4
            + ("debatten är inte härmed avslutad.")
        )
        assert split_speech_from_record([long], NO_HEADINGS).paragraphs == (long,)

    def test_field_code_residue_is_dropped_and_counted(self):
        residue = "STYLEREF Kantrubrik \\* MERGEFORMAT Svar på interpellationer"
        split = split_speech_from_record(["Fru talman!", residue, "Tack för svaret."], NO_HEADINGS)
        assert split.paragraphs == ("Fru talman!", "Tack för svaret.")
        assert split.notes == ()
        assert split.furniture == 1

    def test_a_paragraph_that_is_only_a_heading_is_dropped(self):
        headings = frozenset({"Svar på interpellationer"})
        split = split_speech_from_record(
            ["Fru talman!", "Svar på interpellationer", "Svar på interpellationer är viktiga."],
            headings,
        )
        assert split.paragraphs == ("Fru talman!", "Svar på interpellationer är viktiga.")
        assert split.furniture == 1

    def test_a_record_with_no_speech_leaves_nothing_to_quote(self):
        split = split_speech_from_record(["(Applåder)"], NO_HEADINGS)
        assert split.paragraphs == ()


class TestHelpers:
    def test_normalise_collapses_every_kind_of_whitespace(self):
        assert normalise(" Herr talman!\r\n\tTack. ") == "Herr talman! Tack."

    def test_the_margin_heading_is_read_out_of_the_residue(self):
        residue = "STYLEREF Kantrubrik \\* MERGEFORMAT Svar på interpellationer"
        assert margin_heading(residue) == "Svar på interpellationer"

    def test_ordinary_text_carries_no_margin_heading(self):
        assert margin_heading("Svar på interpellationer") is None
