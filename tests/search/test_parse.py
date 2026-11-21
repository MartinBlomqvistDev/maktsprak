"""Tests for search/corpus/parse.py."""

from __future__ import annotations

import pytest

from search.corpus.parse import (
    detect_layout,
    html_paragraphs,
    index_headings,
    line_paragraphs,
    lost_emphasis_boundaries,
    margin_heading,
    normalise,
    read_paragraphs,
    split_speech_from_record,
)


def rules(split) -> list[str]:
    """List the rules that set paragraphs aside, in order."""
    return [item.rule for item in split.set_aside]


class TestLayout:
    def test_paragraph_elements_mean_html(self):
        assert detect_layout("<p>Fru talman!</p>") == "html"

    def test_paragraph_elements_in_capitals_mean_html_as_well(self):
        text = "<P>Herr talman!</P>\r\n<P>Tack.</P>"
        assert detect_layout(text) == "html"
        assert read_paragraphs(text) == ["Herr talman!", "Tack."]

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

    def test_a_last_paragraph_left_open_keeps_its_text(self):
        assert html_paragraphs("<p>Herr talman!</p><p>Sista stycket utan slut") == [
            "Herr talman!",
            "Sista stycket utan slut",
        ]

    def test_a_paragraph_left_open_ends_where_the_next_one_opens(self):
        assert html_paragraphs("<p>Hela talet<p>Tack.") == ["Hela talet", "Tack."]

    @pytest.mark.parametrize(
        "text",
        ["<p>första raden<br>andra raden</p>", "<p>a<div>b</div></p>", "<p>a<br/>b</p>"],
    )
    def test_an_element_the_source_does_not_use_raises(self, text):
        with pytest.raises(ValueError, match="Unknown element"):
            html_paragraphs(text)


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

    def test_entities_are_decoded(self):
        text = "\r\nblev Vin &amp; Sprit en av många hundratals leverantörer.\r\n"
        assert line_paragraphs(text) == ["blev Vin & Sprit en av många hundratals leverantörer."]

    def test_the_soft_hyphen_is_dropped(self):
        assert line_paragraphs("\r\nen rätts\u00adstat att kunna vägra\r\n") == [
            "en rättsstat att kunna vägra"
        ]

    def test_a_line_with_nothing_visible_is_blank(self):
        assert line_paragraphs("\r\nTack.\r\n\u00ad\r\n \r\n") == ["Tack."]

    def test_a_bare_line_feed_ends_a_line(self):
        text = "\r\nländer. Framtidsdrömmarna är bortsopade.\nFattigdomen breder ut sig.\r\n"
        assert line_paragraphs(text) == [
            "länder. Framtidsdrömmarna är bortsopade.",
            "Fattigdomen breder ut sig.",
        ]

    def test_a_bare_line_feed_inside_a_sentence_still_joins(self):
        text = "\r\nden mycket trevliga \nsamvaro vi hade där.\r\n"
        assert line_paragraphs(text) == ["den mycket trevliga samvaro vi hade där."]

    def test_the_tail_of_a_compound_is_joined_to_its_emphasised_head(self):
        text = "\r\nkonsekvenser det blir för public service\r\n-företagen. Oberoende Filmares\r\n"
        assert line_paragraphs(text) == [
            "konsekvenser det blir för public service-företagen. Oberoende Filmares"
        ]

    def test_a_list_led_by_dashes_after_a_colon_is_not_joined(self):
        text = "\r\nFrågorna om äganderätten är:\r\n-Kan en ordning med intrång försvaras?\r\n"
        assert line_paragraphs(text) == [
            "Frågorna om äganderätten är:",
            "-Kan en ordning med intrång försvaras?",
        ]

    def test_an_emphasised_run_after_a_finished_sentence_stays_in_the_paragraph(self):
        text = "\r\nVi har stora utmaningar. \r\nEn\r\n mening handlar om klimatet.\r\n"
        assert line_paragraphs(text) == ["Vi har stora utmaningar. En mening handlar om klimatet."]

    def test_an_emphasised_run_after_a_salutation_stays_in_the_paragraph(self):
        text = "\r\nFru talman! \r\nOm\r\n det har pågått oegentligheter är det allvarligt.\r\n"
        assert line_paragraphs(text) == [
            "Fru talman! Om det har pågått oegentligheter är det allvarligt."
        ]

    def test_an_emphasised_run_after_a_colon_stays_in_the_paragraph(self):
        text = "\r\nJag vill betona: \r\nännu inte\r\n har presenterat förslag.\r\n"
        assert line_paragraphs(text) == ["Jag vill betona: ännu inte har presenterat förslag."]

    def test_whole_sentences_after_a_colon_stay_paragraphs_of_their_own(self):
        text = "\r\nJag frågar: \r\nNär kommer det?\r\nÄr ministern beredd?\r\n"
        assert line_paragraphs(text) == ["Jag frågar:", "När kommer det?", "Är ministern beredd?"]

    def test_a_title_quoted_after_a_colon_stays_in_the_sentence_whatever_its_length(self):
        title = (
            "System för exportkontroll för produkter med dubbla användningsområden från EU: "
            "säkerhet och konkurrenskraft i en föränderlig värld"
        )
        opening = "Herr talman! Vi har i dag att ta ställning till en grönbok:"
        text = f"\r\n{opening} \r\n{title}\r\n.\r\n"
        assert line_paragraphs(text) == [f"{opening} {title}."]

    def test_a_long_line_after_a_finished_sentence_is_a_paragraph_of_its_own(self):
        text = (
            "\r\nI allmänhet är det fråga om viss interregional trafik, som då hanteras av "
            "Rikstrafiken. \r\nStaten satsar via Rikstrafiken stora belopp på att upprätthålla en "
            "tillfredsställande transportförsörjning runtom i vårt land. Reduceringen bör ses i "
            "ljuset av detta.\r\n Avslutningsvis kan jag nämna att Kollektivtrafikutredningen "
            "också har behandlat kollektivtrafik i glesbygd.\r\n"
        )
        assert line_paragraphs(text) == [
            "I allmänhet är det fråga om viss interregional trafik, som då hanteras av "
            "Rikstrafiken.",
            "Staten satsar via Rikstrafiken stora belopp på att upprätthålla en "
            "tillfredsställande transportförsörjning runtom i vårt land. Reduceringen bör ses i "
            "ljuset av detta.",
            "Avslutningsvis kan jag nämna att Kollektivtrafikutredningen också har behandlat "
            "kollektivtrafik i glesbygd.",
        ]

    def test_a_salutation_after_a_finished_sentence_is_no_emphasised_run(self):
        text = (
            "\r\nDet är bra för regionen. \r\nHerr talman! Jag yrkar bifall.\r\n"
            " Jag vill också säga något om skolan.\r\n"
        )
        assert line_paragraphs(text) == [
            "Det är bra för regionen.",
            "Herr talman! Jag yrkar bifall. Jag vill också säga något om skolan.",
        ]

    def test_a_long_line_with_a_stray_leading_space_is_a_new_paragraph(self):
        text = (
            "\r\nMan tar ut en avgift om transportören inte uppfyller den kontrollskyldigheten."
            "\r\n Fru talman! Det utvidgade transportörsansvaret har diskuterats mycket.\r\n"
        )
        assert line_paragraphs(text) == [
            "Man tar ut en avgift om transportören inte uppfyller den kontrollskyldigheten.",
            "Fru talman! Det utvidgade transportörsansvaret har diskuterats mycket.",
        ]

    def test_a_salutation_with_a_stray_leading_space_opens_a_paragraph_whatever_its_length(self):
        text = (
            "\r\nBristerna i regeringens bostadspolitik tycks påverka räntan, vilket är olyckligt "
            "för samhällsekonomin.\r\n Herr talman! Jag yrkar bifall till finansutskottet "
            "förslag. \r\n"
        )
        assert line_paragraphs(text) == [
            "Bristerna i regeringens bostadspolitik tycks påverka räntan, vilket är olyckligt "
            "för samhällsekonomin.",
            "Herr talman! Jag yrkar bifall till finansutskottet förslag.",
        ]

    def test_a_leading_space_after_an_emphasised_run_continues_the_paragraph(self):
        text = (
            "\r\nSocialdemokraterna uttalar i detta betänkande att den arsenal av stödformer som i "
            "dag finns inom regionalpolitiken måste finnas kvar. Man påstår att enbart generella "
            "åtgärder inte räcker. \r\nThe same procedure as last year. The same procedure as "
            "every year.\r\n Trots att vi år efter år kan konstatera att regionalpolitiken ger "
            "dåligt utbyte fortsätter man i samma spår.\r\n"
        )
        assert len(line_paragraphs(text)) == 1

    def test_a_leading_space_after_a_short_line_continues_the_paragraph(self):
        text = (
            "\r\nOmsättningsskatten infördes då genom proposition nr 1959:168 \r\n"
            "Förordning om allmän\r\nvaruskatt.\r\n Den behandlades av dåvarande "
            "bevillningsutskottet, motsvarande skatteutskottet, och av konstitutionsutskottet.\r\n"
        )
        assert line_paragraphs(text) == [
            "Omsättningsskatten infördes då genom proposition nr 1959:168 Förordning om allmän",
            "varuskatt. Den behandlades av dåvarande bevillningsutskottet, motsvarande "
            "skatteutskottet, och av konstitutionsutskottet.",
        ]

    def test_a_leading_space_before_a_quotation_mark_continues_the_paragraph(self):
        text = (
            "\r\nDe här stora företagsorganisationerna som tillsammans har 242 000 medlemmar "
            "jämfört med Företagarnas 55 000 har skrivit ett pressmeddelande.\r\n"
            ' "Den analys av lagen som vi genomfört visar på en kraftig försämring för '
            'kapitalförsörjningen till småföretagen."\r\nSå skriver Företagarförbundet.\r\n'
        )
        assert line_paragraphs(text) == [
            "De här stora företagsorganisationerna som tillsammans har 242 000 medlemmar "
            "jämfört med Företagarnas 55 000 har skrivit ett pressmeddelande. "
            '"Den analys av lagen som vi genomfört visar på en kraftig försämring för '
            'kapitalförsörjningen till småföretagen."',
            "Så skriver Företagarförbundet.",
        ]

    def test_bullets_the_source_replaced_with_question_marks_stay_list_items(self):
        text = (
            "\r\nBland anslagstillskotten kan nämnas\r\n?åtgärdspaket mot arbetslösheten\r\n"
            "?nytt filmavtal - ytterligare 55 miljoner\r\n? ökade medel till teateralliansen\r\n"
            "Det är bra.\r\n? Alla ska ha rätt till språk.\r\n"
        )
        assert line_paragraphs(text) == [
            "Bland anslagstillskotten kan nämnas",
            "?åtgärdspaket mot arbetslösheten",
            "?nytt filmavtal - ytterligare 55 miljoner",
            "? ökade medel till teateralliansen",
            "Det är bra.",
            "? Alla ska ha rätt till språk.",
        ]

    def test_a_question_mark_after_an_emphasised_run_is_joined(self):
        text = "\r\nVad menar man med \r\nsocial Europe\r\n? Det vet ingen.\r\n"
        assert line_paragraphs(text) == ["Vad menar man med social Europe? Det vet ingen."]

    def test_an_interjection_ends_its_paragraph_and_the_speaker_goes_on_after_it(self):
        text = (
            "\r\nVi hittade en möjlighet, herr talman .\r\n(TALMANNEN: Absolut!)\r\n"
            ". att låsa fast en del av propositionen.\r\n"
        )
        assert line_paragraphs(text) == [
            "Vi hittade en möjlighet, herr talman .",
            "(TALMANNEN: Absolut!)",
            ". att låsa fast en del av propositionen.",
        ]

    def test_applause_ends_its_paragraph(self):
        text = "\r\n(Applåder)\r\n Man måste samla i ladorna under goda år.\r\n"
        assert line_paragraphs(text) == ["(Applåder)", "Man måste samla i ladorna under goda år."]

    def test_an_interjection_that_emphasis_split_is_one_paragraph(self):
        assert line_paragraphs("\r\n(RAGNWI MARCELIND (kd): \r\nYes\r\n.)\r\n") == [
            "(RAGNWI MARCELIND (kd): Yes.)"
        ]

    def test_a_break_where_the_source_lost_the_spaces_is_kept_and_counted(self):
        text = "\r\nMan talar om w\r\nider Europe\r\n och g\r\nreater Middle East\r\n och allt.\r\n"
        assert line_paragraphs(text) == [
            "Man talar om w",
            "ider Europe och g",
            "reater Middle East och allt.",
        ]
        assert lost_emphasis_boundaries(text) == 2

    def test_other_layouts_have_no_lost_boundaries(self):
        assert lost_emphasis_boundaries("<p>Man talar om w</p><p>ider Europe</p>") == 0


class TestSpeechAndRecord:
    def test_plain_speech_is_kept_whole(self):
        paragraphs = ["Fru talman! Tack för svaret.", "Jag yrkar bifall till reservation 3."]
        split = split_speech_from_record(paragraphs, "html")
        assert split.paragraphs == tuple(paragraphs)
        assert split.notes == ()
        assert split.furniture == 0

    def test_applause_becomes_a_note_and_the_speech_goes_on(self):
        split = split_speech_from_record(
            ["Vi vann valet.", "(Applåder)", "Nu börjar arbetet."], "html"
        )
        assert split.paragraphs == ("Vi vann valet.", "Nu börjar arbetet.")
        assert split.notes == ("(Applåder)",)
        assert rules(split) == ["insertion"]

    def test_an_interjection_by_somebody_else_is_never_the_speakers(self):
        split = split_speech_from_record(
            ["Det går bra.", "(MATS ODELL (kd): Det hjälper ju inte!)", "Jo, det gör det."],
            "lines",
        )
        assert split.paragraphs == ("Det går bra.", "Jo, det gör det.")
        assert split.notes == ("(MATS ODELL (kd): Det hjälper ju inte!)",)

    @pytest.mark.parametrize(
        ("closing", "rule"),
        [
            ("Överläggningen var härmed avslutad.", "closing_note"),
            ("Interpellationsdebatten var härmed avslutad.", "closing_note"),
            ("Den utrikespolitiska debatten var härmed avslutad.", "closing_note"),
            ("Överläggningen av härmed avslutad.", "closing_note"),
            ("Överläggningen var härmed avslutad", "closing_note"),
            ("Härmed var överläggningen avslutad.", "closing_note"),
            ("Överläggningen var därmed avslutad.", "closing_note"),
            ("Överläggningen var härmad avslutad.", "closing_note"),
            ("Frågestunden var därmed avslutad.", "closing_note"),
            ("I detta anförande instämde Johan Pehrson (fp).", "concurrence"),
            (
                "Andre vice talmannen konstaterade att interpellanten inte var närvarande.",
                "chair_narration",
            ),
            (
                "Talmannen meddelade att en promemoria hade delats ut till kammarens ledamöter.",
                "chair_narration",
            ),
            (
                "Ålderspresidenten meddelade att interpellanten anmält att hon var förhindrad.",
                "chair_narration",
            ),
            (
                "Tjänstgörande ålderspresidenten konstaterade att interpellanten inte var "
                "närvarande i kammaren och förklarade interpellationsdebatten avslutad.",
                "chair_narration",
            ),
            (
                "Kammaren biföll konstitutionsutskottets förslag och valde därmed",
                "chamber_narration",
            ),
            (
                "Kammaren valde i enlighet med konstitutionsutskottets förslag till",
                "chamber_narration",
            ),
            (
                "Då Åsa Domeij, som framställt interpellationen, anmält att hon var förhindrad att "
                "närvara vid sammanträdet medgav tredje vice talmannen att Ingegerd Saarinen i "
                "stället fick delta i överläggningen.",
                "leave_of_absence",
            ),
            (
                "Då Alice Åström, som framställt interpellationen, anmält att hon var förhindrad "
                "att närvara vid sammanträdet förklarade talmannen överläggningen avslutad.",
                "leave_of_absence",
            ),
            (
                "Då interpellanten anmält att han var förhindrad att närvara vid sammanträdet "
                "medgav tjänstgörande ålderspresidenten att Jytte Guteland (S) i stället fick "
                "delta i interpellationsdebatten.",
                "leave_of_absence",
            ),
            ("Företogs val av 17 ledamöter i utskotten.", "election"),
            ("Val av 17 ledamöter i samtliga utskott företogs.", "election"),
            ("Efter härmed avslutad överläggning företogs val med slutna sedlar.", "election"),
            ("Från konstitutionsutskottet hade följande skrivelse inkommit:", "document_read"),
            (
                "Följande skrivelser med föreslagna namn hade kommit in från "
                "konstitutionsutskottet:",
                "document_read",
            ),
            ("Den överlämnade skrivelsen hade följande lydelse:", "document_read"),
            ("Följande skrivelse hade kommit in:", "document_read"),
            (
                "Håkan Svenneling m.fl. (V) hade, med stöd av 2 kap. 22 § första stycket "
                "regeringsformen, genom skrivelser framställt yrkande om att lagförslag skulle "
                "vila i minst tolv månader.",
                "demand_narration",
            ),
            (
                "Konstitutionsutskottet hade enhälligt föreslagit att professorn, jur.dr "
                "Hans-Gunnar Axberger skulle väljas till ny justitieombudsman.",
                "committee_narration",
            ),
            ("Votering:", "vote"),
        ],
    )
    def test_the_record_closing_a_speech_takes_everything_after_it(self, closing, rule):
        split = split_speech_from_record(
            ["Tack för debatten!", closing, "Till riksdagen", "Berit Andnor (s)"], "html"
        )
        assert split.paragraphs == ("Tack för debatten!",)
        assert split.notes == (closing, "Till riksdagen", "Berit Andnor (s)")
        assert rules(split) == [rule, "after_closing", "after_closing"]

    @pytest.mark.parametrize(
        "speech",
        [
            "Härmed är mitt anförande avslutat.",
            "Talmannen har ju uppmanat oss att fatta oss kort.",
            "Kammaren beslutade i fjol om ett tillkännagivande till regeringen.",
            "Kammaren biföll i våras en motion från Liberalerna och Erik Ullenhag om fortsatt "
            "arbete mot antiziganism och för romsk inkludering. Det blev ett tillkännagivande "
            "till regeringen. Hur försvarar Agneta Börjesson detta?",
            "Då kan talmannen säga: Den nuvarande regeringen bör sitta kvar.",
            "Då förklarade jag att den frågan var avslutad.",
            "Jag vill precis som talmannen påpekade säga att jag svarar för regeringen.",
            "Det är (som bekant) inte första gången.",
            "Vi föreslår därför att det nationella målet ska ha följande lydelse:",
            "Vi har, med stöd av 13 kap. regeringsformen, framställt yrkande om att pröva frågan.",
            "Voteringen i morgon innebär ett vägval.",
            "Frågan var därmed avslutad.",
            "Val av urvalsregler handlar i grunden om att välja vilka som ska antas till "
            "högskoleutbildning.",
            "I protokollet står det: Följande skrivelse hade kommit in: ett brev från länet.",
            "Kammaren valde i fjol att rikta ett tillkännagivande till regeringen.",
            "Anna Kinberg Batra (M): Vi ska sänka skatten. Så sade hon i valrörelsen.",
        ],
    )
    def test_speech_that_resembles_the_record_stays_speech(self, speech):
        split = split_speech_from_record([speech, "Tack."], "html")
        assert split.paragraphs == (speech, "Tack.")
        assert split.notes == ()

    def test_a_long_paragraph_that_ends_like_a_closing_note_stays_speech(self):
        long = (
            "Jag vill säga att "
            + "vi har arbetat länge med detta och " * 4
            + ("debatten är inte härmed avslutad.")
        )
        assert split_speech_from_record([long], "html").paragraphs == (long,)

    def test_a_record_with_no_speech_leaves_nothing_to_quote(self):
        split = split_speech_from_record(["(Applåder)"], "html")
        assert split.paragraphs == ()

    def test_an_interjection_given_without_parentheses_is_a_note(self):
        split = split_speech_from_record(
            ["Kan Jens Holm bekräfta det?", "JENS HOLM (V): Jodå. Det kan jag.", "Bra."], "lines"
        )
        assert split.paragraphs == ("Kan Jens Holm bekräfta det?", "Bra.")
        assert rules(split) == ["unbracketed_interjection"]


class TestInterjections:
    def test_an_interjection_over_several_paragraphs_is_set_aside_whole(self):
        paragraphs = [
            "Det börjar bli en vana.",
            "(TREDJE VICE TALMANNEN: Jag vill påminna om att överläggningarna inte får störas.",
            "Ordet går nu till statsrådet Johan Forssell.)",
            "Fru talman! Du är så välkommen.",
        ]
        split = split_speech_from_record(paragraphs, "html")
        assert split.paragraphs == ("Det börjar bli en vana.", "Fru talman! Du är så välkommen.")
        assert split.notes == tuple(paragraphs[1:3])
        assert rules(split) == ["interjection_span", "interjection_span"]

    def test_the_speakers_words_after_the_closing_parenthesis_are_kept_verbatim(self):
        split = split_speech_from_record(
            ["(GÖRAN HÄGGLUND (kd): Say no more!", "Ligg lågt!). har vi nästa på väg in."],
            "html",
        )
        assert split.notes == ("(GÖRAN HÄGGLUND (kd): Say no more!", "Ligg lågt!)")
        assert split.paragraphs == (". har vi nästa på väg in.",)

    def test_an_interjection_glued_to_the_speakers_words_is_cut_at_its_parenthesis(self):
        split = split_speech_from_record(
            ["(MATS ODELL (kd): Tack!). i nutid, i långtid och i mellantid."], "html"
        )
        assert split.notes == ("(MATS ODELL (kd): Tack!)",)
        assert split.paragraphs == (". i nutid, i långtid och i mellantid.",)
        assert rules(split) == ["interjection_cut"]

    def test_an_interjection_the_source_never_closes_is_set_aside_alone(self):
        split = split_speech_from_record(
            [
                "Fru talman!",
                "(GUNILLA CARLSSON i Tyresö (m): Jag sade att det var Holger Gustafsson.",
                "(FÖRSTE VICE TALMANNEN: Urban Ahlin har ordet.)",
                "Jag tycker att det är bra.",
            ],
            "lines",
        )
        assert split.paragraphs == ("Fru talman!", "Jag tycker att det är bra.")
        assert rules(split) == ["interjection_unclosed", "insertion"]

    def test_an_unclosed_interjection_never_runs_past_another_parenthesis(self):
        split = split_speech_from_record(
            [
                "(GUNILLA CARLSSON i Tyresö (m): Jag sade att det var Holger Gustafsson.",
                "(FÖRSTE VICE TALMANNEN: Urban Ahlin har ordet.)",
                "Jag har två punkter: 1) skolan och 2) vården.",
            ],
            "lines",
        )
        assert split.paragraphs == ("Jag har två punkter: 1) skolan och 2) vården.",)
        assert rules(split) == ["interjection_unclosed", "insertion"]

    def test_a_parenthesis_in_the_speakers_own_words_is_no_interjection(self):
        split = split_speech_from_record(
            ["(Se bilaga 2.) Det har jag redan sagt.", "Tack."], "html"
        )
        assert split.paragraphs == ("(Se bilaga 2.) Det har jag redan sagt.", "Tack.")

    def test_two_parentheses_around_the_speakers_words_are_no_insertion(self):
        paragraph = "(Applåder) Får jag bara påpeka att politiken ger resultat! (Applåder)"
        assert split_speech_from_record([paragraph], "html").paragraphs == (paragraph,)


class TestFurniture:
    def test_field_code_residue_is_dropped_and_counted(self):
        residue = "STYLEREF Kantrubrik \\* MERGEFORMAT Svar på interpellationer"
        split = split_speech_from_record(["Fru talman!", residue, "Tack för svaret."], "html")
        assert split.paragraphs == ("Fru talman!", "Tack för svaret.")
        assert split.notes == ()
        assert split.furniture == 1

    def test_residue_after_the_first_half_of_a_hyphenated_heading_is_furniture(self):
        residue = "Gransknings- STYLEREF Kantrubrik \\* MERGEFORMAT Fel! Ogiltig självreferens."
        split = split_speech_from_record(["Hennes uttalande.", residue, "Hennes andra."], "html")
        assert split.paragraphs == ("Hennes uttalande.", "Hennes andra.")
        assert rules(split) == ["field_code"]

    def test_residue_after_words_that_could_be_speech_raises(self):
        residue = "Jag vill säga att STYLEREF Kantrubrik \\* MERGEFORMAT Svar"
        with pytest.raises(ValueError, match="Field-code residue inside a paragraph"):
            split_speech_from_record([residue], "html")

    def test_a_paragraph_that_is_only_a_heading_is_dropped(self):
        headings = index_headings({"Svar på interpellationer"})
        split = split_speech_from_record(
            ["Fru talman!", "Svar på interpellationer", "Svar på interpellationer är viktiga."],
            "html",
            headings,
        )
        assert split.paragraphs == ("Fru talman!", "Svar på interpellationer är viktiga.")
        assert split.furniture == 1
        assert rules(split) == ["heading"]

    def test_a_heading_is_not_matched_in_the_lines_layout(self):
        headings = index_headings({"Biologisk mångfald"})
        paragraphs = ["Grundvatten av god kvalitet", "Biologisk mångfald", "Det är viktigt."]
        split = split_speech_from_record(paragraphs, "lines", headings)
        assert split.paragraphs == tuple(paragraphs)

    def test_a_heading_broken_over_paragraphs_is_dropped(self):
        headings = index_headings({"Bastjänstgöring för läkare"})
        split = split_speech_from_record(
            ["Det är ett bra förslag.", "Bastjänstgöring", "för läkare", "Vi stöder det."],
            "html",
            headings,
        )
        assert split.paragraphs == ("Det är ett bra förslag.", "Vi stöder det.")
        assert rules(split) == ["heading_pieces", "heading_pieces"]

    @pytest.mark.parametrize(
        ("heading", "broken"),
        [
            (
                "Ett starkt straffrättsligt skydd för den personliga integriteten",
                "Ett starkt straffrättsligt skydd för den person-liga integriteten",
            ),
            (
                "Strategisk exportkontroll 2015 \u2013 krigsmateriel och produkter",
                "Strategisk export-kontroll 2015 - krigsmateriel och produkter",
            ),
            (
                "Upphovsrätten på den digitala inre marknaden",
                "Upphovsrätten påden digitala inre marknaden",
            ),
        ],
    )
    def test_a_heading_broken_at_the_end_of_a_printed_line_is_dropped(self, heading, broken):
        headings = index_headings({heading})
        split = split_speech_from_record(["Fru talman!", broken, "Tack."], "html", headings)
        assert split.paragraphs == ("Fru talman!", "Tack.")
        assert rules(split) == ["heading_line_break"]

    def test_a_list_item_that_only_equals_another_debates_heading_stays_speech(self):
        headings = index_headings({"Ekonomisk tillväxt"})
        paragraphs = ["De är", "rättssamhällets principer", "ekonomisk tillväxt", "Det är allt."]
        split = split_speech_from_record(paragraphs, "html", headings)
        assert split.paragraphs == tuple(paragraphs)

    def test_a_short_heading_is_not_matched_in_pieces(self):
        headings = index_headings({"Barn och unga"})
        paragraphs = ["Två grupper är viktiga:", "barn och", "unga", "Tack."]
        split = split_speech_from_record(paragraphs, "html", headings)
        assert split.paragraphs == tuple(paragraphs)

    def test_the_title_run_together_with_its_subtitle_is_dropped(self):
        titles = ("Granskningsbetänkande", "Regeringens ansvar för förvaltningen (kapitel 4)")
        split = split_speech_from_record(
            [
                "Gransknings-betänkandeRegeringens ansvarför förvaltningen (kapitel 4)",
                "Herr talman! Då var vi inne i kap. 4.",
            ],
            "html",
            titles=titles,
        )
        assert split.paragraphs == ("Herr talman! Då var vi inne i kap. 4.",)
        assert rules(split) == ["heading_with_subtitle"]
        assert split.furniture == 1

    def test_the_title_and_subtitle_of_another_debate_are_dropped_as_well(self):
        headings = index_headings(
            (), [("Granskningsbetänkande", "Vissa frågor om statsråds tjänsteutövning (kapitel 3)")]
        )
        split = split_speech_from_record(
            [
                "Det var en brist i beredningen.",
                "Gransknings-betänkandeVissa frågor om statsråds tjänste-utövning (kapitel 3)",
                "Jag yrkar bifall.",
            ],
            "html",
            headings,
            ("Granskningsbetänkande", "Handläggning av vissa regeringsärenden (kapitel 2)"),
        )
        assert split.paragraphs == ("Det var en brist i beredningen.", "Jag yrkar bifall.")
        assert rules(split) == ["heading_with_subtitle"]

    def test_a_list_item_that_spells_a_title_without_a_subtitle_stays_speech(self):
        headings = index_headings({"Stöd till Ukraina"})
        paragraphs = ["Regeringen har tre prioriteringar:", "Stöd till Ukraina.", "Ett försvar."]
        split = split_speech_from_record(paragraphs, "html", headings, ("Stöd till Ukraina", ""))
        assert split.paragraphs == tuple(paragraphs)

    def test_a_piece_of_the_speechs_own_title_is_dropped(self):
        titles = ("Aktuell debatt om situationen inom barn- och ungdomspsykiatrin", "")
        split = split_speech_from_record(
            ["Det duger inte.", "Aktuell debatt om", "Vi måste göra mer."], "html", titles=titles
        )
        assert split.paragraphs == ("Det duger inte.", "Vi måste göra mer.")
        assert rules(split) == ["title_fragment"]

    def test_a_sentence_that_repeats_a_piece_of_the_title_stays_speech(self):
        titles = ("Aktuell debatt om situationen inom barn- och ungdomspsykiatrin", "")
        split = split_speech_from_record(
            ["Det gäller barn- och ungdomspsykiatrin.", "Tack."], "html", titles=titles
        )
        assert split.paragraphs == ("Det gäller barn- och ungdomspsykiatrin.", "Tack.")

    def test_the_end_of_a_sentence_that_is_a_piece_of_the_title_stays_speech(self):
        paragraphs = [
            "Herr talman! Vi har i utskottet behandlat Riksrevisionens rapport om "
            "Arbetsförmedlingens arbete med Förberedande och orienterande",
            "utbildning.",
        ]
        titles = ("Insatsen Förberedande och orienterande utbildning", "")
        split = split_speech_from_record(paragraphs, "html", titles=titles)
        assert split.paragraphs == tuple(paragraphs)

    def test_a_short_piece_of_the_title_stays_speech(self):
        paragraphs = ["Tre saker är viktiga:", "Skatt", "Jobb, vård och skola."]
        split = split_speech_from_record(paragraphs, "html", titles=("Skatt på bostäder", ""))
        assert split.paragraphs == tuple(paragraphs)

    def test_the_speechs_own_title_with_one_letter_missing_is_dropped(self):
        split = split_speech_from_record(
            [
                "Chefs-JsO är administrativ chef för ämbetet.",
                "Val av chefsjustite-ombudsman",
                "Efter att nuvarande chefs-JO har avböjt omval har KU arbetat med att finna en "
                "efterträdare.",
            ],
            "html",
            titles=("Val av chefsjustitieombudsman", ""),
        )
        assert rules(split) == ["title_misprint"]
        assert split.furniture == 1

    def test_after_the_closing_a_piece_of_the_title_is_the_records_narration(self):
        split = split_speech_from_record(
            [
                "Jag yrkar bifall.",
                "Kammaren biföll konstitutionsutskottets förslag och valde därmed till",
                "justitieombudsman",
                "Stefan Holgersson",
            ],
            "html",
            titles=("Val av justitieombudsman", ""),
        )
        assert rules(split) == ["chamber_narration", "after_closing", "after_closing"]
        assert split.furniture == 0

    def test_a_page_footer_inside_a_sentence_is_dropped_and_the_words_kept(self):
        split = split_speech_from_record(
            [
                "Denna strategi kommer fortsättningsvis att ta sig Vid protokollet",
                "ANDERS NORIN",
                "/Eva-Lena Ekman mänskliga rättigheter. Men vi vill se en regional strategi.",
            ],
            "html",
        )
        assert split.paragraphs == (
            "Denna strategi kommer fortsättningsvis att ta sig",
            "mänskliga rättigheter. Men vi vill se en regional strategi.",
        )
        assert split.notes == ()
        assert rules(split) == ["page_footer"] * 3

    def test_a_page_footer_whose_signature_ends_its_paragraph_is_dropped_whole(self):
        split = split_speech_from_record(
            [
                "Denna strategi kommer fortsättningsvis att ta sig Vid protokollet",
                "ANDERS NORIN",
                "/Eva-Lena Ekman",
                "mänskliga rättigheter. Men vi vill se en regional strategi.",
            ],
            "html",
        )
        assert split.paragraphs == (
            "Denna strategi kommer fortsättningsvis att ta sig",
            "mänskliga rättigheter. Men vi vill se en regional strategi.",
        )
        assert [item.text for item in split.set_aside] == [
            "Vid protokollet",
            "ANDERS NORIN",
            "/Eva-Lena Ekman",
        ]

    @pytest.mark.parametrize(
        "clerk",
        [
            "/Eva-Lena Ekman Mänskliga rättigheter är en grund för strategin.",
            "/Ekman mänskliga rättigheter.",
            "/eva-lena ekman mänskliga rättigheter.",
        ],
    )
    def test_a_clerks_signature_that_could_take_the_speakers_words_raises(self, clerk):
        paragraphs = ["Denna strategi kommer att ta sig Vid protokollet", "ANDERS NORIN", clerk]
        with pytest.raises(ValueError, match="clerk's signature"):
            split_speech_from_record(paragraphs, "html")


class TestHelpers:
    def test_normalise_collapses_every_kind_of_whitespace(self):
        assert normalise(" Herr talman!\r\n\tTack. ") == "Herr talman! Tack."

    def test_normalise_drops_the_soft_hyphen(self):
        assert normalise("ett sms\u00admeddelande till allmänheten") == (
            "ett smsmeddelande till allmänheten"
        )

    def test_the_margin_heading_is_read_out_of_the_residue(self):
        residue = "STYLEREF Kantrubrik \\* MERGEFORMAT Svar på interpellationer"
        assert margin_heading(residue) == "Svar på interpellationer"

    def test_ordinary_text_carries_no_margin_heading(self):
        assert margin_heading("Svar på interpellationer") is None
