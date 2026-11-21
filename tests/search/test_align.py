"""Tests for search/corpus/align.py and the report of search/pipeline/03_reconcile_legacy.py."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from search.config import ReconcileSettings, load_settings
from search.corpus.align import (
    ACROSS_SLUGS,
    BY_CONTENT,
    IN_A_PAIRED_SPEECH,
    IN_ANOTHER_PARTY,
    IN_ORDER,
    NO_CONTENT_MATCH,
    PARTY_ABSENT,
    TOO_SHORT,
    UNDER_ANOTHER_NAME,
    AlignParams,
    Speech,
    Unmatched,
    align_party,
    align_speaker,
    edges_agree,
    is_same_speech,
    legacy_key,
    legacy_slug,
    names_agree,
    new_speaker_slug,
    overlap,
    prefix_offset,
    reconcile_protocol,
    score_pair,
    session_of,
    speaker_slug,
    split_legacy_text,
    split_place,
    strip_party,
    strip_reply_marker,
    tokenise,
)

PARAMS = AlignParams(
    edge_tokens=8,
    min_sequence_share=0.6,
    min_shared_tokens=5,
    missing_tokens=5,
    surplus_tokens=20,
)
SETTINGS = ReconcileSettings(
    coverage_threshold=0.99,
    edge_tokens=8,
    min_sequence_share=0.6,
    min_shared_tokens=5,
    missing_tokens=5,
    surplus_tokens=20,
    listed_ids=60,
    listed_deviations=50,
    listed_missing=25,
    listed_tokens=30,
)
#: The one session in scope in the synthetic corpora of :class:`TestReconcile`.
SESSION = "2003/04"

SCHOOL = (
    "Fru talman! Skolan behöver fler lärare och mindre administration. Jag vill att "
    "regeringen satsar på läromedel, på speciallärare och på elevhälsan i hela landet."
)
POLICE = (
    "Herr talman! Polisen dras inte tillbaka från Bergslagen. Det kommer fler poliser "
    "till orterna, och närpolisen ska synas i vardagen där människor bor och arbetar."
)
RAILWAY = (
    "Fru talman! Järnvägen i norra Sverige har eftersatt underhåll. Tågen blir sena, "
    "godset fastnar och pendlarna tappar tålamodet med Trafikverkets planering."
)
HOUSING = (
    "Herr talman! Bostadsbristen slår hårdast mot unga. Hyrorna i nyproduktionen är för "
    "höga, och kommunerna planerar för lite mark för hyresrätter i regionerna."
)
TAIL = (
    "Överläggningen var härmed avslutad. § 9 Svar på interpellation 2024/25:728 om "
    "rättspsykiatrin Innehållsförteckning 1 § 8 Svar på interpellation 2"
)


def new(key: str, text: str, slug: str = "anna-andersson", notes: str = "") -> Speech:
    """Build a speech as the new corpus gives it."""
    return Speech(key, slug, tokenise(text), tokenise(notes))


def old(key: str, text: str, slug: str = "anna-andersson") -> Speech:
    """Build a speech as the older corpus gives it."""
    return Speech(key, slug, tokenise(text))


def pairs(matches) -> list[tuple[str, str]]:
    """Reduce matches to their keys, for comparison."""
    return [(match.old_key, match.new_key) for match in matches]


class TestSlug:
    def test_title_and_name_fold_into_one_slug(self):
        assert speaker_slug("Statsminister ULF KRISTERSSON") == "statsminister-ulf-kristersson"

    def test_swedish_letters_fold_to_their_latin_base(self):
        assert speaker_slug("Ålderspresident ANDERS BJÖRCK") == "alderspresident-anders-bjorck"
        assert speaker_slug("Statsrådet CAMILLA WALTERSSON GRÖNVALL") == (
            "statsradet-camilla-waltersson-gronvall"
        )
        assert speaker_slug("José Müller") == "jose-muller"

    def test_hyphen_and_space_give_the_same_slug(self):
        assert speaker_slug("JAN-EMANUEL JOHANSSON") == speaker_slug("JAN EMANUEL JOHANSSON")

    def test_runs_of_other_characters_become_one_hyphen(self):
        assert speaker_slug("  ANNA  ANDERSSON, i Göteborg. ") == "anna-andersson-i-goteborg"

    def test_the_party_is_stripped_from_the_new_corpus_speaker(self):
        assert strip_party("Statsrådet STEFAN ATTEFALL (KD)") == "Statsrådet STEFAN ATTEFALL"
        assert strip_party("Anna Andersson") == "Anna Andersson"
        assert new_speaker_slug("Anna Andersson (S)") == "anna-andersson"
        assert new_speaker_slug("Statsrådet STEFAN ATTEFALL (KD)") == ("statsradet-stefan-attefall")

    @pytest.mark.parametrize(
        "speaker_raw",
        [
            "BENGT-ANDERS JOHANSSON (m) replik",
            "BENGT-ANDERS JOHANSSON (M)replik",
            "BENGT-ANDERS JOHANSSON (M) (replik)",
            "BENGT-ANDERS JOHANSSON (M)",
        ],
    )
    def test_a_reply_marker_after_the_party_is_stripped_with_it(self, speaker_raw):
        assert strip_party(speaker_raw) == "BENGT-ANDERS JOHANSSON"
        assert new_speaker_slug(speaker_raw) == "bengt-anders-johansson"

    def test_a_name_without_a_marker_keeps_its_last_word(self):
        assert strip_reply_marker("Anna Andersson (S)") == "Anna Andersson (S)"

    def test_the_slug_is_read_out_of_an_older_id(self):
        assert legacy_slug("GQ091_bo-lundgren_m", "GQ091", "M") == "bo-lundgren"
        assert legacy_slug("HC09139_eva-lindh_s", "HC09139", "S") == "eva-lindh"

    def test_a_malformed_older_id_raises(self):
        with pytest.raises(ValueError, match="does not read as"):
            legacy_slug("HC09139_eva-lindh_s", "HC09139", "M")


#: An older slug and the new slug of the same speaker, as the two corpora
#: write them: one pair for each way the forms differ.
ONE_NAME_IN_TWO_FORMS = {
    "a title in two forms": (
        "statsminister-ulf-kristersson",
        "statsministern-ulf-kristersson",
    ),
    "a title on the new side only": (
        "ebba-busch",
        "energi-och-naringsministern-ebba-busch",
    ),
    "a title on the older side only": (
        "alderspresident-krister-ornfjader",
        "krister-ornfjader",
    ),
    "a place after the name": (
        "ulrika-carlsson-i-skovde",
        "ulrika-carlsson",
    ),
    "a hyphen in the surname": (
        "helena-storckenf-eldt",
        "helena-storckenfeldt",
    ),
    "a hyphen in the surname under a title": (
        "statsminister-ulf-kriste-rsson",
        "statsministern-ulf-kristersson",
    ),
    "a hyphen in the given name": (
        "jamstalldhetsminister-nin-a-larsson",
        "jamstalldhetsministern-nina-larsson",
    ),
    "a misprinted surname": (
        "arbetsmarknadsminister-elisabeth-svantessson",
        "arbetsmarknadsminister-elisabeth-svantesson",
    ),
    "two given names written as one": (
        "minister-for-civilt-forsvar-carloskar-bohlin",
        "minister-for-civilt-forsvar-carl-oskar-bohlin",
    ),
    "stray words inside the name": (
        "justitie-och-migrationsminister-morgan-svar-pa-johansson",
        "justitie-och-migrationsministern-morgan-johansson",
    ),
}


class TestNamesAgree:
    @pytest.mark.parametrize(
        ("old_slug", "new_slug"),
        ONE_NAME_IN_TWO_FORMS.values(),
        ids=ONE_NAME_IN_TWO_FORMS.keys(),
    )
    def test_one_name_in_two_forms_agrees(self, old_slug, new_slug):
        assert names_agree(old_slug, new_slug)

    def test_identical_slugs_agree(self):
        assert names_agree("anna-andersson", "anna-andersson")

    @pytest.mark.parametrize(
        ("old_slug", "new_slug"),
        [
            ("anna-andersson", "bo-berg"),
            ("statsradet-anna-berg", "statsradet-bo-berg"),
            ("anna-andersson", "anna-berg"),
            ("per-ekstrom", "per-ek"),
            ("lars-johansson", "lars-jonsson"),
            ("anna-karlsson", "anna-carlsson"),
            ("anna-andersson", ""),
        ],
    )
    def test_two_names_do_not_agree(self, old_slug, new_slug):
        assert not names_agree(old_slug, new_slug)

    def test_a_shared_surname_does_not_agree_without_the_given_name(self):
        assert not names_agree("anna-berg", "bo-berg")
        assert names_agree("statsradet-bo-berg", "bo-berg")

    def test_a_name_agrees_with_the_older_slug_of_each_member_who_carries_it(self):
        assert names_agree("anna-berg-i-lund", "anna-berg")
        assert names_agree("anna-berg-i-umea", "anna-berg")

    @pytest.mark.parametrize(
        ("old_slug", "new_slug"),
        [
            ("sven-svensson-i-malmo", "eva-nilsson-i-malmo"),
            ("sven-svensson-i-hisings-backa", "eva-nilsson-i-hisings-backa"),
        ],
    )
    def test_two_members_of_one_place_do_not_agree(self, old_slug, new_slug):
        assert not names_agree(old_slug, new_slug)

    @pytest.mark.parametrize(
        ("old_slug", "new_slug"),
        [
            ("anna-ek-i-boden", "anna-ek-i-umea"),
            ("anna-ek-i-hisings-backa", "anna-ek-i-backa"),
        ],
    )
    def test_one_name_in_two_places_does_not_agree(self, old_slug, new_slug):
        assert not names_agree(old_slug, new_slug)

    @pytest.mark.parametrize(
        ("old_slug", "new_slug"),
        [("statsradet-anna-ek", "statsradet"), ("anna-ek", "ek")],
    )
    def test_a_new_slug_of_one_token_does_not_agree(self, old_slug, new_slug):
        assert not names_agree(old_slug, new_slug)

    @pytest.mark.parametrize(
        ("slug", "with_place"),
        [
            ("ulrika-carlsson", "ulrika-carlsson-i-skovde"),
            ("sven-svensson", "sven-svensson-i-hisings-backa"),
        ],
    )
    def test_a_place_on_one_side_agrees_in_both_orders(self, slug, with_place):
        assert names_agree(slug, with_place)
        assert names_agree(with_place, slug)

    def test_a_title_on_one_side_and_a_place_on_the_other_do_not_agree(self):
        assert not names_agree("statsminister-goran-persson", "goran-persson-i-simrishamn")
        assert not names_agree("goran-persson-i-simrishamn", "statsminister-goran-persson")

    def test_three_tokens_after_the_marker_are_words_and_no_place(self):
        old_slug = "tredje-vice-talmannen-jag-vill-i-debatten-tasso-stafilidis"

        assert split_place(old_slug) == (old_slug.split("-"), ())
        assert names_agree(old_slug, "tasso-stafilidis")

    def test_a_place_of_two_tokens_is_kept_whole(self):
        assert split_place("sven-svensson-i-hisings-backa") == (
            ["sven", "svensson"],
            ("hisings", "backa"),
        )

    def test_a_split_surname_under_a_title_on_one_side_agrees_by_the_run_of_tokens(self):
        assert names_agree("anna-storckenf-eldt", "statsradet-anna-storckenfeldt")

    def test_one_letter_more_agrees_just_above_the_likeness_bound(self):
        assert names_agree("jan-ericsson", "jan-ericson")

    def test_one_letter_more_does_not_agree_just_under_the_likeness_bound(self):
        assert not names_agree("bo-eriksson", "bo-erikson")

    def test_the_answer_can_depend_on_which_slug_is_the_older(self):
        assert names_agree("anna-k-ek", "anna-ek")
        assert not names_agree("anna-ek", "anna-k-ek")


class TestTokenise:
    def test_lower_case_letters_and_digits_only(self):
        assert tokenise("Fru talman! Det är 2,5 %, inte mer.") == (
            "fru",
            "talman",
            "det",
            "är",
            "2",
            "5",
            "inte",
            "mer",
        )

    def test_a_hyphen_inside_a_word_is_dropped(self):
        assert tokenise("EU-länder") == tokenise("EUländer") == ("euländer",)

    def test_a_soft_hyphen_is_dropped(self):
        assert tokenise("barnfamil\u00adjer") == ("barnfamiljer",)

    def test_a_range_is_one_token_whichever_dash_sets_it(self):
        assert tokenise("2002-2006") == tokenise("2002\u20132006") == ("20022006",)

    def test_a_free_standing_hyphen_leaves_nothing(self):
        assert tokenise("Herr talman - jag") == ("herr", "talman", "jag")

    def test_whitespace_and_line_breaks_do_not_count(self):
        assert tokenise("Fru\n talman!\r\n  Tack.") == tokenise("Fru talman! Tack.")


class TestScore:
    def test_overlap_counts_a_token_at_most_as_often_as_in_both(self):
        assert overlap({"a": 3, "b": 1}, {"a": 1, "b": 2, "c": 5}) == 2

    def test_identical_texts(self):
        score = score_pair(new("n", SCHOOL), old("o", SCHOOL), PARAMS)
        assert score.identical
        assert score.new_in_old == score.old_in_new == score.sequence_share == 1.0
        assert score.head_agrees and score.tail_agrees
        assert score.surplus == score.missing == 0

    def test_the_blind_hyphen_removal_is_no_difference(self):
        score = score_pair(
            new("n", "Alla EU-länder ska delta."), old("o", "Alla EUländer ska delta."), PARAMS
        )
        assert score.identical

    def test_a_swallowed_tail_is_surplus_and_the_notes_explain_part_of_it(self):
        speech = new("n", POLICE, notes="Överläggningen var härmed avslutad.")
        score = score_pair(speech, old("o", f"{POLICE} {TAIL}"), PARAMS)

        assert score.new_in_old == 1.0
        assert score.old_extends_new
        assert score.head_agrees and not score.tail_agrees
        assert score.surplus == len(tokenise(TAIL))
        assert score.surplus_with_notes == len(tokenise(TAIL)) - 4
        assert score.old_in_new < score.old_in_new_with_notes < 1.0
        assert is_same_speech(score, PARAMS)

    def test_a_new_speech_shorter_than_the_edge_agrees_at_the_head_of_a_longer_one(self):
        score = score_pair(
            new("n", "Herr talman jag yrkar bifall"),
            old("o", "Herr talman jag yrkar bifall till reservationen nummer ett i betänkandet"),
            PARAMS,
        )
        assert score.old_extends_new
        assert score.head_agrees and not score.tail_agrees

    def test_a_new_speech_shorter_than_the_edge_agrees_at_the_tail_it_ends(self):
        assert edges_agree(("b", "c"), ("x", "y", "a", "b", "c"), 8) == (False, True)

    def test_an_older_speech_shorter_than_the_edge_disagrees_at_both_ends(self):
        assert edges_agree(tuple("abcdefghij"), tuple("abc"), 8) == (False, False)

    def test_an_empty_new_speech_agrees_at_neither_end(self):
        assert edges_agree((), ("a", "b"), 8) == (False, False)

    def test_two_speeches_by_one_speaker_are_not_the_same(self):
        assert not is_same_speech(score_pair(new("n", SCHOOL), old("o", RAILWAY), PARAMS), PARAMS)

    def test_a_shared_greeting_alone_is_not_enough(self):
        score = score_pair(new("n", "Fru talman! Tack."), old("o", "Fru talman! Tack."), PARAMS)
        assert score.identical
        assert not is_same_speech(score, PARAMS)


class TestAlignSpeaker:
    def test_equal_counts_pair_in_order(self):
        olds = [old("o0", SCHOOL), old("o1", RAILWAY)]
        news = [new("n0", SCHOOL), new("n1", RAILWAY)]
        matches, rest_old, rest_new = align_speaker(olds, news, PARAMS)

        assert pairs(matches) == [("o0", "n0"), ("o1", "n1")]
        assert {match.method for match in matches} == {IN_ORDER}
        assert rest_old == rest_new == []

    def test_equal_counts_out_of_step_fall_back_to_content(self):
        olds = [old("o0", SCHOOL), old("o1", RAILWAY)]
        news = [new("n0", RAILWAY), new("n1", HOUSING)]
        matches, rest_old, rest_new = align_speaker(olds, news, PARAMS)

        assert pairs(matches) == [("o1", "n0")]
        assert {match.method for match in matches} == {BY_CONTENT}
        assert (rest_old, rest_new) == ([0], [1])

    def test_a_speech_missing_from_the_older_side(self):
        olds = [old("o0", SCHOOL), old("o1", HOUSING)]
        news = [new("n0", SCHOOL), new("n1", RAILWAY), new("n2", HOUSING)]
        matches, rest_old, rest_new = align_speaker(olds, news, PARAMS)

        assert pairs(matches) == [("o0", "n0"), ("o1", "n2")]
        assert (rest_old, rest_new) == ([], [1])

    def test_a_speech_missing_from_the_new_side(self):
        olds = [old("o0", SCHOOL), old("o1", RAILWAY), old("o2", HOUSING)]
        news = [new("n0", SCHOOL), new("n1", HOUSING)]
        matches, rest_old, rest_new = align_speaker(olds, news, PARAMS)

        assert pairs(matches) == [("o0", "n0"), ("o2", "n1")]
        assert (rest_old, rest_new) == ([1], [])

    def test_the_order_is_kept_when_a_text_repeats(self):
        olds = [old("o0", SCHOOL), old("o1", SCHOOL)]
        news = [new("n0", SCHOOL), new("n1", RAILWAY), new("n2", SCHOOL)]
        matches, _, rest_new = align_speaker(olds, news, PARAMS)

        assert pairs(matches) == [("o0", "n0"), ("o1", "n2")]
        assert rest_new == [1]


class TestAlignParty:
    def test_a_speaker_spelled_differently_is_paired_by_content(self):
        minister = "justitieminister-gunnar-strommer"
        olds = [old("o0", POLICE, minister), old("o1", RAILWAY, minister)]
        news = [
            new("n0", POLICE, "justitieministern-gunnar-strommer"),
            new("n1", SCHOOL, "anna-andersson"),
            new("n2", RAILWAY, "justitieministern-gunnar-strommer"),
        ]
        alignment = align_party(olds, news, PARAMS)

        assert pairs(alignment.matches) == [("o0", "n0"), ("o1", "n2")]
        assert {match.method for match in alignment.matches} == {ACROSS_SLUGS}
        assert alignment.unmatched_old == ()
        assert alignment.unmatched_new == ("n1",)

    def test_speakers_with_matching_slugs_are_paired_first(self):
        olds = [old("o0", SCHOOL, "anna-andersson"), old("o1", HOUSING, "bo-berg")]
        news = [new("n0", HOUSING, "bo-berg"), new("n1", SCHOOL, "anna-andersson")]
        alignment = align_party(olds, news, PARAMS)

        assert pairs(alignment.matches) == [("o0", "n1"), ("o1", "n0")]
        assert {match.method for match in alignment.matches} == {IN_ORDER}

    def test_two_members_identical_formulaic_speeches_stay_unpaired(self):
        formula = "Herr talman! Jag yrkar bifall till reservation 3 i betänkandet."
        olds = [old("o0", formula, "anna-andersson")]
        news = [new("n0", formula, "bo-berg")]
        alignment = align_party(olds, news, PARAMS)

        assert alignment.matches == ()
        assert alignment.unmatched_old == ("o0",)
        assert alignment.unmatched_new == ("n0",)

    def test_two_members_who_share_a_surname_stay_unpaired(self):
        olds = [old("o0", POLICE, "anna-berg")]
        news = [new("n0", POLICE, "bo-berg")]
        alignment = align_party(olds, news, PARAMS)

        assert alignment.matches == ()
        assert alignment.unmatched_old == ("o0",)
        assert alignment.unmatched_new == ("n0",)

    @pytest.mark.parametrize(
        ("old_slug", "new_slug"),
        ONE_NAME_IN_TWO_FORMS.values(),
        ids=ONE_NAME_IN_TWO_FORMS.keys(),
    )
    def test_one_name_in_two_forms_pairs_across_the_slugs(self, old_slug, new_slug):
        olds = [old("o0", POLICE, old_slug), old("o1", RAILWAY, old_slug)]
        news = [new("n0", POLICE, new_slug), new("n1", RAILWAY, new_slug)]
        alignment = align_party(olds, news, PARAMS)

        assert pairs(alignment.matches) == [("o0", "n0"), ("o1", "n1")]
        assert {match.method for match in alignment.matches} == {ACROSS_SLUGS}

    def test_a_speech_is_paired_with_the_name_that_agrees_among_several_candidates(self):
        olds = [old("o0", POLICE, "statsradet-bo-berg")]
        news = [new("n0", POLICE, "anna-berg"), new("n1", POLICE, "bo-berg")]
        alignment = align_party(olds, news, PARAMS)

        assert pairs(alignment.matches) == [("o0", "n1")]
        assert alignment.unmatched_new == ("n0",)

    def test_a_slug_without_a_candidate_does_not_end_the_second_step(self):
        olds = [old("o0", SCHOOL, "anna-andersson"), old("o1", HOUSING, "bo-berg")]
        news = [new("n0", SCHOOL, "cia-cedergren"), new("n1", HOUSING, "statsradet-bo-berg")]
        alignment = align_party(olds, news, PARAMS)

        assert pairs(alignment.matches) == [("o1", "n1")]
        assert alignment.unmatched_old == ("o0",)
        assert alignment.unmatched_new == ("n0",)

    def test_a_swallowed_tail_still_pairs_in_order(self):
        olds = [old("o0", SCHOOL), old("o1", f"{HOUSING} {TAIL}")]
        news = [new("n0", SCHOOL), new("n1", HOUSING, notes="Överläggningen var härmed avslutad.")]
        alignment = align_party(olds, news, PARAMS)

        assert pairs(alignment.matches) == [("o0", "n0"), ("o1", "n1")]
        assert alignment.matches[1].score.old_extends_new


class TestReconcileProtocol:
    def test_a_party_missing_on_one_side_is_named(self):
        result = reconcile_protocol(
            {"S": [old("o0", SCHOOL)], "M": [old("o1", POLICE, "bo-berg")]},
            {"S": [new("n0", SCHOOL)], "C": [new("n1", RAILWAY, "cia-cedergren")]},
            PARAMS,
        )

        assert pairs(result.matches) == [("o0", "n0")]
        assert [(u.key, u.party, u.reason, u.found_in) for u in result.unmatched_old] == [
            ("o1", "M", PARTY_ABSENT, None)
        ]
        assert [(u.key, u.party, u.reason, u.found_in) for u in result.unmatched_new] == [
            ("n1", "C", PARTY_ABSENT, None)
        ]

    def test_a_speech_under_another_party_is_diagnosed_and_not_paired(self):
        result = reconcile_protocol(
            {"S": [old("o0", SCHOOL)], "V": [old("o1", RAILWAY)]},
            {"S": [new("n0", SCHOOL), new("n1", RAILWAY)]},
            PARAMS,
        )

        assert pairs(result.matches) == [("o0", "n0")]
        assert [(u.key, u.reason, u.found_in) for u in result.unmatched_old] == [
            ("o1", PARTY_ABSENT, IN_ANOTHER_PARTY)
        ]
        assert [(u.key, u.reason, u.found_in) for u in result.unmatched_new] == [
            ("n1", NO_CONTENT_MATCH, IN_ANOTHER_PARTY)
        ]

    def test_a_speech_under_another_name_is_diagnosed_and_not_paired(self):
        result = reconcile_protocol(
            {"S": [old("o0", SCHOOL), old("o1", POLICE, "anna-berg")]},
            {"S": [new("n0", SCHOOL), new("n1", POLICE, "bo-berg")]},
            PARAMS,
        )

        assert pairs(result.matches) == [("o0", "n0")]
        assert [(u.key, u.reason, u.found_in) for u in result.unmatched_old] == [
            ("o1", NO_CONTENT_MATCH, UNDER_ANOTHER_NAME)
        ]
        assert [(u.key, u.reason, u.found_in) for u in result.unmatched_new] == [
            ("n1", NO_CONTENT_MATCH, UNDER_ANOTHER_NAME)
        ]

    def test_another_name_with_other_words_is_no_diagnosis(self):
        result = reconcile_protocol(
            {"S": [old("o0", POLICE, "anna-berg")]},
            {"S": [new("n0", RAILWAY, "bo-berg")]},
            PARAMS,
        )

        assert result.matches == ()
        assert [u.found_in for u in result.unmatched_old + result.unmatched_new] == [None, None]

    def test_the_same_words_under_an_agreeing_slug_are_not_under_another_name(self):
        result = reconcile_protocol(
            {"S": [old("o0", SCHOOL, "anna-k-ek"), old("o1", POLICE, "anna-k-ek")]},
            {"S": [new("n0", POLICE, "anna-ek"), new("n1", SCHOOL, "anna-ek")]},
            PARAMS,
        )

        assert len(result.matches) == 1
        unmatched = result.unmatched_old + result.unmatched_new
        assert [(u.reason, u.found_in) for u in unmatched] == [(NO_CONTENT_MATCH, None)] * 2

    def test_another_name_is_read_with_the_older_slug_as_the_older(self):
        result = reconcile_protocol(
            {"S": [old("o0", POLICE, "anna-ek")]},
            {"S": [new("n0", POLICE, "anna-k-ek")]},
            PARAMS,
        )

        assert result.matches == ()
        assert [u.found_in for u in result.unmatched_old + result.unmatched_new] == [
            UNDER_ANOTHER_NAME,
            UNDER_ANOTHER_NAME,
        ]

    def test_a_speech_read_twice_is_diagnosed(self):
        result = reconcile_protocol(
            {"S": [old("o0", SCHOOL), old("o1", SCHOOL)]},
            {"S": [new("n0", SCHOOL)]},
            PARAMS,
        )

        assert len(result.matches) == 1
        assert [(u.reason, u.found_in) for u in result.unmatched_old] == [
            (NO_CONTENT_MATCH, IN_A_PAIRED_SPEECH)
        ]

    def test_a_fragment_too_short_to_pair_is_named(self):
        result = reconcile_protocol(
            {"C": [old("o0", SCHOOL), old("o1", "........ 11")]},
            {"C": [new("n0", SCHOOL)]},
            PARAMS,
        )

        assert pairs(result.matches) == [("o0", "n0")]
        assert [(u.key, u.reason) for u in result.unmatched_old] == [("o1", TOO_SHORT)]


class TestLegacyRecords:
    def test_a_record_splits_on_blank_lines(self):
        assert split_legacy_text(f"{SCHOOL}\n\n{RAILWAY}\n\n \n\n") == [SCHOOL, RAILWAY]

    def test_a_speech_is_keyed_by_its_record_and_position(self):
        assert legacy_key("GQ091_bo-lundgren_m", 2) == "GQ091_bo-lundgren_m#2"


class TestSessions:
    @pytest.mark.parametrize(
        ("dok_id", "session"),
        [
            ("GP0921", "2001/02"),
            ("GQ091", "2002/03"),
            ("GZ0985", "2011/12"),
            ("H109116", "2013/14"),
        ],
    )
    def test_the_prefix_names_the_session(self, dok_id, session):
        offset = prefix_offset("GR091", "2003/04")
        assert session_of(dok_id, offset) == session

    def test_the_offset_is_the_same_across_sessions(self):
        assert prefix_offset("GR091", "2003/04") == prefix_offset("HC09139", "2024/25")

    def test_the_century_turns_cleanly(self):
        assert session_of("JF091", prefix_offset("GR091", "2003/04")) == "2099/00"


def load_reconcile_script():
    """Import search/pipeline/03_reconcile_legacy.py, whose name is not a module name.

    The module is registered before it runs, because Pydantic resolves the
    report models' postponed annotations through ``sys.modules``.
    """
    path = Path(__file__).resolve().parents[2] / "search" / "pipeline" / "03_reconcile_legacy.py"
    spec = importlib.util.spec_from_file_location("reconcile_legacy", path)
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestReportFigures:
    """The helpers that turn the tally into the report's figures."""

    script = load_reconcile_script()

    def test_the_alignment_parameters_have_no_defaults_of_their_own(self):
        with pytest.raises(ValidationError):
            AlignParams()

    def test_the_alignment_parameters_are_taken_from_the_settings(self):
        assert self.script.align_params(SETTINGS) == PARAMS

    def test_a_changed_setting_reaches_the_alignment_parameters(self):
        settings = SETTINGS.model_copy(update={"min_sequence_share": 0.8, "surplus_tokens": 3})
        params = self.script.align_params(settings)
        assert (params.min_sequence_share, params.surplus_tokens) == (0.8, 3)
        assert params.edge_tokens == PARAMS.edge_tokens

    def test_a_share_keeps_its_counts_and_rounds(self):
        figure = self.script.share(2, 3)
        assert (figure.numerator, figure.denominator, figure.share) == (2, 3, 0.6667)

    def test_a_share_of_nothing_is_zero(self):
        assert self.script.share(0, 0).share == 0.0

    @pytest.mark.parametrize("denominator", [100, 300, 287_600, 1_000_000])
    def test_exactly_the_threshold_meets_it(self, denominator):
        numerator = denominator * 99 // 100
        assert self.script.meets_threshold(numerator, denominator, 0.99)
        assert not self.script.meets_threshold(numerator - 1, denominator, 0.99)

    def test_an_empty_denominator_meets_no_threshold(self):
        assert not self.script.meets_threshold(0, 0, 0.99)

    def test_the_spread_reads_each_percentile_at_its_index_in_the_sorted_shares(self):
        values = [i / 100 for i in range(101)]
        figure = self.script.spread(list(reversed(values)))
        assert (figure.mean, figure.median, figure.p05, figure.p01, figure.minimum) == (
            0.5,
            0.5,
            0.05,
            0.01,
            0.0,
        )

    def test_the_spread_of_no_pairs_is_zero(self):
        assert self.script.spread([]).median == 0.0

    def test_one_sided_protocols_are_counted_by_year_and_the_list_is_capped(self):
        cap = 7
        ids = [f"GR{number:04d}" for number in range(cap + 5)]
        year_of = {protocol: 2003 + (number % 2) for number, protocol in enumerate(ids)}
        figure = self.script.one_sided(reversed(ids), year_of, dict.fromkeys(ids, 2), cap)

        assert figure.protocols == cap + 5
        assert figure.speeches == 2 * (cap + 5)
        assert sum(figure.by_year.values()) == cap + 5
        assert list(figure.by_year) == ["2003", "2004"]
        assert figure.ids == tuple(ids[:cap])
        assert figure.ids_not_listed == 5


def new_row(
    anf_id: str, dok_id: str, day: date, speaker_raw: str, party: str, text: str, rm: str = SESSION
) -> dict:
    """A row of the new corpus, with the columns the script reads."""
    return {
        "anf_id": anf_id,
        "dok_id": dok_id,
        "rm": rm,
        "date": day,
        "speaker_raw": speaker_raw,
        "party": party,
        "paragraphs": [text],
        "notes": [],
    }


def old_row(record_id: str, protocol_id: str, day: date, speaker: str, party: str, text: str):
    """A row of the older corpus, with the columns the script reads."""
    return {
        "id": record_id,
        "protocol_id": protocol_id,
        "protocol_date": datetime(day.year, day.month, day.day),
        "speaker": speaker,
        "party": party,
        "text": text,
    }


class TestReconcile:
    """The whole step on two synthetic corpora written to a temporary directory."""

    script = load_reconcile_script()

    def rows(self) -> tuple[list[dict], list[dict]]:
        """Build the rows of the new corpus and of the older corpus.

        Protocol GR091 is on both sides, GR092 on the new side only and GR093
        on the older side only. The older side also has GQ091, of 2002/03.
        """
        first, second, third = date(2003, 10, 1), date(2003, 10, 2), date(2003, 10, 3)
        new_rows = [
            new_row("GR091-1", "GR091", first, "ANNA ANDERSSON (s)", "S", SCHOOL),
            new_row("GR091-2", "GR091", first, "ANNA ANDERSSON (s) replik", "S", RAILWAY),
            new_row("GR092-1", "GR092", second, "Bo Berg (M)", "M", HOUSING),
        ]
        old_rows = [
            old_row(
                "GR091_anna-andersson_s",
                "GR091",
                first,
                "ANNA ANDERSSON",
                "S",
                f"{SCHOOL}\n\n{RAILWAY}",
            ),
            old_row("GR093_bo-berg_m", "GR093", third, "BO BERG", "M", POLICE),
            old_row("GQ091_bo-berg_m", "GQ091", date(2002, 10, 1), "BO BERG", "M", POLICE),
        ]
        return new_rows, old_rows

    def run(
        self,
        tmp_path: Path,
        settings: ReconcileSettings = SETTINGS,
        first_session: str = SESSION,
        last_session: str = SESSION,
    ):
        """Reconcile the corpora of :meth:`rows`."""
        new_rows, old_rows = self.rows()
        return self.reconcile_rows(
            tmp_path, new_rows, old_rows, settings, first_session, last_session
        )

    def write_corpora(
        self, new_path: Path, old_path: Path, new_rows: list[dict], old_rows: list[dict]
    ) -> None:
        """Write the two corpora to parquet files."""
        new_schema = pa.schema(
            [
                ("anf_id", pa.string()),
                ("dok_id", pa.string()),
                ("rm", pa.string()),
                ("date", pa.date32()),
                ("speaker_raw", pa.string()),
                ("party", pa.string()),
                ("paragraphs", pa.list_(pa.string())),
                ("notes", pa.list_(pa.string())),
            ]
        )
        pq.write_table(pa.Table.from_pylist(new_rows, schema=new_schema), new_path)
        pq.write_table(pa.Table.from_pylist(old_rows), old_path)

    def reconcile_rows(
        self,
        tmp_path: Path,
        new_rows: list[dict],
        old_rows: list[dict],
        settings: ReconcileSettings,
        first_session: str = SESSION,
        last_session: str = SESSION,
    ):
        """Write the two corpora to parquet files and reconcile them."""
        new_path, old_path = tmp_path / "new.parquet", tmp_path / "old.parquet"
        self.write_corpora(new_path, old_path, new_rows, old_rows)
        return self.script.reconcile(new_path, old_path, settings, first_session, last_session)

    @pytest.fixture
    def reports_written(self, tmp_path, monkeypatch) -> list[Path]:
        """Set the script's main up to run on the corpora of :meth:`rows`.

        The corpora are written where the shipped settings look for them under
        a data root, the command line names that root, and no report reaches
        the disk. Returns the paths the script asks to write a report to.
        """
        settings = load_settings(data_root=tmp_path)
        new_path = settings.data.search / self.script.SPEECHES_NAME
        old_path = settings.data.legacy_corpus
        for path in (new_path, old_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.write_corpora(new_path, old_path, *self.rows())
        written: list[Path] = []
        monkeypatch.setattr(self.script, "write_report", lambda report, path: written.append(path))
        monkeypatch.setattr(sys, "argv", ["03_reconcile_legacy.py", "--data-root", str(tmp_path)])
        return written

    def test_a_session_before_the_first_in_scope_is_out_of_scope(self, tmp_path):
        report = self.run(tmp_path)

        assert report.sessions_in_scope == ("2003/04", "2003/04")
        assert report.new_corpus.sessions == ("2003/04", "2003/04")
        assert report.old_corpus.speeches == 4
        assert [(item.session, item.speeches) for item in report.old_corpus.out_of_scope] == [
            ("2002/03", 1)
        ]
        assert report.old_corpus.in_scope_speeches == 3
        assert report.old_corpus.slug_disagreements == 0

    def test_the_scope_is_the_same_when_the_new_corpus_holds_a_session_outside_it(self, tmp_path):
        new_rows, old_rows = self.rows()
        outside = new_row(
            "GQ091-1", "GQ091", date(2002, 10, 1), "Bo Berg (M)", "M", POLICE, rm="2002/03"
        )
        report = self.reconcile_rows(tmp_path, [*new_rows, outside], old_rows, SETTINGS)

        assert report.new_corpus.sessions == ("2002/03", "2003/04")
        assert [(item.session, item.speeches) for item in report.old_corpus.out_of_scope] == [
            ("2002/03", 1)
        ]
        assert (report.coverage.numerator, report.coverage.denominator) == (2, 3)

    def test_the_report_names_the_first_and_the_last_session_in_scope_in_that_order(self, tmp_path):
        new_rows, old_rows = self.rows()
        earlier = new_row(
            "GQ091-1", "GQ091", date(2002, 10, 1), "Bo Berg (M)", "M", POLICE, rm="2002/03"
        )
        report = self.reconcile_rows(
            tmp_path, [*new_rows, earlier], old_rows, SETTINGS, "2002/03", SESSION
        )

        assert report.sessions_in_scope == ("2002/03", "2003/04")

    @pytest.mark.parametrize(
        ("first_session", "last_session", "absent"),
        [
            ("2002/03", "2003/04", "2002/03"),
            ("2003/04", "2005/06", "2004/05, 2005/06"),
        ],
    )
    def test_a_session_in_scope_without_a_speech_in_the_new_corpus_raises(
        self, tmp_path, first_session, last_session, absent
    ):
        with pytest.raises(ValueError, match=f"in scope: {absent}[.]$"):
            self.run(tmp_path, first_session=first_session, last_session=last_session)

    def test_an_older_corpus_without_a_protocol_in_scope_raises(self, tmp_path):
        new_rows, old_rows = self.rows()
        outside = [row for row in old_rows if row["protocol_id"] == "GQ091"]
        with pytest.raises(ValueError, match="no protocol of the sessions 2003/04 to 2003/04"):
            self.reconcile_rows(tmp_path, new_rows, outside, SETTINGS)

    def test_a_first_session_later_than_the_last_raises(self, tmp_path):
        with pytest.raises(ValueError, match="2004/05"):
            self.run(tmp_path, first_session="2004/05", last_session="2003/04")

    def test_the_script_writes_no_report_when_a_session_in_scope_has_no_speech(
        self, reports_written
    ):
        with pytest.raises(ValueError, match="in scope: 2004/05"):
            self.script.main()

        assert reports_written == []

    def test_the_script_takes_the_sessions_in_scope_from_the_settings(
        self, tmp_path, reports_written, monkeypatch
    ):
        settings = load_settings(data_root=tmp_path)
        asked_for: list[tuple[str, str]] = []
        reconcile = self.script.reconcile

        def reconcile_one_session(new_path, old_path, settings, first_session, last_session):
            asked_for.append((first_session, last_session))
            return reconcile(new_path, old_path, settings, SESSION, SESSION)

        monkeypatch.setattr(self.script, "reconcile", reconcile_one_session)
        self.script.main()

        assert asked_for == [(settings.corpus.first_session, settings.source.last_session)]
        assert reports_written == [settings.tracked.reports / self.script.REPORT_NAME]

    def test_every_field_the_report_texts_name_exists(self, tmp_path):
        report = self.run(tmp_path)
        settings = load_settings(data_root=tmp_path)

        named = [
            (report.scope, settings, "corpus.first_session"),
            (report.scope, settings, "source.last_session"),
            (report.scope, report, "sessions_in_scope"),
            (report.scope, report, "old_corpus.out_of_scope"),
            (report.pairing, report.params, "min_shared_tokens"),
            (report.pairing, report.params, "min_sequence_share"),
            (report.pairing, report, "overlap.old_in_new"),
            (report.pairing, report, "surplus"),
        ]
        for text, owner, name in named:
            assert name in text
            for part in name.split("."):
                owner = getattr(owner, part)

    def test_coverage_counts_the_older_in_scope_speeches_and_misses_the_threshold(self, tmp_path):
        report = self.run(tmp_path)

        assert (report.coverage.numerator, report.coverage.denominator) == (2, 3)
        assert not report.coverage_meets_threshold
        assert (report.reverse.numerator, report.reverse.denominator) == (2, 3)

    def test_a_reply_is_paired_in_order_under_its_speaker(self, tmp_path):
        assert self.run(tmp_path).methods == {IN_ORDER: 2}

    def test_protocols_on_one_side_only_are_named(self, tmp_path):
        report = self.run(tmp_path)

        assert report.old_only_protocols.ids == ("GR093",)
        assert report.new_only_protocols.ids == ("GR092",)
        assert [(row.year, row.old_matched, row.new_matched) for row in report.years] == [
            (2003, 2, 2)
        ]

    def test_the_report_carries_the_threshold_and_parameters_it_was_given(self, tmp_path):
        settings = SETTINGS.model_copy(update={"coverage_threshold": 0.5, "missing_tokens": 9})
        report = self.run(tmp_path, settings)

        assert report.coverage_threshold == 0.5
        assert report.coverage_meets_threshold
        assert report.params == self.script.align_params(settings)
        assert report.params.missing_tokens == 9

    def test_the_list_of_unmatched_speeches_is_capped_by_the_setting(self):
        unmatched = [Unmatched(f"GR091_x_s#{n}", "S", TOO_SHORT, None) for n in range(5)]
        tally = self.script._Tally(PARAMS, listed_ids=3)
        tally.unmatched["old"].extend(unmatched)

        figure = tally.left_over("old")

        assert figure.speeches == 5
        assert len(figure.listed) == 3
        assert figure.not_listed == 2

    def test_every_list_in_the_report_is_capped_by_its_own_setting(self, tmp_path):
        day = date(2003, 10, 1)
        speakers = [("Anna Andersson", "S"), ("Bo Berg", "M"), ("Cecilia Carlsson", "C")]
        speakers += [("David Dahl", "V"), ("Eva Ek", "KD")]
        new_rows, old_rows = [], []
        for number, (name, party) in enumerate(speakers):
            slug = name.lower().replace(" ", "-")
            base = SCHOOL
            new_rows.append(
                new_row(
                    f"GR091-{number}",
                    "GR091",
                    day,
                    f"{name.upper()} ({party.lower()})",
                    party,
                    f"{base} nyord{'x' * number}",
                )
            )
            old_rows.append(
                old_row(
                    f"GR091_{slug}_{party.lower()}",
                    "GR091",
                    day,
                    name.upper(),
                    party,
                    f"{base} gammalord{'y' * number}",
                )
            )
        uncapped = SETTINGS.model_copy(
            update={"listed_tokens": 50, "listed_deviations": 50, "listed_missing": 50}
        )
        report = self.reconcile_rows(tmp_path, new_rows, old_rows, uncapped)
        assert len(report.top_missing_tokens) == 5
        assert len(report.top_surplus_tokens) == 5
        assert len(report.largest_deviations) == 5
        assert len(report.largest_missing) == 5

        capped = uncapped.model_copy(
            update={"listed_tokens": 2, "listed_deviations": 3, "listed_missing": 4}
        )
        report = self.reconcile_rows(tmp_path, new_rows, old_rows, capped)
        assert len(report.top_missing_tokens) == 2
        assert len(report.top_surplus_tokens) == 2
        assert len(report.largest_deviations) == 3
        assert len(report.largest_missing) == 4
