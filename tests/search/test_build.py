"""Tests for search/corpus/build.py and search/corpus/store.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

from search.config import Settings, load_settings
from search.corpus.build import (
    ANOTHER_PARTY_IN_SPEAKER,
    HARD_WRAPPED,
    LOST_EMPHASIS_BOUNDARY,
    NO_PARTY_IN_SPEAKER,
    NO_TEXT,
    ONLY_RECORD_TEXT,
    OUTSIDE_THE_EIGHT,
    TEST_RECORD,
    BuildReport,
    Outcome,
    build_corpus,
    heading_pairs_in,
    headings_in,
    parties_in_speaker,
    read_record,
)
from search.corpus.fetch import MANIFEST_NAME, FetchedFile, Manifest, sha256_of, write_manifest
from search.corpus.parse import NO_HEADINGS, Headings, index_headings
from search.corpus.store import SPEECHES_NAME, iter_speeches
from search.models import deep_link

FIELD_CODE = "STYLEREF Kantrubrik \\* MERGEFORMAT Svar på interpellationer"
SPEECH_WITH_RECORD_TEXT = (
    "<p>Herr talman! Vi vann valet.</p><p>(Applåder)</p>"
    f"<p>{FIELD_CODE}</p><p>Svar på interpellationer</p>"
    "<p>Nu börjar arbetet.</p><p>Överläggningen var härmed avslutad.</p>"
    "<p>(Beslut fattades under § 14.)</p>"
)
SPEECH_IN_LINES = "\r\nFru talman! Man får ha bara \r\nen\r\n kund.\r\n"
LOST_SPACES = "\r\nMan talar om w\r\nider Europe\r\n och allt det där.\r\n"
SPEECH_TITLE = "Fru talman! Jag vill börja med att yrka bifall till reservationerna 1, 2, 4 och 5."
WRAPPED = "Fru talman! Många barnfamil-\njer har haft det jobbigt.\n"
STRAY_TEXT = "<p>Fru talman!</p>Tack."
FIRST_SESSION = "2003/04"


def raw(number: int = 1, text: str | None = "<p>Fru talman! Tack.</p>", **changes: object) -> dict:
    """Build a source record as the Riksdag publishes it."""
    record: dict[str, object] = {
        "dok_hangar_id": "1",
        "dok_id": "H209100",
        "dok_titel": "Protokoll 2014/15:100 Tisdagen den 12 maj",
        "dok_rm": "2014/15",
        "dok_nummer": "100",
        "dok_datum": "2015-05-12 00:00:00",
        "avsnittsrubrik": "Svar på interpellation  2014/15:1 om skolan",
        "underrubrik": None,
        "kammaraktivitet": "interpellationsdebatt",
        "anforande_id": "a6354706",
        "anforande_nummer": str(number),
        "talare": "Anna Andersson (S)",
        "parti": "S",
        "anforandetext": text,
        "intressent_id": "0123456789012",
        "rel_dok_id": "H21001",
        "replik": "N",
        "systemdatum": "2015-05-13 10:16:38",
    }
    record.update(changes)
    return record


def write_dataset(path: Path, records: list[dict]) -> Path:
    """Write records to a dataset zip, every other one with a byte-order mark."""
    with zipfile.ZipFile(path, "w") as archive:
        for index, record in enumerate(records):
            payload = json.dumps({"anforande": record}, ensure_ascii=False).encode("utf-8")
            if index % 2 == 0:
                payload = b"\xef\xbb\xbf" + payload
            archive.writestr(f"{record['dok_id']}-{record['anforande_nummer']}.json", payload)
    return path


def session_of(dataset: Path) -> str:
    """Read the session from a dataset's name: ``anforande-201415.json.zip`` is 2014/15."""
    digits = dataset.name.removeprefix("anforande-")[:6]
    return f"{digits[:4]}/{digits[4:]}"


def manifest_for(*datasets: Path) -> Manifest:
    """Record datasets in a manifest, as a fetch would have."""
    return Manifest(
        files=tuple(
            FetchedFile(
                session=session_of(dataset),
                file=dataset.name,
                url=f"https://example.org/{dataset.name}",
                bytes=dataset.stat().st_size,
                sha256=sha256_of(dataset),
                fetched=date(2026, 11, 20),
            )
            for dataset in datasets
        )
    )


def read(record: dict, headings: Headings = NO_HEADINGS) -> Outcome:
    """Read one record into a corpus that starts at ``FIRST_SESSION``."""
    return read_record(record, headings, first_session=FIRST_SESSION)


def build(
    manifest: Manifest, raw_dir: Path, destination: Path, first_session: str | None = None
) -> BuildReport:
    """Build from the sessions the manifest holds.

    The corpus starts at *first_session*, or at the manifest's first session
    when none is given.
    """
    sessions = [entry.session for entry in manifest.files]
    return build_corpus(
        manifest,
        raw_dir,
        destination,
        sessions=sessions,
        first_session=first_session or sessions[0],
    )


class TestReadRecord:
    def test_maps_the_published_fields(self):
        record = raw(
            number=7,
            parti="fp",
            replik="Y",
            kammaraktivitet="-",
            intressent_id="",
            rel_dok_id=None,
            talare="Statsrådet  ANNA ANDERSSON (fp)",
        )
        speech = read(record).speech

        assert speech is not None
        assert speech.anf_id == "H209100-7"
        assert (speech.rm, speech.dok_nummer, speech.anf_nummer) == ("2014/15", 100, 7)
        assert speech.date == date(2015, 5, 12)
        assert speech.party == "L"
        assert speech.is_reply is True
        assert speech.speaker_raw == "Statsrådet ANNA ANDERSSON (fp)"
        assert speech.title == "Svar på interpellation 2014/15:1 om skolan"
        assert (speech.activity, speech.person_id, speech.rel_dok_id) == (None, None, None)
        assert speech.layout == "html"
        assert speech.paragraphs == ("Fru talman! Tack.",)
        assert speech.url == deep_link("H209100", 7)

    def test_the_records_notes_stay_beside_the_speech(self):
        headings = index_headings({"Svar på interpellationer"})
        outcome = read(raw(text=SPEECH_WITH_RECORD_TEXT), headings)

        assert outcome.speech is not None
        assert outcome.speech.paragraphs == ("Herr talman! Vi vann valet.", "Nu börjar arbetet.")
        assert outcome.speech.notes == (
            "(Applåder)",
            "Överläggningen var härmed avslutad.",
            "(Beslut fattades under § 14.)",
        )
        assert [item.rule for item in outcome.set_aside] == [
            "insertion",
            "field_code",
            "heading",
            "closing_note",
            "after_closing",
        ]

    @pytest.mark.parametrize(
        ("changes", "reason"),
        [
            ({"parti": "TALMANNEN"}, OUTSIDE_THE_EIGHT),
            ({"parti": None}, OUTSIDE_THE_EIGHT),
            ({"parti": "-"}, OUTSIDE_THE_EIGHT),
            ({"anforandetext": None}, NO_TEXT),
            ({"anforandetext": "  \r\n "}, NO_TEXT),
            ({"dok_rm": "2002/03", "anforandetext": WRAPPED}, HARD_WRAPPED),
            ({"talare": "Anna Andersson (M)"}, ANOTHER_PARTY_IN_SPEAKER),
            ({"anforandetext": "<p>(Applåder)</p>"}, ONLY_RECORD_TEXT),
            ({"dok_id": "H909999", "anforandetext": "<p>Askjdhajksdhjka</p>"}, TEST_RECORD),
        ],
    )
    def test_a_record_left_out_names_its_reason(self, changes, reason):
        outcome = read(raw(**changes))
        assert outcome.speech is None
        assert outcome.excluded_as == reason

    def test_an_unknown_reply_flag_raises(self):
        with pytest.raises(ValueError, match="^H209100-1: unknown reply flag"):
            read(raw(replik="J"))

    def test_a_date_outside_the_session_raises(self):
        with pytest.raises(ValueError, match="outside session"):
            read(raw(dok_datum="2016-12-16 00:00:00"))

    def test_a_wrapped_record_in_a_session_the_corpus_keeps_raises(self):
        with pytest.raises(ValueError, match="^H209100-4: the text is hard-wrapped in 2003/04"):
            read(raw(number=4, dok_rm="2003/04", text=WRAPPED))

    def test_a_record_with_line_breaks_before_the_first_session_raises(self):
        with pytest.raises(
            ValueError, match="^H209100-4: the text has the lines layout in 2002/03"
        ):
            read(raw(number=4, dok_rm="2002/03", text=SPEECH_IN_LINES))

    def test_a_record_with_paragraph_elements_before_the_first_session_raises(self):
        with pytest.raises(ValueError, match="^H209100-1: the text has the html layout in 2002/03"):
            read(raw(dok_rm="2002/03"))

    def test_an_error_in_reading_the_text_names_the_record(self):
        with pytest.raises(ValueError, match="^H209100-7: Text outside the paragraph elements"):
            read(raw(number=7, text=STRAY_TEXT))

    def test_a_field_the_speech_model_refuses_names_the_record_once(self):
        with pytest.raises(ValueError, match="H209100-7: 2016-12-16 lies outside") as raised:
            read(raw(number=7, dok_datum="2016-12-16 00:00:00"))

        assert str(raised.value).count("H209100-7: ") == 1

    def test_an_error_without_the_key_is_given_it_in_front(self):
        with pytest.raises(ValueError, match="^H209100-7: Invalid isoformat string"):
            read(raw(number=7, dok_datum="12 maj 2015"))

    def test_a_heading_field_that_opens_like_a_speech_is_no_title(self):
        speech = read(raw(avsnittsrubrik=SPEECH_TITLE)).speech
        assert speech is not None
        assert speech.title is None

    def test_a_break_where_the_source_lost_the_spaces_is_flagged_and_kept(self):
        outcome = read(raw(text=LOST_SPACES))
        assert outcome.speech is not None
        assert outcome.speech.paragraphs == ("Man talar om w", "ider Europe och allt det där.")
        assert outcome.flags == (LOST_EMPHASIS_BOUNDARY,)

    def test_a_speech_with_no_known_defect_carries_no_flag(self):
        assert read(raw()).flags == ()


class TestPartyInTheSpeakerField:
    @pytest.mark.parametrize(
        ("speaker", "parties"),
        [
            ("Anna Andersson (S)", {"S"}),
            ("ANNA ANDERSSON (kd)", {"KD"}),
            ("ANNA ANDERSSON (fp)", {"L"}),
            ("ANNA ANDERSSON (FP)", {"L"}),
            ("ANNA ANDERSSON (m) replik", {"M"}),
            ("ANNA ANDERSSON (M) (replik)", {"M"}),
            ("Andre vice talman ANNA ANDERSSON (fp)", {"L"}),
            ("Utbildningsminister ANNA ANDERSSON (S)", {"S"}),
            ("ANNA ANDERSSON (S) (M)", {"S", "M"}),
            ("ANNA ANDERSSON (replik)", set()),
            ("ANNA ANDERSSON (S", set()),
            ("TALMANNEN", set()),
        ],
    )
    def test_reads_the_parties_named_in_parentheses(self, speaker, parties):
        assert parties_in_speaker(speaker) == parties

    @pytest.mark.parametrize(
        ("speaker", "party_field", "party"),
        [
            ("Anna Andersson (S)", "S", "S"),
            ("ANNA ANDERSSON (s)", "S", "S"),
            ("ANNA ANDERSSON (fp)", "fp", "L"),
            ("ANNA ANDERSSON (fp)", "L", "L"),
            ("ANNA ANDERSSON (L)", "FP", "L"),
            ("ANNA ANDERSSON (m) replik", "M", "M"),
            ("ANNA ANDERSSON (M) (replik)", "M", "M"),
            ("Utbildningsminister ANNA ANDERSSON (S)", "S", "S"),
        ],
    )
    def test_a_speaker_field_that_names_the_records_party_is_kept_unflagged(
        self, speaker, party_field, party
    ):
        outcome = read(raw(talare=speaker, parti=party_field))
        assert outcome.speech is not None
        assert (outcome.speech.speaker_raw, outcome.speech.party) == (speaker, party)
        assert outcome.flags == ()

    @pytest.mark.parametrize(
        ("speaker", "party_field"),
        [
            ("Anna Andersson (M)", "S"),
            ("ANNA ANDERSSON (fp)", "M"),
            ("ANNA ANDERSSON (s) replik", "V"),
            ("Utbildningsminister ANNA ANDERSSON (S) (M)", "S"),
        ],
    )
    def test_a_speaker_field_that_names_another_party_is_left_out(self, speaker, party_field):
        outcome = read(raw(talare=speaker, parti=party_field))
        assert outcome.speech is None
        assert outcome.excluded_as == ANOTHER_PARTY_IN_SPEAKER
        assert outcome.set_aside == ()

    @pytest.mark.parametrize("speaker", ["Anna Andersson", "ANNA ANDERSSON (replik)"])
    def test_a_speaker_field_that_names_no_party_is_kept_and_flagged(self, speaker):
        outcome = read(raw(talare=speaker))
        assert outcome.speech is not None
        assert outcome.speech.party == "S"
        assert outcome.flags == (NO_PARTY_IN_SPEAKER,)

    def test_a_speech_with_two_known_defects_carries_both_flags(self):
        outcome = read(raw(text=LOST_SPACES, talare="Anna Andersson"))
        assert outcome.flags == (LOST_EMPHASIS_BOUNDARY, NO_PARTY_IN_SPEAKER)


class TestHeadings:
    def test_gathers_debate_titles_and_margin_headings(self):
        records = [raw(text=SPEECH_WITH_RECORD_TEXT), raw(underrubrik="Skolan", text=None)]
        assert headings_in(records) == {
            "Svar på interpellation 2014/15:1 om skolan",
            "Svar på interpellationer",
            "Skolan",
        }

    def test_a_heading_field_that_opens_like_a_speech_is_not_gathered(self):
        assert headings_in([raw(avsnittsrubrik=SPEECH_TITLE, text=None)]) == set()

    def test_an_error_in_reading_a_text_with_a_field_code_names_the_record(self):
        record = raw(number=3, text=f"<p>{FIELD_CODE}</p>Tack.")
        with pytest.raises(ValueError, match="^H209100-3: Text outside the paragraph elements"):
            headings_in([record])

    def test_gathers_the_titles_that_come_with_a_subtitle(self):
        records = [
            raw(underrubrik="Skolan"),
            raw(),
            raw(avsnittsrubrik=SPEECH_TITLE, underrubrik="Skolan"),
            raw(avsnittsrubrik="Skolan", underrubrik=SPEECH_TITLE),
        ]
        assert heading_pairs_in(records) == {
            ("Svar på interpellation 2014/15:1 om skolan", "Skolan")
        }


class TestBuild:
    def records(self) -> list[dict]:
        return [
            raw(number=2, text=SPEECH_WITH_RECORD_TEXT),
            raw(number=1, text=SPEECH_IN_LINES, parti="fp", talare="ANNA ANDERSSON (fp)"),
            raw(number=3, parti="TALMANNEN"),
            raw(number=4, text=None),
            raw(number=6, text="<p>(Applåder)</p>"),
            raw(number=7, dok_id="H909999", text="<p>Sdf lksjdklf</p>"),
            raw(
                number=8,
                text=LOST_SPACES,
                intressent_id="",
                kammaraktivitet="-",
                rel_dok_id=None,
                avsnittsrubrik=None,
            ),
        ]

    def test_every_record_is_kept_or_counted(self, tmp_path):
        early = write_dataset(
            tmp_path / "anforande-200203.json.zip",
            [raw(number=5, dok_rm="2002/03", text=WRAPPED)],
        )
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", self.records())
        destination = tmp_path / "search" / "anforanden.parquet"

        report = build(manifest_for(early, dataset), tmp_path, destination, "2014/15")

        assert report.source_records == 8
        assert report.speeches == 3
        assert report.excluded == {
            HARD_WRAPPED: 1,
            NO_TEXT: 1,
            ONLY_RECORD_TEXT: 1,
            OUTSIDE_THE_EIGHT: 1,
            TEST_RECORD: 1,
        }
        assert sum(report.excluded.values()) + report.speeches == report.source_records
        assert report.excluded_party_labels == {"TALMANNEN": 1}
        assert report.hyphenated_line_ends_in_wrapped == 1
        assert report.set_aside == {
            "after_closing": 1,
            "closing_note": 1,
            "field_code": 1,
            "heading": 1,
            "insertion": 2,
        }
        assert report.dropped_headings == {"Svar på interpellationer": 1}
        assert report.flagged == {LOST_EMPHASIS_BOUNDARY: ("H209100-8",)}
        assert report.speeches_with_notes == 1
        assert report.speeches_by_party == {"L": 1, "S": 2}
        assert (report.paragraphs, report.words) == (5, 26)
        assert report.words_per_speech["max"] == 10

        early_session, session = report.sessions
        assert (early_session.session, early_session.layouts) == ("2002/03", {"wrapped": 1})
        assert session.session == "2014/15"
        assert session.layouts == {"html": 3, "lines": 2}
        assert (session.first_date, session.last_date) == (date(2015, 5, 12), date(2015, 5, 12))

    def test_the_speeches_are_stored_in_order_and_read_back_whole(self, tmp_path):
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", self.records())
        destination = tmp_path / "anforanden.parquet"
        build(manifest_for(dataset), tmp_path, destination)

        speeches = list(iter_speeches(destination))

        assert [speech.anf_id for speech in speeches] == ["H209100-1", "H209100-2", "H209100-8"]
        assert speeches[0].paragraphs == ("Fru talman! Man får ha bara en kund.",)
        assert speeches[0].layout == "lines"
        assert speeches[1].notes[0] == "(Applåder)"

    def test_a_speech_without_the_optional_fields_is_read_back_as_built(self, tmp_path):
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", self.records())
        destination = tmp_path / "anforanden.parquet"
        build(manifest_for(dataset), tmp_path, destination)

        stored = {speech.anf_id: speech for speech in iter_speeches(destination)}

        expected = read(self.records()[-1]).speech
        assert expected is not None
        assert (expected.person_id, expected.activity, expected.title) == (None, None, None)
        assert stored["H209100-8"] == expected

    def test_the_margin_heading_of_another_debate_is_dropped(self, tmp_path):
        chapters = "Granskningsbetänkande"
        records = [
            raw(
                number=1,
                avsnittsrubrik=chapters,
                underrubrik="Vissa frågor om statsråds tjänsteutövning (kapitel 3)",
                text="<p>Herr talman! Nu till kapitel 3.</p>",
            ),
            raw(
                number=2,
                avsnittsrubrik=chapters,
                underrubrik="Handläggning av vissa regeringsärenden (kapitel 2)",
                text=(
                    "<p>Herr talman! Vi börjar med kapitel 2.</p>"
                    "<p>Gransknings-betänkandeVissa frågor om statsråds tjänste-utövning "
                    "(kapitel 3)</p><p>Det var allt.</p>"
                ),
            ),
        ]
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", records)
        destination = tmp_path / "anforanden.parquet"

        report = build(manifest_for(dataset), tmp_path, destination)

        assert report.set_aside == {"heading_with_subtitle": 1}
        stored = {speech.anf_id: speech for speech in iter_speeches(destination)}
        assert stored["H209100-2"].paragraphs == (
            "Herr talman! Vi börjar med kapitel 2.",
            "Det var allt.",
        )

    def test_the_same_inputs_give_the_same_report(self, tmp_path):
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", self.records())
        manifest = manifest_for(dataset)
        first = build(manifest, tmp_path, tmp_path / "a.parquet")
        second = build(manifest, tmp_path, tmp_path / "b.parquet")
        assert first == second
        assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()

    def test_a_key_that_occurs_twice_stops_the_build_and_writes_nothing(self, tmp_path):
        first = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=1)])
        second = write_dataset(
            tmp_path / "anforande-201516.json.zip",
            [raw(number=1, dok_rm="2015/16", dok_datum="2016-05-12 00:00:00")],
        )
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(ValueError, match="occurs twice"):
            build(manifest_for(first, second), tmp_path, destination)

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_dataset_that_changed_since_it_was_fetched_stops_the_build(self, tmp_path):
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=1)])
        manifest = manifest_for(dataset)
        write_dataset(dataset, [raw(number=1), raw(number=2)])

        with pytest.raises(ValueError, match="does not match the SHA-256"):
            build(manifest, tmp_path, tmp_path / "anforanden.parquet")

    @pytest.mark.parametrize(
        ("sessions", "missing"),
        [
            (["2014/15", "2015/16"], "2015/16"),
            (["2013/14", "2014/15", "2015/16"], "2013/14, 2015/16"),
        ],
    )
    def test_a_manifest_that_lacks_a_session_stops_the_build_and_writes_nothing(
        self, tmp_path, sessions, missing
    ):
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=1)])
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(ValueError, match=f"^The manifest lacks: {missing}[.] Run step 01"):
            build_corpus(
                manifest_for(dataset),
                tmp_path,
                destination,
                sessions=sessions,
                first_session="2014/15",
            )

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_manifest_with_a_session_outside_the_settings_stops_the_build(self, tmp_path):
        first = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=1)])
        second = write_dataset(tmp_path / "anforande-201516.json.zip", [raw(number=2)])
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(ValueError, match="outside the sessions the settings name: 2015/16[.]$"):
            build_corpus(
                manifest_for(first, second),
                tmp_path,
                destination,
                sessions=["2014/15"],
                first_session="2014/15",
            )

        assert not destination.exists()

    @pytest.mark.parametrize("first_session", ["2003/04", "2020/21"])
    def test_a_first_session_outside_the_sessions_stops_the_build_and_writes_nothing(
        self, tmp_path, first_session
    ):
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=1)])
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(ValueError, match=f"first session, {first_session}, is not one of"):
            build_corpus(
                manifest_for(dataset),
                tmp_path,
                destination,
                sessions=["2014/15"],
                first_session=first_session,
            )

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_session_whose_dataset_holds_no_record_stops_the_build_and_writes_nothing(
        self, tmp_path
    ):
        first = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=1)])
        empty = write_dataset(tmp_path / "anforande-201516.json.zip", [])
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(ValueError, match="^The dataset of session 2015/16 holds no record[.]$"):
            build(manifest_for(first, empty), tmp_path, destination)

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_record_of_an_earlier_session_in_a_kept_sessions_dataset_stops_the_build(
        self, tmp_path
    ):
        records = [raw(number=1), raw(number=2, dok_rm="2002/03", text=WRAPPED)]
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", records)
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(
            ValueError,
            match=(
                "^H209100-2: the record names session 2002/03 in the dataset of session 2014/15[.]$"
            ),
        ):
            build(manifest_for(dataset), tmp_path, destination)

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_record_of_a_later_session_in_an_earlier_sessions_dataset_stops_the_build(
        self, tmp_path
    ):
        early = write_dataset(tmp_path / "anforande-200203.json.zip", [raw(number=1)])
        kept = write_dataset(tmp_path / "anforande-201415.json.zip", [raw(number=2)])
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(
            ValueError,
            match=(
                "^H209100-1: the record names session 2014/15 in the dataset of session 2002/03[.]$"
            ),
        ):
            build(manifest_for(early, kept), tmp_path, destination, "2014/15")

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_session_before_the_first_is_read_counted_and_left_out(self, tmp_path):
        early = [
            raw(number=1, dok_id="GQ0910", dok_rm="2002/03", text=WRAPPED),
            raw(number=2, dok_id="GQ0910", dok_rm="2002/03", text=WRAPPED, parti="fp"),
            raw(number=3, dok_id="GQ0910", dok_rm="2002/03", text=WRAPPED, parti="TALMANNEN"),
        ]
        kept = [raw(dok_id="GR0910", dok_rm="2003/04", dok_datum="2003-10-14 00:00:00")]
        first = write_dataset(tmp_path / "anforande-200203.json.zip", early)
        second = write_dataset(tmp_path / "anforande-200304.json.zip", kept)
        destination = tmp_path / "anforanden.parquet"

        report = build(manifest_for(first, second), tmp_path, destination, FIRST_SESSION)

        assert report.excluded == {HARD_WRAPPED: 2, OUTSIDE_THE_EIGHT: 1}
        assert report.hyphenated_line_ends_in_wrapped == 2
        early_session, kept_session = report.sessions
        assert (early_session.session, early_session.source_records) == ("2002/03", 3)
        assert (early_session.speeches, early_session.first_date) == (0, None)
        assert early_session.layouts == {"wrapped": 2}
        assert (kept_session.session, kept_session.speeches) == ("2003/04", 1)
        assert [speech.anf_id for speech in iter_speeches(destination)] == ["GR0910-1"]

    def test_a_wrapped_record_in_a_session_the_corpus_keeps_stops_the_build(self, tmp_path):
        records = [raw(number=1, dok_rm="2003/04", text=WRAPPED)]
        dataset = write_dataset(tmp_path / "anforande-200304.json.zip", records)
        destination = tmp_path / "anforanden.parquet"

        with pytest.raises(ValueError, match="^H209100-1: the text is hard-wrapped in 2003/04"):
            build(manifest_for(dataset), tmp_path, destination)

        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_the_party_in_the_speaker_field_is_accounted_for(self, tmp_path):
        records = [
            raw(number=1),
            raw(number=2, talare="Anna Andersson (M)"),
            raw(number=3, talare="Anna Andersson"),
        ]
        dataset = write_dataset(tmp_path / "anforande-201415.json.zip", records)

        report = build(manifest_for(dataset), tmp_path, tmp_path / "anforanden.parquet")

        assert report.speeches == 2
        assert report.excluded == {ANOTHER_PARTY_IN_SPEAKER: 1}
        assert report.flagged == {NO_PARTY_IN_SPEAKER: ("H209100-3",)}


def load_build_script() -> ModuleType:
    """Import search/pipeline/02_build_anforanden.py, whose name is not a module name."""
    path = Path(__file__).resolve().parents[2] / "search" / "pipeline" / "02_build_anforanden.py"
    spec = importlib.util.spec_from_file_location("build_anforanden", path)
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBuildScript:
    """Step 02's main, run on a temporary data root with three sessions."""

    script = load_build_script()

    @staticmethod
    def three_sessions(args: argparse.Namespace, reports: Path) -> Settings:
        """Load the settings for the data root the command line names.

        The datasets run from 2002/03 to 2004/05, the corpus starts at
        2003/04, and the tracked reports go to *reports*.
        """
        settings = load_settings(data_root=args.data_root)
        return settings.model_copy(
            update={
                "source": settings.source.model_copy(
                    update={"first_session": "2002/03", "last_session": "2004/05"}
                ),
                "corpus": settings.corpus.model_copy(update={"first_session": "2003/04"}),
                "tracked": settings.tracked.model_copy(update={"reports": reports}),
            }
        )

    def test_the_script_reads_the_fetched_sessions_into_a_corpus_from_its_first_session(
        self, tmp_path, monkeypatch
    ):
        """The datasets run from ``source.first_session``, the corpus from ``corpus.first_session``.

        The 2002/03 dataset holds a hard-wrapped record. Read as a session the
        corpus keeps, it stops the build; left out of the sessions, the
        manifest holds a dataset the build does not read and the build stops.
        """
        reports = tmp_path / "reports"
        monkeypatch.setattr(
            self.script, "settings_from", lambda args: self.three_sessions(args, reports)
        )
        monkeypatch.setattr(sys, "argv", ["02_build_anforanden.py", "--data-root", str(tmp_path)])
        settings = self.three_sessions(argparse.Namespace(data_root=tmp_path), reports)
        raw_dir = settings.data.raw_anforanden
        raw_dir.mkdir(parents=True)
        datasets = [
            write_dataset(
                raw_dir / "anforande-200203.json.zip",
                [raw(number=1, dok_rm="2002/03", text=WRAPPED)],
            ),
            write_dataset(
                raw_dir / "anforande-200304.json.zip",
                [raw(number=2, dok_rm="2003/04", dok_datum="2003-10-14 00:00:00")],
            ),
            write_dataset(
                raw_dir / "anforande-200405.json.zip",
                [raw(number=3, dok_rm="2004/05", dok_datum="2004-10-14 00:00:00")],
            ),
        ]
        write_manifest(manifest_for(*datasets), reports / MANIFEST_NAME)

        self.script.main()

        report = json.loads((reports / self.script.BUILD_REPORT_NAME).read_text(encoding="utf-8"))
        assert [session["session"] for session in report["sessions"]] == [
            "2002/03",
            "2003/04",
            "2004/05",
        ]
        assert report["excluded"] == {HARD_WRAPPED: 1}
        speeches = iter_speeches(settings.data.search / SPEECHES_NAME)
        assert [speech.anf_id for speech in speeches] == ["H209100-2", "H209100-3"]
