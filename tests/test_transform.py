"""Tests for src/maktsprak_pipeline/pipeline/transform.py."""

from __future__ import annotations

from src.maktsprak_pipeline.pipeline.transform import (
    _SPEECH_RE,
    _canonical_party,
    _speaker_slug,
    records_from_text,
)


class TestSpeakerSlug:
    """The id's stability rests entirely on this function."""

    def test_folds_swedish_letters_to_latin(self):
        assert _speaker_slug("CECILIA WIGSTRÖM") == "cecilia-wigstrom"
        assert _speaker_slug("ÅSA ANDERSSON") == "asa-andersson"
        assert _speaker_slug("PÄR ÖBERG") == "par-oberg"

    def test_hyphen_and_space_spellings_agree(self):
        # Real collision from the corpus: the same person, two spellings, filed
        # as two separate speakers by the old raw-string grouping.
        assert _speaker_slug("JAN-EMANUEL JOHANSSON") == _speaker_slug("JAN EMANUEL JOHANSSON")

    def test_case_variants_agree(self):
        # Real collision: note the lower-case "i" in the second spelling.
        assert _speaker_slug("CECILIA WIGSTRÖM I GÖTEBORG") == _speaker_slug(
            "CECILIA WIGSTRÖM i Göteborg"
        )

    def test_different_people_stay_different(self):
        assert _speaker_slug("ANNA ANDERSSON") != _speaker_slug("ANNA ANDERSON")

    def test_is_deterministic(self):
        # The whole point: same input, same output, every run, forever.
        assert _speaker_slug("MARIA MALMER STENERGARD") == "maria-malmer-stenergard"
        assert _speaker_slug(" MARIA  MALMER  STENERGARD ") == "maria-malmer-stenergard"


class TestSpeechRegex:
    """Unit tests for the compiled Riksdag speech segmentation regex."""

    def test_extracts_single_speech(self):
        text = "Anf. 1 Anna Andersson (S): Vi måste satsa på välfärden."
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 1
        speaker, party, body = matches[0]
        assert speaker.strip() == "Anna Andersson"
        assert party == "S"
        assert "välfärden" in body

    def test_extracts_multiple_speakers(self, sample_speech_text):
        matches = _SPEECH_RE.findall(sample_speech_text)
        parties = [m[1] for m in matches]
        assert "S" in parties
        assert "M" in parties
        assert "V" in parties

    def test_stops_at_next_anf(self):
        text = (
            "Anf. 1 Anna (S): Första anförande om skolan. "
            "Anf. 2 Erik (M): Andra anförande om skatter."
        )
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 2
        # First match should not contain "skatter"
        assert "skatter" not in matches[0][2]

    def test_swedish_party_abbreviations(self):
        text = "Anf. 5 Karin Nilsson (KD): Vi värnar om familjen och kristna värden."
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 1
        assert matches[0][1] == "KD"

    def test_two_letter_party_with_swedish_chars(self):
        # Parties like "MP" contain only standard uppercase letters, but
        # the regex allows Å, Ä, Ö, ensure they are accepted.
        text = "Anf. 3 Åsa Andersson (MP): Klimat är vår tids stora fråga."
        matches = _SPEECH_RE.findall(text)
        assert matches[0][1] == "MP"

    def test_no_match_without_anf_prefix(self):
        text = "Anna Andersson (S): Det här är ingen del av protokollet."
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 0

    def test_speaker_name_with_hyphen(self):
        text = "Anf. 7 Anna-Karin Hatt (C): Centerpartiet vill se mer landsbygdsutveckling."
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 1
        assert "Anna-Karin Hatt" in matches[0][0]

    def test_reply_header_with_replik(self):
        # Reply headers place the colon after "replik", not after the party.
        text = "Anf. 8 Jimmie Åkesson (SD) replik: Herr talman! Det saknas svar."
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 1
        speaker, party, body = matches[0]
        assert speaker.strip() == "Jimmie Åkesson"
        assert party == "SD"
        assert "replik" not in body

    def test_party_less_header_does_not_swallow_next_speech(self):
        # A party-less speaker (independent "-", talman, unlisted minister) must
        # not cause the following speech to be absorbed and mis-attributed.
        text = (
            "Anf. 30 Elsa Widding (-): Herr talman! Ett oberoende inlägg. "
            "Anf. 31 Utrikesminister Maria Malmer Stenergard (M): Fru talman! Ett svar."
        )
        matches = _SPEECH_RE.findall(text)
        # Only the party-affiliated minister should match; the "-" member is skipped.
        assert len(matches) == 1
        speaker, party, body = matches[0]
        assert party == "M"
        assert "Maria Malmer Stenergard" in speaker
        assert "oberoende" not in body


class TestRecordIdStability:
    """The regression that cost a corpus rebuild.

    The id was ``f"{protocol_id}_{idx}"`` where idx came from ``enumerate()``
    over first-appearance order, a property of the parser run, not of the
    speech.  When the parser fix changed which speeches were extracted, every
    later index shifted, so ``HD098_60`` meant one speech before the fix and a
    different one after.  Re-ingesting then wrote both, and no join, dedup or
    upsert could tell them apart.
    """

    def _records(self, text):
        # Exercises the real record builder, a reimplementation here could
        # pass while the shipped parser was broken.
        return records_from_text(text, "P1", "2020-01-01", "http://x")

    def test_id_survives_an_extra_speech_appearing_earlier(self):
        # THE bug, reproduced. A parser fix makes an earlier speech visible;
        # every id after it must NOT move.
        before = "Anf. 2 ANNA ANDERSSON (S): Ett. Anf. 3 ERIK ERIKSSON (M): Tva."
        after = (
            "Anf. 1 NYA TALAREN (C): Noll. "
            "Anf. 2 ANNA ANDERSSON (S): Ett. Anf. 3 ERIK ERIKSSON (M): Tva."
        )
        ids_before = {r["id"] for r in self._records(before)}
        ids_after = {r["id"] for r in self._records(after)}
        # Anna and Erik keep their ids; only the new speaker adds one.
        assert ids_before < ids_after
        assert ids_after - ids_before == {"P1_nya-talaren_c"}

    def test_id_does_not_depend_on_document_order(self):
        a = "Anf. 1 ANNA ANDERSSON (S): Ett. Anf. 2 ERIK ERIKSSON (M): Tva."
        b = "Anf. 1 ERIK ERIKSSON (M): Tva. Anf. 2 ANNA ANDERSSON (S): Ett."
        assert {r["id"] for r in self._records(a)} == {r["id"] for r in self._records(b)}

    def test_same_name_different_party_stays_two_records(self):
        # Real case: JEPPE JOHNSSON appears under both S and M in one protocol.
        text = "Anf. 1 JEPPE JOHNSSON (S): Ett. Anf. 2 JEPPE JOHNSSON (M): Tva."
        assert {r["id"] for r in self._records(text)} == {
            "P1_jeppe-johnsson_s",
            "P1_jeppe-johnsson_m",
        }

    def test_spelling_variants_merge_into_one_record(self):
        text = "Anf. 1 JAN-EMANUEL JOHANSSON (S): Ett. Anf. 2 JAN EMANUEL JOHANSSON (S): Tva."
        records = self._records(text)
        assert len(records) == 1
        assert "Ett." in records[0]["text"] and "Tva." in records[0]["text"]

    def test_output_is_sorted_and_reproducible(self):
        text = "Anf. 1 ZULU (S): Z. Anf. 2 ALFA (M): A."
        first = self._records(text)
        assert [r["id"] for r in first] == sorted(r["id"] for r in first)
        assert first == self._records(text)


class TestCanonicalParty:
    """Historical party-abbreviation normalisation for backfilled protocols."""

    def test_folkpartiet_maps_to_liberalerna(self):
        # Pre-2015 protocols label the Liberals "FP"; they must fold into "L"
        # so drift analysis sees one continuous party.
        assert _canonical_party("FP") == "L"

    def test_current_parties_pass_through(self):
        for p in ["S", "M", "SD", "C", "V", "KD", "MP", "L"]:
            assert _canonical_party(p) == p

    def test_unknown_party_returns_none(self):
        assert _canonical_party("-") is None
        assert _canonical_party("XYZ") is None

    def test_lowercase_party_is_normalised(self):
        # Protocols up to ~2009 lower-case the party abbreviation.
        assert _canonical_party("m") == "M"
        assert _canonical_party("s") == "S"
        assert _canonical_party("fp") == "L"  # lowercase + historical rename


class TestSpeechRegexOldFormat:
    """Pre-2010 protocols lower-case the party in the header."""

    def test_lowercase_party_header_matches(self):
        text = "Anf. 2 HANS STENBERG (s): Herr talman! Jag vill börja med att tacka."
        matches = _SPEECH_RE.findall(text)
        assert len(matches) == 1
        speaker, party, body = matches[0]
        assert speaker.strip() == "HANS STENBERG"
        assert party == "s"  # raw capture; canonicalised downstream
        assert "talman" in body
