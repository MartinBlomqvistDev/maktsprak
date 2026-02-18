"""Tests for src/maktsprak_pipeline/pipeline/transform.py."""

from __future__ import annotations

from src.maktsprak_pipeline.pipeline.transform import _SPEECH_RE


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
        # the regex allows Å, Ä, Ö — ensure they are accepted.
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
