"""Tests for the launch-set tone dimensions (vi_dom, readability, inclusive).

Several of these pin findings from the corpus audit rather than abstract
behaviour — the patterns that were *cut*, and why, matter as much as the ones
that were kept, because re-adding them would silently restore a measure that
counts sympathy as hostility.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.maktsprak_pipeline.nlp.tone import inclusive, readability, vi_dom
from src.maktsprak_pipeline.nlp.tone.kernel import TONE_DIMENSIONS, aggregate_cells


def _speeches(rows: list[dict]) -> pd.DataFrame:
    base = {"party": "S", "year": 2020, "protocol_id": "H1", "speaker": "NN", "file_url": None}
    return pd.DataFrame([{**base, **r} for r in rows])


class TestViDomPatternTable:
    def test_table_loads_and_is_free_of_duplicates(self):
        table = vi_dom.load_patterns()
        assert not table["pattern"].duplicated().any()
        assert set(table["slot"]) == {"ingroup", "outgroup"}

    @pytest.mark.parametrize("cut", ["dessa människor", "de här människorna", "eliten"])
    def test_patterns_the_audit_rejected_stay_out(self, cut):
        # "dessa människor" / "de här människorna": sampled matches were
        # overwhelmingly SYMPATHETIC ("vi har all anledning att stötta de här
        # människorna") — they point at a group, they do not other it.
        # Bare "eliten" also means a top tier ("den internationella eliten" =
        # world-class researchers). Re-adding any of these silently restores a
        # measure that counts compassion as hostility.
        assert cut not in set(vi_dom.load_patterns()["pattern"])

    def test_qualified_elite_forms_are_kept(self):
        patterns = set(vi_dom.load_patterns()["pattern"])
        assert "politiska eliten" in patterns
        assert "etablissemanget" in patterns

    def test_left_and_right_outgroups_are_both_represented(self):
        # The symmetry commitment, enforced in code: an out-group defined by
        # wealth and one defined by origin must both be measurable, or the
        # instrument can only ever find the pattern in one direction.
        subtypes = set(vi_dom.load_patterns()["subtype"])
        assert {"economic", "origin", "elite"} <= subtypes


class TestViDomMeasures:
    def test_folk_counts_sentences_not_markers(self):
        # Two markers in one sentence is still one sentence: a rhetorical
        # flourish must not outweigh a whole debate.
        df = _speeches([{"text": "Svenska folket och vanligt folk vet. Andra meningen."}])
        out = vi_dom.measure_folk(df)
        assert out["hits"].iloc[0] == 1
        assert out["n"].iloc[0] == 2

    def test_klasskonflikt_matches_wealth_outgroup(self):
        df = _speeches([{"text": "Skatten sänks för de rikaste. Det är fel."}])
        assert vi_dom.measure_klasskonflikt(df)["hits"].iloc[0] == 1

    def test_klasskonflikt_ignores_folk_references(self):
        df = _speeches([{"text": "Svenska folket förtjänar bättre."}])
        assert vi_dom.measure_klasskonflikt(df)["hits"].iloc[0] == 0

    def test_longest_pattern_wins_the_alternation(self):
        # "de allra rikaste" must not be truncated to "de rikaste"/"de rika".
        df = _speeches([{"text": "Det gynnar de allra rikaste i landet."}])
        spans = vi_dom.measure_klasskonflikt(df)["spans"].iloc[0]
        assert [p for _, _, p in spans] == ["de allra rikaste"]

    def test_richest_countries_is_not_class_conflict(self):
        # The audit caught this live: "Sverige är inte längre ett av de rikaste
        # länderna i världen" (M) was counted as framing a wealthy out-group.
        # It is about countries. exclude_next keeps the pattern and drops the
        # homograph rather than throwing away a useful marker.
        df = _speeches([{"text": "Sverige är inte längre ett av de rikaste länderna i världen."}])
        assert vi_dom.measure_klasskonflikt(df)["hits"].iloc[0] == 0

    def test_richest_people_still_counts(self):
        df = _speeches([{"text": "Skattesänkningar för de rikaste är fel prioritering."}])
        assert vi_dom.measure_klasskonflikt(df)["hits"].iloc[0] == 1

    def test_measures_feed_the_kernel_unchanged(self):
        df = _speeches(
            [
                {"text": "Vanligt folk betalar. De rika slipper.", "party": "V"},
                {"text": "Ingenting relevant här alls.", "party": "M"},
            ]
        )
        cells = aggregate_cells(vi_dom.measure_folk(df))
        assert cells["V"][2020].hits == 1
        assert cells["M"][2020].hits == 0


class TestViDomCensus:
    def test_construction_needs_both_halves_in_one_sentence(self):
        both = _speeches(
            [{"text": "Medan vanligt folk halkar efter drar de rikaste ifrån.", "party": "V"}]
        )
        assert len(vi_dom.vi_dom_census(both)) == 1

    def test_markers_in_different_sentences_are_not_a_construction(self):
        # This is the whole point of the construction detector: an in-group
        # marker somewhere and an out-group marker somewhere else in the same
        # speech is not an us-versus-them sentence.
        apart = _speeches([{"text": "Vanligt folk betalar skatt. De rika bor i Danmark."}])
        assert vi_dom.vi_dom_census(apart) == []

    def test_census_records_the_sentence_and_its_subtype(self):
        df = _speeches(
            [
                {
                    "text": "Skatt för de rikaste i stället för välfärd åt svenska folket.",
                    "party": "S",
                }
            ]
        )
        found = vi_dom.vi_dom_census(df)[0]
        assert found.party == "S"
        assert found.subtypes == ["economic"]
        assert "svenska folket" in found.ingroup
        assert "de rikaste" in found.outgroup
        assert found.sentence.startswith("Skatt för de rikaste")


class TestLix:
    def test_formula_matches_a_hand_computed_value(self):
        # 10 words, 2 sentences, 3 long (>6 letters):
        #   LIX = 10/2 + (3 * 100 / 10) = 5 + 30 = 35
        assert readability.lix(words=10, sentences=2, long_words=3) == pytest.approx(35.0)

    def test_empty_input_is_zero_not_a_crash(self):
        assert readability.lix(0, 0, 0) == 0.0

    def test_long_word_threshold_is_more_than_six_letters(self):
        # "detta" (5) and "mening" (6) are short; "regeringen" (10) and
        # "sjukvården" (10) are long. Björnsson's cut is *more than* six.
        df = _speeches([{"text": "Detta är en mening med regeringen och sjukvården."}])
        out = readability.measure_lix(df)
        assert out["long_words"].iloc[0] == 2

    def test_short_speeches_are_excluded_from_the_pool(self):
        # "Ja, herr talman!" is a speech. It says nothing about how a party
        # speaks, and its per-speech LIX is noise.
        df = _speeches([{"text": "Ja, herr talman!"}])
        assert not readability.measure_lix(df)["included"].iloc[0]

    def test_pooled_lix_is_not_the_average_of_per_speech_lix(self):
        # THE regression for this dimension. One long, dense speech and one
        # long, plain one: pooling computes a single index from the summed
        # counts; averaging would let a short speech swing it.
        dense = " ".join(["regeringens sjukvårdspolitik betraktas kritiskt."] * 40)
        plain = " ".join(["vi ska nu se till att det blir bra för alla."] * 40)
        df = _speeches([{"text": dense, "party": "A"}, {"text": plain, "party": "A"}])
        measured = readability.measure_lix(df)
        pooled = readability.aggregate_lix(measured)["A"][2020]

        per_speech = [
            readability.lix(w, s, lw)
            for w, s, lw in zip(
                measured["words"], measured["sentences"], measured["long_words"], strict=True
            )
        ]
        naive_average = sum(per_speech) / len(per_speech)

        expected = readability.lix(
            int(measured["words"].sum()),
            int(measured["sentences"].sum()),
            int(measured["long_words"].sum()),
        )
        assert pooled.rate == pytest.approx(expected)
        assert pooled.rate != pytest.approx(naive_average)

    def test_thin_cells_are_dropped_on_sentences_not_speeches(self):
        # A speech count cannot see this: plenty of speeches, almost no text.
        df = _speeches([{"text": "En mening här. Och en till. Och en tredje kort mening nu."}] * 10)
        assert readability.aggregate_lix(readability.measure_lix(df)) == {}

    def test_z_is_none_because_lix_is_not_a_proportion(self):
        text = " ".join(["regeringen överväger sjukvårdspolitiska förändringar nu"] * 20) + "."
        df = _speeches([{"text": f"{text} {text} {text}"}] * 30)
        cell = readability.aggregate_lix(readability.measure_lix(df))["S"][2020]
        assert cell.z is None
        assert cell.extra["sentences"] >= readability.MIN_CELL_SENTENCES


class TestInclusive:
    def test_hen_matches_the_pronoun_only(self):
        df = _speeches([{"text": "Hen sade det. Henrik höll med. Hens uppfattning var klar."}])
        out = inclusive.measure_hen(df)
        # "Henrik" must not count — the whole dimension rests on this.
        assert out["hits"].iloc[0] == 2
        assert [p for _, _, p in out["spans"].iloc[0]] == ["hen", "hens"]

    def test_occupational_report_finds_the_chamber_substitution(self):
        df = _speeches(
            [
                {"text": "En riksdagsman talade.", "year": 2004},
                {"text": "En riksdagsledamot talade.", "year": 2020},
                {"text": "En riksdagsledamot till.", "year": 2020},
            ]
        )
        report = inclusive.occupational_report(df, min_pair_mentions=1)
        pair = next(p for p in report["pairs"] if p["gendered"] == "riksdagsman")
        assert pair["gendered_hits"] == 1
        assert pair["neutral_hits"] == 2
        assert pair["ratio"] == pytest.approx(2 / 3, abs=1e-4)
        assert pair["by_year"][2004]["ratio"] == 0.0
        assert pair["by_year"][2020]["ratio"] == 1.0

    def test_pair_never_mentioned_reports_none_not_zero(self):
        # "No data" and "no substitution" are different claims. A pair nobody
        # ever says must not render as a hard 0% adoption.
        report = inclusive.occupational_report(_speeches([{"text": "Helt orelaterat."}]))
        pair = next(p for p in report["pairs"] if p["gendered"] == "brandman")
        assert pair["ratio"] is None

    def test_null_cases_are_kept_in_the_table_on_purpose(self):
        # sjuksköterska (no settled neutral form) and brandman (neutral form has
        # zero corpus uptake) are tracked to show absence of change where none
        # is claimed. Dropping them would quietly hide the honest null result.
        gendered = set(inclusive.load_pairs()["gendered"])
        assert {"sjuksköterska", "brandman"} <= gendered


class TestRegistry:
    def test_launch_dimensions_are_registered(self):
        launch = {k for k, v in TONE_DIMENSIONS.items() if v.status == "launch"}
        assert {"klasskonflikt", "lix", "hen"} <= launch

    @pytest.mark.parametrize("failed", ["folk", "antielit"])
    def test_dimensions_that_failed_the_gate_are_not_chartable(self, failed):
        # `folk` measures the electorate being referred to, not people-centrism
        # ("skogen har en speciell plats i hjärtat hos svenska folket" — V).
        # `antielit` has a median of 2 hits per non-empty party-year cell and
        # 91 of 192 cells empty — not a time series.
        # Both still exist as measure functions because the census needs them;
        # registering either puts a known-bad line on the public site.
        assert failed not in TONE_DIMENSIONS
        assert hasattr(vi_dom, f"measure_{failed}")

    def test_lix_declares_illustrative_receipts(self):
        # A stylometric dimension has no discrete hit to point at, so a receipt
        # can illustrate the arithmetic but can never be evidence of a hit.
        assert TONE_DIMENSIONS["lix"].receipt_kind == "illustrative"
        assert TONE_DIMENSIONS["klasskonflikt"].receipt_kind == "evidentiary"

    def test_techniques_are_genuinely_mixed(self):
        # The defence is "three different techniques, not one algorithm" — if
        # every launch dimension were lexical, that claim would be false.
        techniques = {v.technique for v in TONE_DIMENSIONS.values() if v.status == "launch"}
        assert {"lexical", "stylometric", "structural"} <= techniques
