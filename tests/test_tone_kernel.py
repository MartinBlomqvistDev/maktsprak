"""Tests for src/maktsprak_pipeline/nlp/tone/kernel.py (shared tone kernel).

The kernel is the one place the tone statistics live, so every dimension built
on it inherits whatever these tests do or do not pin down.  The z-score test in
particular re-derives Fightin' Words from the paper's formula independently and
asserts the kernel agrees — the wrapper reuses
``distinctiveness.weighted_log_odds``, and this proves the two-token trick it
relies on actually computes the statistic it claims to.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.maktsprak_pipeline.nlp.tone import kernel
from src.maktsprak_pipeline.nlp.tone.kernel import (
    CellStats,
    DimensionSpec,
    aggregate_cells,
    compile_pattern,
    context_for_span,
    fightin_z,
    find_spans,
    hit_density,
    load_pattern_table,
    raw_rate,
    register,
    sample_receipts,
    sentence_spans,
    sentence_split,
    smoothed_rate,
    suppress_thin_cells,
)


class TestSentenceSplit:
    def test_splits_on_terminal_punctuation(self):
        text = "Vi måste agera. Regeringen har svikit! Vad väntar vi på?"
        assert sentence_split(text) == [
            "Vi måste agera.",
            "Regeringen har svikit!",
            "Vad väntar vi på?",
        ]

    @pytest.mark.parametrize("abbr", ["bl.a.", "t.ex.", "m.fl.", "dvs.", "m.m.", "s.k."])
    def test_abbreviation_does_not_end_a_sentence(self, abbr):
        # A naive split-on-period cuts every one of these in half. Riksdag
        # prose is full of them, so a false boundary here would shred receipts.
        text = f"Det gäller {abbr} skolan och vården."
        assert len(sentence_split(text)) == 1

    def test_offsets_index_the_original_text(self):
        # The masking that protects abbreviations must be length-preserving,
        # or every receipt highlight lands on the wrong characters.
        text = "Vi ser bl.a. detta. Sedan kommer nästa mening."
        for start, end in sentence_spans(text):
            assert text[start:end].strip() == text[start:end]
        assert [text[s:e] for s, e in sentence_spans(text)] == sentence_split(text)

    def test_decimal_number_does_not_split(self):
        assert len(sentence_split("Det kostar 3.5 miljarder kronor.")) == 1

    def test_empty_and_whitespace(self):
        assert sentence_split("") == []
        assert sentence_split("   ") == []
        assert sentence_spans("") == []

    def test_text_without_terminal_punctuation_is_one_sentence(self):
        assert sentence_split("Ingen punkt här") == ["Ingen punkt här"]


class TestCompilePattern:
    def test_phrase_does_not_match_inside_a_longer_word(self):
        # This is the live bug in lexicon.py (`str.count` on a raw substring):
        # "ta över" is counted inside "prata över". Regression, not hypothetical.
        pattern = compile_pattern("ta över")
        assert pattern.findall("vi ska prata över detta") == []
        assert len(pattern.findall("de vill ta över makten")) == 1

    def test_swedish_letters_are_word_characters(self):
        # If å/ä/ö were not treated as word chars, "över" would match inside
        # "överens" and every Swedish phrase pattern would over-count.
        assert compile_pattern("över").findall("vi är överens") == []
        assert len(compile_pattern("över").findall("över gränsen")) == 1

    def test_hen_does_not_match_inside_henrik(self):
        # The inclusive-language dimension rests entirely on this.
        assert compile_pattern("hen").findall("Henrik talade") == []
        assert compile_pattern("hen").findall("hens uppfattning") == []
        assert len(compile_pattern("hen").findall("hen sade att")) == 1

    def test_case_insensitive(self):
        assert len(compile_pattern("eliten").findall("Eliten styr")) == 1


class TestLoadPatternTable:
    def _write(self, tmp_path, rows):
        path = tmp_path / "patterns.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_loads_and_compiles(self, tmp_path):
        path = self._write(tmp_path, [{"pattern": "Eliten", "source": "own"}])
        table = load_pattern_table(path, key_col="pattern", required_cols=["pattern", "source"])
        assert table["pattern"].tolist() == ["eliten"]  # case-folded at load
        assert len(table["regex"].iloc[0].findall("eliten styr")) == 1

    def test_duplicate_key_raises(self, tmp_path):
        # The old tone lexicon had 44 words in two categories each, silently
        # counted twice. Fail loudly instead.
        path = self._write(
            tmp_path,
            [{"pattern": "eliten", "source": "own"}, {"pattern": "Eliten", "source": "own"}],
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_pattern_table(path, key_col="pattern", required_cols=["pattern"])

    def test_missing_required_column_raises(self, tmp_path):
        path = self._write(tmp_path, [{"pattern": "eliten"}])
        with pytest.raises(ValueError, match="missing required column"):
            load_pattern_table(path, key_col="pattern", required_cols=["pattern", "source"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pattern_table(tmp_path / "nope.csv", key_col="pattern", required_cols=["pattern"])


class TestFindSpans:
    def test_returns_offsets_that_slice_back_to_the_match(self, tmp_path):
        path = tmp_path / "p.csv"
        pd.DataFrame([{"pattern": "eliten"}, {"pattern": "de rika"}]).to_csv(path, index=False)
        table = load_pattern_table(path, key_col="pattern", required_cols=["pattern"])

        text = "Eliten och de rika bestämmer."
        spans = find_spans(text, table, key_col="pattern")
        assert [(text[s:e].lower(), p) for s, e, p in spans] == [
            ("eliten", "eliten"),
            ("de rika", "de rika"),
        ]


class TestRawRate:
    def test_basic(self):
        assert raw_rate(2, 1000, per=1000.0) == 2.0

    def test_zero_denominator(self):
        assert raw_rate(0, 0) == 0.0


class TestSmoothedRate:
    # A realistic year: 8 parties, ~1 000 sentences each, marker fires at 20
    # per 1 000 across the board.
    BG_HITS, BG_N = 160, 8000

    def test_empty_cell_returns_exactly_the_background_rate(self):
        # Nothing observed -> the estimate is the prior, which is the pooled
        # all-party rate. 160/8000 * 1000 = 20.
        assert smoothed_rate(0, 0, self.BG_HITS, self.BG_N) == pytest.approx(20.0)

    def test_thin_cell_is_pulled_hard_toward_the_background(self):
        # 2 hits in 10 sentences is a raw rate of 200 per 1 000 — a spike that
        # would dominate the chart on the strength of two sentences. The prior
        # (alpha * 8000 = 80 pseudo-sentences) outweighs the 10 real ones.
        raw = raw_rate(2, 10)
        smoothed = smoothed_rate(2, 10, self.BG_HITS, self.BG_N)
        assert raw == pytest.approx(200.0)
        assert smoothed == pytest.approx((2 + 1.6) / (10 + 80) * 1000)  # 40.0
        assert smoothed < raw / 4
        assert abs(smoothed - 20.0) < abs(raw - 20.0)  # closer to background than raw

    def test_well_attested_cell_is_barely_moved(self):
        # One party genuinely uses the marker far more than the rest: 200 hits
        # in its 1 000 sentences, ~10 each for the other seven. The prior is 80
        # pseudo-sentences against 1 000 real ones, so a real signal this well
        # attested survives nearly intact — smoothing must not erase it.
        bg_hits, bg_n = 270, 8000
        raw = raw_rate(200, 1000)
        smoothed = smoothed_rate(200, 1000, bg_hits, bg_n)
        assert raw == pytest.approx(200.0)
        assert smoothed == pytest.approx((200 + 2.7) / (1000 + 80) * 1000, rel=1e-9)
        assert smoothed / raw > 0.9

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError, match="exceeds n"):
            smoothed_rate(11, 10, 100, 1000)
        with pytest.raises(ValueError, match="larger than the background"):
            smoothed_rate(5, 10, 4, 1000)
        with pytest.raises(ValueError, match="Negative"):
            smoothed_rate(-1, 10, 100, 1000)


def _reference_fightin_z(hits: int, n: int, bg_hits: int, bg_n: int, alpha: float = 0.01) -> float:
    """Fightin' Words z, re-derived straight from Monroe et al. (2008) eq. 22.

    Deliberately *not* built on ``weighted_log_odds`` — this is the independent
    check that the kernel's two-token reuse of it computes the right statistic.
    """
    prior_marker = alpha * bg_hits
    prior_other = alpha * (bg_n - bg_hits)

    cell_marker, cell_other = hits, n - hits
    rest_marker, rest_other = bg_hits - hits, (bg_n - n) - (bg_hits - hits)

    log_odds_cell = math.log((cell_marker + prior_marker) / (cell_other + prior_other))
    log_odds_rest = math.log((rest_marker + prior_marker) / (rest_other + prior_other))
    delta = log_odds_cell - log_odds_rest
    variance = 1.0 / (cell_marker + prior_marker) + 1.0 / (rest_marker + prior_marker)
    return delta / math.sqrt(variance)


class TestFightinZ:
    @pytest.mark.parametrize(
        ("hits", "n", "bg_hits", "bg_n"),
        [
            (50, 1000, 200, 10000),  # over-represented
            (2, 1000, 200, 10000),  # under-represented
            (20, 1000, 200, 10000),  # exactly at the background rate
            (0, 500, 200, 10000),  # zero hits in the cell
            (1, 5, 200, 10000),  # tiny cell
        ],
    )
    def test_matches_an_independent_derivation(self, hits, n, bg_hits, bg_n):
        # The kernel routes through distinctiveness.weighted_log_odds with a
        # two-token vocabulary. If that trick is wrong, this fails.
        assert fightin_z(hits, n, bg_hits, bg_n) == pytest.approx(
            _reference_fightin_z(hits, n, bg_hits, bg_n), rel=1e-9
        )

    def test_sign_follows_over_and_under_representation(self):
        over = fightin_z(50, 1000, 200, 10000)  # 5.0% vs 2.0% background
        under = fightin_z(2, 1000, 200, 10000)  # 0.2% vs 2.0% background
        assert over is not None and over > 2
        assert under is not None and under < -2

    def test_at_background_rate_z_is_near_zero(self):
        # Cell rate == pooled rate -> nothing distinguishes it.
        z = fightin_z(20, 1000, 200, 10000)
        assert z is not None and abs(z) < 0.5

    def test_thin_cell_is_not_significant_on_one_hit(self):
        # 1 hit in 5 sentences is a raw rate of 200 per 1 000 against a
        # background of 20 — a 10x "spike" that must not clear significance.
        z = fightin_z(1, 5, 200, 10000)
        assert z is not None and abs(z) < 1.96

    @pytest.mark.parametrize(
        ("hits", "n", "bg_hits", "bg_n", "why"),
        [
            (0, 0, 200, 10000, "empty cell"),
            (200, 10000, 200, 10000, "cell is the whole background — no rest"),
            (0, 500, 0, 10000, "marker never occurs anywhere"),
            (500, 500, 10000, 10000, "marker occurs everywhere"),
        ],
    )
    def test_undefined_comparisons_return_none(self, hits, n, bg_hits, bg_n, why):
        assert fightin_z(hits, n, bg_hits, bg_n) is None, why

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            fightin_z(5, 10, 4, 1000)


class TestAggregateCells:
    def test_pools_counts_before_dividing_never_averages_rates(self):
        # THE regression for this module. Two speeches from one party:
        #   a 1-word interjection that is 100% marker, and a 99-word speech
        #   with none. Averaging per-speech rates gives 500 per 1 000 — a
        #   fabricated spike. Pooling gives 1/100 = 10 per 1 000, the truth.
        df = pd.DataFrame(
            [
                {"party": "A", "year": 2020, "hits": 1, "n": 1},
                {"party": "A", "year": 2020, "hits": 0, "n": 99},
            ]
        )
        cell = aggregate_cells(df)["A"][2020]
        assert cell.rate == pytest.approx(10.0)
        assert cell.rate != pytest.approx(500.0)
        assert (cell.hits, cell.n, cell.speeches) == (1, 100, 2)

    def test_background_is_the_pooled_year_not_the_whole_corpus(self):
        # B's 2021 behaviour must not affect A's 2020 z-score.
        df = pd.DataFrame(
            [
                {"party": "A", "year": 2020, "hits": 50, "n": 1000},
                {"party": "B", "year": 2020, "hits": 10, "n": 1000},
                {"party": "B", "year": 2021, "hits": 900, "n": 1000},
            ]
        )
        cells = aggregate_cells(df)
        expected = fightin_z(50, 1000, 60, 2000)  # 2020 pool only
        assert cells["A"][2020].z == pytest.approx(expected)

    def test_output_is_group_year_sorted(self):
        df = pd.DataFrame(
            [
                {"party": "B", "year": 2021, "hits": 1, "n": 10},
                {"party": "A", "year": 2022, "hits": 1, "n": 10},
                {"party": "A", "year": 2020, "hits": 1, "n": 10},
            ]
        )
        cells = aggregate_cells(df)
        assert list(cells) == ["A", "B"]
        assert list(cells["A"]) == [2020, 2022]
        assert isinstance(cells["A"][2020], CellStats)

    def test_empty_frame(self):
        assert aggregate_cells(pd.DataFrame(columns=["party", "year", "hits", "n"])) == {}


class TestSuppressThinCells:
    def _cells(self):
        return {
            "A": {
                2020: CellStats(1, 10, speeches=2, rate=100.0, smoothed=30.0, z=0.1),
                2021: CellStats(50, 1000, speeches=40, rate=50.0, smoothed=49.0, z=2.5),
            },
            "B": {2020: CellStats(0, 5, speeches=1, rate=0.0, smoothed=20.0, z=None)},
        }

    def test_drops_cells_below_the_speech_floor(self):
        kept = suppress_thin_cells(self._cells(), min_speeches=8)
        assert list(kept) == ["A"]
        assert list(kept["A"]) == [2021]

    def test_drops_cells_below_the_denominator_floor(self):
        # LIX's failure mode: enough speeches, far too few sentences.
        kept = suppress_thin_cells(self._cells(), min_n=50)
        assert list(kept["A"]) == [2021]
        assert "B" not in kept

    def test_boundary_is_inclusive(self):
        cells = {"A": {2020: CellStats(1, 10, speeches=8, rate=1.0, smoothed=1.0, z=None)}}
        assert suppress_thin_cells(cells, min_speeches=8) == cells
        assert suppress_thin_cells(cells, min_speeches=9) == {}

    def test_no_floors_keeps_everything(self):
        assert suppress_thin_cells(self._cells()) == self._cells()


class TestHitDensity:
    def test_flags_the_antielit_failure_mode(self):
        # The real numbers that got antielit unregistered: a dimension can pass
        # every speech-count floor and still have no evidence in it. 4 cells,
        # 2 empty, the others holding 2 hits each.
        cells = {
            "A": {
                2020: CellStats(2, 5000, speeches=300, rate=0.4, smoothed=0.4, z=1.0),
                2021: CellStats(0, 5000, speeches=300, rate=0.0, smoothed=0.2, z=None),
            },
            "B": {
                2020: CellStats(2, 5000, speeches=300, rate=0.4, smoothed=0.4, z=1.0),
                2021: CellStats(0, 5000, speeches=300, rate=0.0, smoothed=0.2, z=None),
            },
        }
        density = hit_density(cells)
        assert density["median_hits"] == 2
        assert density["empty_share"] == 0.5
        assert density["total_hits"] == 4

    def test_leading_zeros_are_data_not_absence(self):
        # `hen`'s real shape: nothing 2002-2011 because the word had not entered
        # Swedish yet, then a healthy curve. The zeros ARE the finding. Counting
        # them as missing evidence flagged a perfectly good adoption curve as
        # unchartable, which is how this distinction got written.
        years = dict.fromkeys(range(2002, 2012), 0) | {y: 20 for y in range(2012, 2027)}
        cells = {
            "Riksdagen": {
                y: CellStats(h, 300_000, speeches=3000, rate=6.6, smoothed=6.6, z=None)
                for y, h in years.items()
            }
        }
        density = hit_density(cells)
        assert density["median_hits"] == 20
        assert density["leading_empty"] == 10  # informative, not a defect
        assert density["empty_share"] == 0.0  # no holes once it appears

    def test_a_hole_after_the_phenomenon_appears_still_counts(self):
        cells = {
            "A": {
                2020: CellStats(10, 5000, speeches=300, rate=2.0, smoothed=2.0, z=1.0),
                2021: CellStats(0, 5000, speeches=300, rate=0.0, smoothed=1.0, z=None),
            }
        }
        assert hit_density(cells)["empty_share"] == 0.5

    def test_a_healthy_dimension_passes(self):
        cells = {
            "A": {
                y: CellStats(120, 5000, speeches=300, rate=24.0, smoothed=24.0, z=2.0)
                for y in (2020, 2021)
            }
        }
        density = hit_density(cells)
        assert density["median_hits"] == 120
        assert density["empty_share"] == 0.0

    def test_empty_input(self):
        assert hit_density({})["empty_share"] == 1.0


class TestContextForSpan:
    def test_highlight_offsets_slice_back_to_the_match(self):
        # The invariant every receipt depends on: context[hl0:hl1] IS the match.
        text = "Första meningen. Andra meningen har eliten i sig. Tredje meningen."
        start = text.index("eliten")
        context, (h0, h1) = context_for_span(text, start, start + len("eliten"))
        assert context[h0:h1] == "eliten"

    def test_window_includes_neighbouring_sentences(self):
        text = "Ett. Två med eliten. Tre."
        start = text.index("eliten")
        context, _ = context_for_span(text, start, start + 6, window=1)
        assert context.startswith("Ett.")
        assert context.endswith("Tre.")

    def test_window_zero_is_the_sentence_itself(self):
        text = "Ett. Två med eliten. Tre."
        start = text.index("eliten")
        context, (h0, h1) = context_for_span(text, start, start + 6, window=0)
        assert context == "Två med eliten."
        assert context[h0:h1] == "eliten"


class TestSampleReceipts:
    def _df(self, n_spans: int = 10):
        rows = []
        for i in range(n_spans):
            text = f"Inledning nummer {i}. Här nämns eliten tydligt. Avslutning."
            start = text.index("eliten")
            rows.append(
                {
                    "id": f"s{i}",
                    "protocol_id": f"prot{i}",
                    "party": "SD",
                    "year": 2022,
                    "speaker": "NÅGON TALARE",
                    "protocol_date": "2022-05-01",
                    "file_url": f"http://x/{i}.pdf",
                    "text": text,
                    "spans": [(start, start + len("eliten"), "eliten")],
                }
            )
        return pd.DataFrame(rows)

    def test_is_deterministic_under_a_fixed_seed(self):
        # The public site shows these quotes. A refactor must not silently
        # change which ones — that is why the seed is pinned in a test.
        first = sample_receipts(self._df(), dimension="vi_dom")
        second = sample_receipts(self._df(), dimension="vi_dom")
        assert [m.speech_id for m in first["SD"][2022]] == [m.speech_id for m in second["SD"][2022]]

    def test_different_seed_selects_differently(self):
        a = sample_receipts(self._df(20), dimension="vi_dom", seed=13)
        b = sample_receipts(self._df(20), dimension="vi_dom", seed=99)
        assert [m.speech_id for m in a["SD"][2022]] != [m.speech_id for m in b["SD"][2022]]

    def test_caps_at_per_cell(self):
        receipts = sample_receipts(self._df(10), dimension="vi_dom", per_cell=3)
        assert len(receipts["SD"][2022]) == 3

    def test_keeps_all_when_fewer_than_the_cap(self):
        receipts = sample_receipts(self._df(2), dimension="vi_dom", per_cell=3)
        assert len(receipts["SD"][2022]) == 2

    def test_evidentiary_receipt_highlights_the_real_span(self):
        marker = sample_receipts(self._df(1), dimension="vi_dom")["SD"][2022][0]
        assert marker.kind == "evidentiary"
        assert marker.hl is not None
        assert marker.context[marker.hl[0] : marker.hl[1]] == "eliten"
        assert marker.protocol_id == "prot0"
        assert marker.file_url == "http://x/0.pdf"

    def test_illustrative_receipt_carries_no_highlight(self):
        # A stylometric dimension has no discrete hit to point at, so it must
        # not ship a highlight that would imply one.
        marker = sample_receipts(self._df(1), dimension="lix", kind="illustrative")["SD"][2022][0]
        assert marker.kind == "illustrative"
        assert marker.hl is None

    def test_subtype_is_stamped_from_the_pattern_map(self):
        receipts = sample_receipts(self._df(1), dimension="vi_dom", subtype_of={"eliten": "elite"})
        assert receipts["SD"][2022][0].subtype == "elite"

    def test_empty_input(self):
        assert sample_receipts(pd.DataFrame(), dimension="vi_dom") == {}


class TestDimensionSpec:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        saved = dict(kernel.TONE_DIMENSIONS)
        kernel.TONE_DIMENSIONS.clear()
        yield
        kernel.TONE_DIMENSIONS.clear()
        kernel.TONE_DIMENSIONS.update(saved)

    def _spec(self, **kwargs):
        defaults = dict(
            id="dummy",
            label_sv="Dummy",
            unit_sv="per 1 000 ord",
            technique="structural",
            measure_fn=lambda df: df,
        )
        return DimensionSpec(**{**defaults, **kwargs})

    def test_lexical_dimension_without_patterns_is_rejected(self):
        # It would score zero forever and look like a real flat line.
        with pytest.raises(ValueError, match="no pattern_paths"):
            self._spec(technique="lexical")

    def test_stylometric_dimension_cannot_claim_evidentiary_receipts(self):
        with pytest.raises(ValueError, match="illustrative"):
            self._spec(technique="stylometric", receipt_kind="evidentiary")

    def test_register_rejects_a_duplicate_id(self):
        register(self._spec())
        with pytest.raises(ValueError, match="already registered"):
            register(self._spec())

    def test_register_populates_the_registry(self):
        spec = register(self._spec(id="tone_x"))
        assert kernel.TONE_DIMENSIONS["tone_x"] is spec
