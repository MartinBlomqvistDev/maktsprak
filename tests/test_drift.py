"""Tests for src/maktsprak_pipeline/nlp/drift.py (temporal drift analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.maktsprak_pipeline.nlp.drift import (
    _js_divergence,
    add_year,
    frame_trajectories,
    party_divergence_by_year,
    term_trajectories,
    top_movers,
    yearly_signatures,
)


class TestAddYear:
    def test_derives_integer_year_and_drops_unparseable(self):
        df = pd.DataFrame(
            {"protocol_date": ["2019-03-01", "2022-11-15", "not-a-date"], "text": ["a", "b", "c"]}
        )
        out = add_year(df)
        assert out["year"].tolist() == [2019, 2022]
        assert out["year"].dtype.kind == "i"


class TestTopMovers:
    def _corpus(self):
        # "klimat" only appears late; "kärnkraft" only early; "välfärd" is steady.
        rows = []
        for _ in range(30):
            rows.append({"year": 2005, "text": "kärnkraft kärnkraft välfärd"})
            rows.append({"year": 2020, "text": "klimat klimat välfärd"})
        return pd.DataFrame(rows)

    def test_riser_and_faller_are_separated(self):
        risers, fallers = top_movers(
            self._corpus(), split_year=2015, min_count=2, stopwords=frozenset()
        )
        riser_words = [w for w, _ in risers]
        faller_words = [w for w, _ in fallers]
        assert "klimat" in riser_words
        assert "kärnkraft" in faller_words

    def test_steady_word_does_not_top_either_list(self):
        risers, fallers = top_movers(
            self._corpus(), split_year=2015, min_count=2, stopwords=frozenset()
        )
        # "välfärd" is used equally in both eras, so it must not be the #1 mover.
        assert risers[0][0] != "välfärd"
        assert fallers[0][0] != "välfärd"

    def test_default_split_is_median_year(self):
        # Should run without an explicit split_year.
        risers, fallers = top_movers(self._corpus(), min_count=2, stopwords=frozenset())
        assert risers and fallers

    def test_politician_names_filtered_by_default(self):
        # A minister's surname is trivially "distinctive" to their years in
        # office, but it's noise, not a topic — must not survive the default
        # stopword filter.
        rows = []
        for _ in range(30):
            rows.append({"year": 2005, "text": "kärnkraft kärnkraft välfärd"})
            rows.append({"year": 2020, "text": "klimat klimat wallström wallström"})
        df = pd.DataFrame(rows)
        risers, _ = top_movers(df, split_year=2015, min_count=2)
        assert "wallström" not in [w for w, _ in risers]


class TestYearlySignatures:
    def _corpus(self):
        rows = []
        for _ in range(20):
            rows.append({"year": 2019, "text": "brexit brexit förhandling"})
            rows.append({"year": 2020, "text": "pandemin pandemin covid"})
            rows.append({"year": 2022, "text": "ukraina ukraina invasion"})
        return pd.DataFrame(rows)

    def test_each_year_gets_its_own_top_word(self):
        sig = yearly_signatures(self._corpus(), min_count=2, stopwords=frozenset())
        assert sig[2019][0][0] == "brexit"
        assert sig[2020][0][0] in {"pandemin", "covid"}
        assert sig[2022][0][0] == "ukraina"

    def test_sorted_by_year(self):
        sig = yearly_signatures(self._corpus(), min_count=2, stopwords=frozenset())
        assert list(sig.keys()) == [2019, 2020, 2022]

    def test_politician_names_filtered_by_default(self):
        rows = []
        for _ in range(20):
            rows.append({"year": 2019, "text": "brexit brexit hallengren hallengren"})
            rows.append({"year": 2022, "text": "ukraina ukraina invasion"})
        df = pd.DataFrame(rows)
        sig = yearly_signatures(df, min_count=2)
        assert "hallengren" not in [w for w, _ in sig[2019]]


class TestFrameTrajectories:
    def test_frame_hits_matched_as_substring(self):
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "party": ["A", "A"],
                "text": ["gängkriminalitet är ett problem", "vi pratar om annat"],
            }
        )
        frames = {"Brott": ["gäng"]}
        result = frame_trajectories(df, frames=frames)
        # "gängkriminalitet" contains the stem "gäng" and should count as a hit.
        assert result["Brott"]["A"][2020] > 0

    def test_two_parties_can_diverge_on_a_frame(self):
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "party": ["A", "B"],
                "text": ["klimat klimat klimat", "skatt skatt skatt"],
            }
        )
        frames = {"Klimat": ["klimat"]}
        result = frame_trajectories(df, frames=frames)
        assert result["Klimat"]["A"][2020] > result["Klimat"]["B"][2020]
        assert result["Klimat"]["B"][2020] == 0.0

    def test_output_structure_is_frame_group_year(self):
        df = pd.DataFrame({"year": [2020], "party": ["A"], "text": ["klimat"]})
        result = frame_trajectories(df, frames={"Klimat": ["klimat"]})
        assert set(result.keys()) == {"Klimat"}
        assert set(result["Klimat"].keys()) == {"A"}
        assert set(result["Klimat"]["A"].keys()) == {2020}



class TestTermTrajectories:
    def test_relative_frequency_per_year(self):
        df = pd.DataFrame(
            {
                "year": [2018, 2018, 2019],
                "text": ["klimat klimat skog", "klimat skog skog", "skog skog skog"],
            }
        )
        traj = term_trajectories(df, ["klimat"], per=100.0)
        # 2018: 3 "klimat" of 6 tokens = 50 per 100; 2019: 0.
        assert traj[2018]["klimat"] == 50.0
        assert traj[2019]["klimat"] == 0.0

    def test_sorted_by_year(self):
        df = pd.DataFrame({"year": [2020, 2010, 2015], "text": ["a", "b", "c"]})
        traj = term_trajectories(df, ["a"])
        assert list(traj.keys()) == [2010, 2015, 2020]


class TestJensenShannon:
    def test_identical_distributions_zero(self):
        p = np.array([0.5, 0.5])
        assert _js_divergence(p, p) == 0.0

    def test_disjoint_distributions_max(self):
        # Disjoint supports → JS divergence of 1 bit (base-2).
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert abs(_js_divergence(p, q) - 1.0) < 1e-9

    def test_symmetric(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.2, 0.8])
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-12


class TestPartyDivergenceByYear:
    def test_converging_years_have_lower_divergence(self):
        rows = []
        # 2010: parties talk about disjoint topics (high divergence).
        for _ in range(20):
            rows.append({"year": 2010, "party": "A", "text": "skatt skatt skatt"})
            rows.append({"year": 2010, "party": "B", "text": "klimat klimat klimat"})
        # 2020: parties talk about the same topics (low divergence).
        for _ in range(20):
            rows.append({"year": 2020, "party": "A", "text": "skatt klimat"})
            rows.append({"year": 2020, "party": "B", "text": "skatt klimat"})
        df = pd.DataFrame(rows)
        div = party_divergence_by_year(df, min_count=1, stopwords=frozenset())
        assert div[2010] > div[2020]
        assert div[2020] < 1e-9  # identical vocabularies → ~0
