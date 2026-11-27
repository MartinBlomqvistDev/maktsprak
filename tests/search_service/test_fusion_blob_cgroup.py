"""Tests for search_service/spike/fusion.py, blob.py and cgroup.py."""

from __future__ import annotations

import numpy as np
import pytest

from search.index.fuse import reciprocal_rank_fusion
from search_service.spike.blob import StringBlob, write_strings
from search_service.spike.cgroup import parse_flat_keyed, parse_status, read_usage
from search_service.spike.fusion import rrf


class TestRrf:
    def test_scores_are_summed_over_the_rankings(self):
        fused = rrf([[1, 2], [2, 3]], k=60, limit=10)
        scores = dict(fused)
        assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
        assert scores[1] == pytest.approx(1 / 61)
        assert scores[3] == pytest.approx(1 / 62)
        assert [item for item, _ in fused] == [2, 1, 3]

    def test_equal_scores_and_ranks_are_ordered_by_item(self):
        assert [item for item, _ in rrf([[5], [4]], k=60, limit=2)] == [4, 5]

    def test_equal_scores_are_ordered_by_the_better_rank_first(self):
        # At k = 10, rank 1 alone and rank 12 in both legs score exactly 1/11.
        lexical = [7, *range(100, 110), 3]
        vector = [*range(200, 211), 3]
        assert 1 / (10 + 1) == 1 / (10 + 12) + 1 / (10 + 12)
        fused = [item for item, _ in rrf([lexical, vector], k=10, limit=3)]
        assert fused == [7, 200, 3]

    @pytest.mark.parametrize("seed", range(40))
    def test_the_order_is_the_measured_hybrids(self, seed):
        rng = np.random.default_rng(seed)
        items = rng.permutation(60)
        legs = [
            rng.permutation(items)[: rng.integers(5, 40)].tolist(),
            rng.permutation(items)[: rng.integers(5, 40)].tolist(),
        ]
        measured = reciprocal_rank_fusion(legs, k=10.0, limit=10)
        fused = rrf(legs, k=10, limit=10)
        assert [item for item, _ in fused] == [hit.passage for hit in measured]
        assert [score for _, score in fused] == [hit.score for hit in measured]

    def test_the_limit_cuts_the_list(self):
        assert len(rrf([[1, 2, 3]], k=60, limit=2)) == 2

    def test_no_rankings_give_nothing(self):
        assert rrf([], k=60, limit=3) == []


class TestStringBlob:
    def test_strings_come_back_as_written(self, tmp_path):
        strings = ["Kärnkraften ska byggas ut.", "", "Ön och åkern\nnästa rad"]
        assert write_strings(tmp_path / "text", strings) == 3
        blob = StringBlob.open(tmp_path / "text")
        assert len(blob) == 3
        assert [blob[i] for i in range(3)] == strings

    def test_an_index_outside_raises(self, tmp_path):
        write_strings(tmp_path / "text", ["a"])
        with pytest.raises(IndexError):
            StringBlob.open(tmp_path / "text")[1]

    def test_no_strings_at_all(self, tmp_path):
        write_strings(tmp_path / "text", [])
        assert len(StringBlob.open(tmp_path / "text")) == 0

    def test_offsets_that_do_not_fit_the_data_are_refused(self):
        with pytest.raises(ValueError, match="offsets"):
            StringBlob(np.zeros(3, dtype=np.uint8), np.array([0, 5]))


class TestCgroup:
    def test_flat_keyed_files(self):
        assert parse_flat_keyed("anon 10\nfile 20\nbad line here\n") == {"anon": 10, "file": 20}

    def test_status_lines_are_read_in_bytes(self):
        text = "Name:\tpython\nVmRSS:\t   12 kB\nVmHWM:\t 30 kB\n"
        assert parse_status(text) == {"VmRSS": 12 * 1024, "VmHWM": 30 * 1024}

    def test_a_cgroup_is_read(self, tmp_path):
        (tmp_path / "memory.current").write_text("1000\n")
        (tmp_path / "memory.peak").write_text("3000\n")
        (tmp_path / "memory.stat").write_text("anon 400\nfile 600\nworkingset_refault_file 7\n")
        (tmp_path / "memory.events").write_text("oom 0\noom_kill 0\n")
        (tmp_path / "cpu.stat").write_text("usage_usec 99\nnr_throttled 2\nthrottled_usec 50\n")
        status = tmp_path / "status"
        status.write_text("VmRSS:\t1 kB\nVmHWM:\t2 kB\nRssAnon:\t1 kB\nRssFile:\t0 kB\n")
        usage = read_usage(tmp_path, status)
        assert (usage.memory_current, usage.memory_peak) == (1000, 3000)
        assert (usage.anon, usage.file, usage.refaults, usage.oom_kills) == (400, 600, 7, 0)
        assert (usage.cpu_usage_us, usage.throttled, usage.throttled_us) == (99, 2, 50)
        assert (usage.rss, usage.rss_peak) == (1024, 2048)

    def test_without_a_cgroup_every_counter_is_none(self, tmp_path):
        usage = read_usage(tmp_path / "missing", tmp_path / "missing-status")
        assert all(value is None for value in usage.model_dump().values())
