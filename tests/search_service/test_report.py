"""Tests for search_service/spike/report.py."""

from __future__ import annotations

import pytest

from search_service.spike.cgroup import Usage
from search_service.spike.contracts import Timings
from search_service.spike.report import (
    GIB,
    ColdStart,
    ReferenceColdStart,
    RunResult,
    SearchRecord,
    cloud_run_estimates,
    distribution,
    overlap,
    summarize_cold,
    summarize_memory,
    summarize_phase,
    summarize_reference,
    verdict,
)
from search_service.spike.settings import Budget, load_spike_settings

BUDGET = Budget(p95_ms=500, cold_start_s=8, memory_gib=2)


def record(ms: float, *, store="a", phase="hybrid", query="q1", kind="party", hits=("p1",)):
    return SearchRecord(
        run=store,
        phase=phase,
        query_id=query,
        question_set="natural",
        filter_kind=kind,
        client_ms=ms,
        timings=Timings(total_ms=ms - 1, lexical_ms=ms / 2),
        allowed=None,
        hit_ids=hits,
        usage=Usage(),
    )


def cold(total: float) -> ColdStart:
    return ColdStart(
        start_s=0.1,
        ready_s=total - 0.2,
        first_search_ms=200,
        total_s=total,
        process_age_s=None,
        open_s=None,
    )


class TestDistribution:
    def test_percentiles_are_linear_between_neighbours(self):
        summary = distribution([float(v) for v in range(1, 101)])
        assert summary.p50 == pytest.approx(50.5)
        assert summary.p95 == pytest.approx(95.05)
        assert (summary.count, summary.max) == (100, 100)

    def test_nothing_gives_none(self):
        assert distribution([]) is None


class TestPhase:
    def test_a_phase_is_summarised_by_filter_and_set(self):
        phase = summarize_phase("hybrid", [record(10), record(30, kind="period")], truncated=False)
        assert phase.searches == 2
        assert phase.client_ms.mean == 20
        assert set(phase.by_filter) == {"party", "period"}
        assert phase.parts_ms == {"lexical_ms": 10}
        assert phase.latencies_ms == (10.0, 30.0)


class TestVerdict:
    def memory(self, peak: int, kills: int = 0):
        before = Usage(memory_peak=peak // 2, oom_kills=0)
        after = Usage(memory_peak=peak, oom_kills=kills)
        return summarize_memory(before, [], after)

    def test_within_every_bound(self):
        phase = summarize_phase("hybrid", [record(400)] * 20, truncated=False)
        result = verdict(
            BUDGET, phase, summarize_cold([cold(7.9)]), self.memory(GIB), survived=True
        )
        assert result.within_budget

    def test_the_bounds_are_inclusive(self):
        phase = summarize_phase("hybrid", [record(500)] * 20, truncated=False)
        result = verdict(BUDGET, phase, summarize_cold([cold(8.0)]), self.memory(2 * GIB), True)
        assert (result.within_p95, result.within_cold_start, result.within_memory) == (True,) * 3

    def test_the_slowest_cold_start_decides(self):
        phase = summarize_phase("hybrid", [record(100)] * 20, truncated=False)
        result = verdict(BUDGET, phase, summarize_cold([cold(2), cold(9)]), self.memory(GIB), True)
        assert not result.within_cold_start and result.cold_start_s == 9

    def test_a_truncated_phase_is_over_the_budget(self):
        phase = summarize_phase("hybrid", [record(100)] * 20, truncated=True)
        assert not verdict(
            BUDGET, phase, summarize_cold([cold(1)]), self.memory(GIB), True
        ).within_p95

    def test_a_kill_for_memory_is_over_the_budget(self):
        phase = summarize_phase("hybrid", [record(100)] * 20, truncated=False)
        result = verdict(BUDGET, phase, summarize_cold([cold(1)]), self.memory(GIB, kills=1), True)
        assert not result.within_memory and not result.within_budget

    def test_a_container_that_died_is_over_the_budget(self):
        result = verdict(BUDGET, None, summarize_cold([cold(1)]), None, survived=False)
        assert not result.within_budget and result.p95_ms is None


class TestMemory:
    def test_counters_are_counted_over_the_warm_phases(self):
        before = Usage(refaults=10, major_faults=5, throttled=1, throttled_us=1_000_000, anon=50)
        after = Usage(refaults=15, major_faults=5, throttled=4, throttled_us=3_500_000, anon=70)
        summary = summarize_memory(before, [Usage(anon=90)], after)
        assert (summary.refaults, summary.major_faults, summary.throttled_periods) == (5, 0, 3)
        assert summary.throttled_s == 2.5
        assert summary.anon_max_bytes == 90


class TestOverlap:
    def test_the_share_of_the_references_passages_found(self):
        mine = [
            record(1, store="a", hits=("p1", "p2")),
            record(1, store="a", query="q2", hits=("p3",)),
        ]
        theirs = [
            record(1, store="b", hits=("p2", "p9")),
            record(1, store="b", query="q2", hits=("p3",)),
        ]
        result = overlap(mine, theirs, "hybrid")
        assert (result.store, result.reference, result.queries) == ("a", "b", 2)
        assert result.mean_shared_at_10 == pytest.approx(0.75)

    def test_nothing_in_common_gives_none(self):
        assert overlap([record(1, phase="vector")], [record(1, store="b")], "vector") is None


class TestReference:
    def cold(self, total: float) -> ReferenceColdStart:
        return ReferenceColdStart(
            start_s=0.1, ready_s=total - 0.1, first_predict_ms=80, total_s=total
        )

    def test_the_factor_is_cloud_runs_figure_over_the_median_here(self):
        settings = load_spike_settings().reference
        colds = [self.cold(9.0), self.cold(10.0), self.cold(14.0)]
        result = summarize_reference(
            settings,
            colds,
            [40.0, 50.0],
            pinned_cpu="2",
            image_bytes=1,
            cpus_seen=1,
            threads=25,
            warm_throttled_periods=0,
            warm_throttled_s=0.0,
            memory_peak_bytes=GIB,
            failure=None,
        )
        assert (result.median_s, result.max_s) == (10.0, 14.0)
        assert (result.pinned_cpu, result.cpus_seen, result.threads) == ("2", 1, 25)
        assert (result.warm_throttled_periods, result.warm_throttled_s) == (0, 0.0)
        assert result.factor == pytest.approx(settings.cloud_run_cold_s / 10.0, abs=0.01)
        assert result.warm_ms.count == 2

    def test_without_a_cold_start_there_is_no_factor(self):
        settings = load_spike_settings().reference
        result = summarize_reference(
            settings,
            [],
            [],
            image_bytes=None,
            cpus_seen=None,
            memory_peak_bytes=None,
            failure="TimeoutError: no answer",
        )
        assert result.factor is None and result.median_s is None and result.warm_ms is None


class TestCloudRunEstimates:
    def run_with(self, name, slowest):
        cold_starts = summarize_cold([cold(slowest)]) if slowest is not None else None
        return RunResult(
            name=name,
            image="image",
            store=None,
            model=None,
            store_bytes=None,
            prepared=None,
            cold=cold_starts,
            phases=(),
            memory=None,
            failure=None,
            verdict=verdict(BUDGET, None, cold_starts, None, True),
            sweep_verdicts={},
        )

    def reference(self, median, pinned_cpu=None):
        settings = load_spike_settings().reference
        colds = [] if median is None else [ReferenceColdStart(
            start_s=0.1, ready_s=median - 0.1, first_predict_ms=80, total_s=median
        )]  # fmt: skip
        return summarize_reference(
            settings,
            colds,
            [],
            pinned_cpu=pinned_cpu,
            image_bytes=None,
            cpus_seen=None,
            memory_peak_bytes=None,
            failure=None,
        )

    def test_each_run_is_scaled_by_each_factor_and_marked_past_the_budget(self):
        runs = [self.run_with("a", 3.1), self.run_with("b", 1.0), self.run_with("c", None)]
        references = [self.reference(13.0), self.reference(8.0, "2"), self.reference(None)]
        estimates = cloud_run_estimates(BUDGET, runs, references)
        found = {(e.run, e.pinned_cpu): (e.factor, e.scaled_s, e.over_budget) for e in estimates}
        assert found == {
            ("a", None): (2.0, 6.2, False),
            ("b", None): (2.0, 2.0, False),
            ("a", "2"): (3.25, 10.08, True),
            ("b", "2"): (3.25, 3.25, False),
        }

    def test_the_verdict_stays_on_the_cold_start_measured_here(self):
        run = self.run_with("a", 3.1)
        estimates = cloud_run_estimates(BUDGET, [run], [self.reference(8.0)])
        assert estimates[0].over_budget
        assert run.verdict.within_cold_start
