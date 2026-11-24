"""Step 14, the check of the tuned configuration in a build of the search service."""

from __future__ import annotations

import importlib.util
import json
import sys
from types import ModuleType

import numpy as np
import pytest

from search.config import REPO_ROOT
from search.index.embed import Answered


def load_step() -> ModuleType:
    """Import step 14 as a module."""
    path = REPO_ROOT / "search" / "pipeline" / "14_check_service.py"
    spec = importlib.util.spec_from_file_location("pipeline_14_check_service", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STEP = load_step()


def test_the_waits_give_the_shares_answered_in_time_and_the_questions_retried():
    answered = [
        Answered(seconds=seconds, attempts=attempts)
        for seconds, attempts in ((0.5, 1), (1.5, 1), (2.5, 2), (12.0, 3))
    ]
    waiting = STEP.summarize_waiting(answered)
    assert waiting.within == {"1": 0.25, "2": 0.5, "3": 0.75, "5": 0.75, "10": 0.75}
    assert waiting.quantiles_s["p50"] == pytest.approx(2.0)
    assert waiting.max_s == 12.0
    assert waiting.retried == 2


def test_the_closeness_is_the_cosine_of_each_pair_of_rows():
    asked = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    evaluated = np.array([[1.0, 0.0], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32)
    closeness = STEP.summarize_closeness(asked, evaluated)
    assert closeness.min == pytest.approx(0.8)
    assert closeness.median == pytest.approx(1.0)


def test_the_agreement_counts_the_passages_of_the_ten_best_the_build_returned():
    exact = {"a": [str(n) for n in range(10)], "b": ["x", "y"], "c": [], "d": ["z"]}
    built = {"a": [str(n) for n in range(9)] + ["99"], "b": ["y", "x"], "c": []}
    agreed = STEP.agreement(built, exact)
    assert (agreed.questions, agreed.missing) == (3, 1)
    assert agreed.recall_10 == pytest.approx((0.9 + 1 + 1) / 3, abs=1e-6)
    assert (agreed.lowest, agreed.below_one) == (0.9, 1)
    assert agreed.identical == 1


def test_the_hits_are_read_by_phase_for_one_run(tmp_path):
    lines = [
        {"run": "s", "phase": "hybrid", "query_id": "q1", "hit_ids": ["p1", "p2"]},
        {"run": "s", "phase": "cold", "query_id": "q1", "hit_ids": ["p9"]},
        {"run": "other", "phase": "hybrid", "query_id": "q2", "hit_ids": ["p3"]},
        {"run": "s", "phase": "vector", "query_id": "q1", "hit_ids": ["p2"]},
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    assert STEP.read_hits(path, "s") == {
        "hybrid": {"q1": ["p1", "p2"]},
        "vector": {"q1": ["p2"]},
        "lexical": {},
    }


def test_a_report_of_the_runner_must_hold_one_finished_run(tmp_path):
    path = tmp_path / "report.json"
    good = {"finished": True, "budget": {"p95_ms": 500}, "runs": [{"name": "s", "failure": None}]}
    path.write_text(json.dumps(good), encoding="utf-8")
    assert STEP.the_run(path)["name"] == "s"
    for bad in (
        {**good, "finished": False},
        {**good, "runs": []},
        {**good, "runs": [{"name": "s", "failure": "killed"}]},
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit):
            STEP.the_run(path)
