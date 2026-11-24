"""Tests for search_service/spike/run.py, without Docker."""

from __future__ import annotations

import json

import pytest

from search_service.spike import run
from search_service.spike.cgroup import Usage
from search_service.spike.contracts import SearchFilter, Timings
from search_service.spike.query_model import Preparation, PreparedModel
from search_service.spike.report import (
    ColdStart,
    ModelLabel,
    ReferenceColdStart,
    RunResult,
    SearchRecord,
    SpikeReport,
    Verdict,
    summarize_cold,
    summarize_reference,
)
from search_service.spike.settings import QueryModelSettings, QueryModelSpec
from search_service.spike.workload import Workload, WorkloadQuery
from tests.search_service.spike_kit import kit_settings


def fake_docker(monkeypatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(run, "docker", lambda *args, **_: calls.append(args) or "id")
    return calls


def pairs_of(command: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(command[1::2], command[2::2], strict=False))


class TestRun:
    def test_the_build_context_holds_the_code_and_the_dockerfile(self, tmp_path):
        context = run.stage_context(tmp_path / "context")
        assert (context / "Dockerfile").is_file()
        assert (context / "search" / "index" / "lexical.py").is_file()
        assert (context / "search_service" / "spike" / "server.py").is_file()
        assert (context / "search_service" / "spike" / "requirements-torch.txt").is_file()
        assert not list(context.rglob("__pycache__"))

    def test_both_images_are_targets_of_the_one_dockerfile(self, monkeypatch):
        calls = fake_docker(monkeypatch)
        settings = kit_settings()
        run.build_images(settings)
        targets = {(call[2], call[4]) for call in calls}
        assert targets == {
            ("slim", settings.runs.images.slim),
            ("full", settings.runs.images.full),
        }

    def test_a_measured_container_has_the_limits_and_no_swap(self, monkeypatch):
        calls = fake_docker(monkeypatch)
        settings = kit_settings()
        subject = run.Subject(store=settings.store("numpy-8-float32"), model=None)
        assert run.create_container(settings, subject) == "id"
        command = calls[0]
        pairs = pairs_of(command)
        assert pairs["--cpus"] == "1"
        assert pairs["--memory"] == pairs["--memory-swap"] == "2g"
        assert "SPIKE_STORE=numpy-8-float32" in command
        assert not any(part.startswith("SPIKE_MODEL=") for part in command)
        assert "LANCE_CPU_THREADS=1" in command and "OPENBLAS_NUM_THREADS=1" in command
        assert command[-1] == settings.runs.images.slim

    def test_a_request_carries_the_query_its_filter_and_its_mode(self):
        query = WorkloadQuery(
            query_id="natural-0001",
            question_set="natural",
            text="Ska det byggas ny kärnkraft?",
            filter=SearchFilter(party="KD"),
            filter_kind="party",
            vector_seed=9,
        )
        request = run.search_request(query, "vector")
        assert (request.text, request.vector_seed, request.mode) == (query.text, 9, "vector")
        assert request.filter.party == "KD"

    def test_an_engine_inside_wsl_names_windows_paths_its_own_way(self):
        cli = run.DockerCli(command=("wsl", "--", "docker"), wsl=True)
        assert cli.path(r"C:\Work\Corpus\data\search") == "/mnt/c/Work/Corpus/data/search"
        assert run.DockerCli().path("/data/search") == "/data/search"

    @pytest.mark.parametrize(
        ("runtime", "precision", "image"),
        [("torch", "bf16", "full"), ("onnx", "int8", "slim")],
    )
    def test_a_container_with_a_model_names_it_and_runs_its_runtimes_image(
        self, monkeypatch, runtime, precision, image
    ):
        calls = fake_docker(monkeypatch)
        settings = kit_settings()
        model = ModelLabel(name="bekko-a25m", runtime=runtime, precision=precision, dim=384)
        run.create_container(settings, run.Subject(store=None, model=model))
        command = calls[0]
        assert "SPIKE_MODEL=bekko-a25m" in command
        assert f"SPIKE_RUNTIME={runtime}" in command
        assert f"SPIKE_PRECISION={precision}" in command
        assert not any(part.startswith("SPIKE_STORE=") for part in command)
        assert command[-1] == getattr(settings.runs.images, image)

    def test_a_run_is_named_after_what_it_serves(self):
        settings = kit_settings()
        store = settings.store("numpy-8-int8")
        model = ModelLabel(name="m", runtime="onnx", precision="fp32", dim=8)
        assert run.Subject(store=store, model=None).name == "numpy-8-int8"
        assert run.Subject(store=None, model=model).name == "m-onnx-fp32"
        assert run.Subject(store=store, model=model).name == "numpy-8-int8+m-onnx-fp32"

    def test_the_models_are_checked_on_the_warm_up_then_the_measured_questions(self):
        def query(number: int, text: str) -> WorkloadQuery:
            return WorkloadQuery(
                query_id=f"q{number}",
                question_set="natural",
                text=text,
                filter=SearchFilter(party="S"),
                filter_kind="party",
                vector_seed=number,
            )

        workload = Workload(
            warmup=(query(1, "a"), query(2, "b")),
            queries=(query(3, "b"), query(4, "c"), query(5, "d")),
            party_view=(),
        )
        assert run.check_texts(workload, 3) == ["a", "b", "c"]
        assert run.check_texts(workload, 10) == ["a", "b", "c", "d"]


class TestReference:
    def test_its_container_has_the_limits_and_no_thread_settings(self, monkeypatch):
        calls = fake_docker(monkeypatch)
        settings = kit_settings()
        run.reference_container(settings)
        command = calls[0]
        pairs = pairs_of(command)
        assert pairs["--cpus"] == "1"
        assert pairs["--memory"] == pairs["--memory-swap"] == "2g"
        assert pairs["-p"] == f"127.0.0.1:{settings.runs.port}:{settings.reference.port}"
        assert "-e" not in command
        assert "--cpuset-cpus" not in command
        assert command[-1] == settings.reference.image

    def test_its_second_container_is_pinned_to_one_cpu_within_the_same_limits(self, monkeypatch):
        calls = fake_docker(monkeypatch)
        settings = kit_settings()
        run.reference_container(settings, "2")
        pairs = pairs_of(calls[0])
        assert pairs["--cpuset-cpus"] == "2"
        assert pairs["--cpus"] == "1"
        assert pairs["--memory"] == pairs["--memory-swap"] == "2g"

    def test_it_runs_under_the_limits_alone_then_pinned(self):
        reference = kit_settings().reference
        assert run.reference_conditions(reference) == [None, reference.pinned_cpu]
        alone = reference.model_copy(update={"pinned_cpu": None})
        assert run.reference_conditions(alone) == [None]

    def test_the_quotas_throttling_is_read_from_the_cgroup(self, monkeypatch):
        stat = "usage_usec 900\nnr_periods 40\nnr_throttled 12\nthrottled_usec 345678\n"
        monkeypatch.setattr(run, "docker", lambda *args, **_: stat)
        assert run.container_throttling("id") == (12, 345678)
        monkeypatch.setattr(run, "docker", lambda *args, **_: "")
        assert run.container_throttling("id") == (None, None)

    def test_the_busiest_process_gives_the_thread_count(self, monkeypatch):
        status = "Name:\tsh\nThreads:\t1\nName:\tuvicorn\nThreads:\t25\nVmRSS:\t10 kB\n"
        monkeypatch.setattr(run, "docker", lambda *args, **_: status)
        assert run.most_threads("id") == 25
        monkeypatch.setattr(run, "docker", lambda *args, **_: "")
        assert run.most_threads("id") is None

    def test_its_context_holds_the_service_and_the_cached_classifier(self, tmp_path):
        settings = kit_settings()
        reference = settings.reference
        snapshot = run.hub_snapshot(tmp_path / "hub", reference.model, reference.revision)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        context = run.stage_reference_context(tmp_path / "context", settings, tmp_path / "hub")
        assert (context / run.REFERENCE_MODEL_DIR / "config.json").is_file()
        assert (context / "inference_service" / "app.py").is_file()
        assert (context / "inference_service" / "requirements.txt").is_file()
        assert "torch==2.5.1" in (context / "Dockerfile").read_text(encoding="utf-8")

    def test_without_the_classifier_in_the_cache_there_is_no_context(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="holds no"):
            run.stage_reference_context(tmp_path / "context", kit_settings(), tmp_path / "hub")

    def test_the_hub_cache_names_a_snapshot_by_model_and_revision(self, tmp_path):
        path = run.hub_snapshot(tmp_path, "MartinBlomqvist/maktsprak_classifier_clean", "ab" * 20)
        assert path.parts[-3:] == (
            "models--MartinBlomqvist--maktsprak_classifier_clean",
            "snapshots",
            "ab" * 20,
        )


def prepared_model(name: str, runtime: str, precision: str, dim: int) -> PreparedModel:
    return PreparedModel(
        name=name,
        model=f"a/{name}",
        revision="0" * 40,
        runtime=runtime,
        precision=precision,
        dim=dim,
        max_tokens=16,
        query_prompt="",
        bytes=1,
    )


class TestPreparations:
    def test_prepared_and_unprepared_variants_are_told_apart(self):
        entries = [
            Preparation(
                label="small-onnx-fp32",
                prepared=prepared_model("small", "onnx", "fp32", 8),
                error=None,
            ),
            Preparation(label="small-onnx-int8", prepared=None, error="RuntimeError: no"),
        ]
        text = json.dumps([entry.model_dump(mode="json") for entry in entries])
        prepared, unprepared = run.read_preparations(text)
        assert list(prepared) == ["small-onnx-fp32"]
        assert unprepared == {"small-onnx-int8": "RuntimeError: no"}

    def test_no_file_means_nothing_prepared(self):
        assert run.read_preparations("") == ({}, {})


def result(name: str, within: bool, store=None, model=None) -> RunResult:
    verdict = Verdict(
        p95_ms=100,
        cold_start_s=1,
        memory_peak_gib=1,
        survived=True,
        within_p95=within,
        within_cold_start=True,
        within_memory=True,
        within_budget=within,
    )
    return RunResult(
        name=name,
        image="image",
        store=store,
        model=model,
        store_bytes=None,
        prepared=None,
        cold=None,
        phases=(),
        memory=None,
        failure=None,
        verdict=verdict,
        sweep_verdicts={},
    )


def model_settings():
    def spec(name: str, dim: int) -> QueryModelSpec:
        return QueryModelSpec(
            name=name, model=f"a/{name}", revision="0" * 40, dim=dim, query_prompt="", max_tokens=16
        )

    models = kit_settings().query_models.model_copy(
        update={"models": (spec("small", 8), spec("wide", 1024))}
    )
    assert isinstance(models, QueryModelSettings)
    return kit_settings().model_copy(update={"query_models": models})


class TestPairs:
    def test_only_pairs_within_the_budget_alone_are_measured(self):
        settings = model_settings()
        stores = [
            result(spec.name, within=spec.name != "numpy-8-int8", store=spec)
            for spec in settings.stores
        ]
        models = [result("small-onnx-fp32", True), result("small-torch-bf16", False)]
        prepared = {
            "small-onnx-fp32": prepared_model("small", "onnx", "fp32", 8),
            "small-torch-bf16": prepared_model("small", "torch", "bf16", 8),
        }
        pairs, skipped = run.plan_pairs(settings, stores, models, prepared)
        names = {pair.name for pair in pairs}
        assert "numpy-8-float32+small-onnx-fp32" in names
        assert "numpy-8-int8+small-onnx-fp32" not in names
        assert not any(name.endswith("small-torch-bf16") for name in names)
        reasons = {(s.store, s.model): s.reason for s in skipped}
        assert reasons[("numpy-8-int8", "small-onnx-fp32")] == "the store alone is over the budget"
        assert (
            reasons[("numpy-8-float32", "small-torch-bf16")] == "the model alone is over the budget"
        )
        assert reasons[("numpy-8-float32", "small-torch-fp32")] == "the model was not prepared"

    def test_a_model_of_another_dimension_is_never_paired(self):
        settings = model_settings()
        stores = [result(spec.name, True, store=spec) for spec in settings.stores]
        pairs, skipped = run.plan_pairs(settings, stores, [], {})
        assert pairs == []
        assert not any(s.model.startswith("wide") for s in skipped)
        assert {s.reason for s in skipped} == {"the model was not prepared"}

    def test_only_prepared_variants_are_measured_alone(self):
        settings = model_settings()
        prepared = {"small-onnx-int8": prepared_model("small", "onnx", "int8", 8)}
        labels = run.model_labels(settings, prepared)
        assert [label.label for label in labels] == ["small-onnx-int8"]


def search_record(run_name: str, query: str) -> SearchRecord:
    return SearchRecord(
        run=run_name,
        phase="hybrid",
        query_id=query,
        question_set="natural",
        filter_kind="party",
        client_ms=12.5,
        timings=Timings(total_ms=11.0),
        allowed=None,
        hit_ids=("p1", "p2"),
        usage=Usage(throttled=0),
    )


def report_of(runs, references=()) -> SpikeReport:
    settings = kit_settings()
    return SpikeReport(
        created="2026-11-22T10:00:00+00:00",
        finished=False,
        budget=settings.budget,
        container=settings.container,
        search=settings.search,
        vectors=settings.vectors,
        workload={},
        environment={},
        images={settings.runs.images.slim: None},
        runs=tuple(runs),
        unprepared={},
        skipped=(),
        overlaps=(),
        references=tuple(references),
    )


class TestResume:
    def test_only_runs_that_ended_without_a_failure_are_kept(self):
        settings = kit_settings()
        done = result("numpy-8-float32", True, store=settings.store("numpy-8-float32"))
        failed = result("numpy-8-int8", True).model_copy(update={"failure": "TimeoutError: no"})
        kept = run.finished_runs(report_of([done, failed]))
        assert list(kept) == ["numpy-8-float32"]
        assert run.finished_runs(None) == {}

    def test_each_condition_of_the_reference_is_kept_by_its_cpu(self):
        reference = kit_settings().reference
        quota = summarize_reference(
            reference, [], [], image_bytes=None, cpus_seen=22, memory_peak_bytes=None, failure=None
        )
        pinned = summarize_reference(
            reference,
            [],
            [],
            pinned_cpu="2",
            image_bytes=None,
            cpus_seen=1,
            memory_peak_bytes=None,
            failure="TimeoutError: no",
        )
        kept = run.finished_references(report_of([], [quota, pinned]))
        assert list(kept) == [None]

    def test_a_report_survives_being_written_and_read_back(self, tmp_path):
        settings = kit_settings()
        label = ModelLabel(name="small", runtime="onnx", precision="int8", dim=8)
        report = report_of(
            [result("numpy-8-float32", True, store=settings.store("numpy-8-float32"), model=label)]
        )
        path = tmp_path / "storage_spike.json"
        run.write_atomically(report, path)
        assert run.read_report(path) == report
        assert [p.name for p in tmp_path.iterdir()] == ["storage_spike.json"]
        assert run.read_report(tmp_path / "none.json") is None

    def test_only_the_records_of_the_kept_runs_are_read_back(self, tmp_path):
        path = tmp_path / "searches.jsonl"
        lines = [
            search_record(name, query) for name, query in [("a", "q1"), ("b", "q1"), ("a", "q2")]
        ]
        path.write_text("".join(line.model_dump_json() + "\n" for line in lines), encoding="utf-8")
        kept = run.kept_records(path, {"a"})
        assert list(kept) == ["a"]
        assert [record.query_id for record in kept["a"]] == ["q1", "q2"]
        assert run.kept_records(tmp_path / "none.jsonl", {"a"}) == {}

    def test_a_rewrite_adds_the_estimates_and_measures_nothing(self, tmp_path, monkeypatch):
        calls = fake_docker(monkeypatch)
        reference = summarize_reference(
            kit_settings().reference,
            [ReferenceColdStart(start_s=0.1, ready_s=9.0, first_predict_ms=80, total_s=10.0)],
            [],
            image_bytes=None,
            cpus_seen=22,
            memory_peak_bytes=None,
            failure=None,
        )
        slow = ColdStart(
            start_s=0.1, ready_s=3.0, first_search_ms=50, total_s=4.0,
            process_age_s=None, open_s=None,
        )  # fmt: skip
        measured = result("numpy-8-float32", True).model_copy(
            update={"cold": summarize_cold([slow])}
        )
        path = tmp_path / "storage_spike.json"
        run.write_atomically(report_of([measured], [reference]), path)
        assert run.rewrite_report(path) == 0
        rewritten = run.read_report(path)
        assert [(e.run, e.scaled_s, e.over_budget) for e in rewritten.cloud_run_estimates] == [
            ("numpy-8-float32", 10.4, True)
        ]
        assert rewritten.runs == (measured,)
        assert calls == []
        assert run.rewrite_report(tmp_path / "none.json") == 1

    def test_a_run_measured_again_is_set_aside_with_its_first_container(self):
        settings = kit_settings()
        first = result("lance-8-float32", True, store=settings.store("lance-8-float32"))
        other = result("numpy-8-float32", True, store=settings.store("numpy-8-float32"))
        previous = report_of([first, other])
        kept, superseded = run.split_kept(previous, ["lance-8-float32"], "IO threads unset")
        assert list(kept) == ["numpy-8-float32"]
        assert [(s.run.name, s.reason) for s in superseded] == [
            ("lance-8-float32", "IO threads unset")
        ]
        assert superseded[0].container == previous.container
        again = previous.model_copy(update={"superseded": tuple(superseded)})
        assert run.split_kept(again, [], None)[1] == superseded

    def test_measuring_again_needs_a_reason_and_a_run_the_report_holds(self):
        previous = report_of([result("numpy-8-float32", True)])
        with pytest.raises(ValueError, match="needs a reason"):
            run.split_kept(previous, ["numpy-8-float32"], None)
        with pytest.raises(ValueError, match="holds no run"):
            run.split_kept(previous, ["lance-8-float32"], "why")
        assert run.split_kept(None, [], None) == ({}, [])

    def test_the_command_line_refuses_to_measure_again_without_resuming(self):
        with pytest.raises(SystemExit):
            run.parse_args(["--remeasure", "lance-8-float32", "--reason", "why"])
        with pytest.raises(SystemExit):
            run.parse_args(["--resume", "--remeasure", "lance-8-float32"])
        args = run.parse_args(["--resume", "--remeasure", "lance-8-float32", "--reason", "why"])
        assert (args.remeasure, args.reason) == (["lance-8-float32"], "why")


class TestMemory:
    def test_a_memory_limit_sets_the_containers_and_the_budget(self, monkeypatch):
        settings = run.with_memory(kit_settings(), "4g")
        assert (settings.container.memory, settings.budget.memory_gib) == ("4g", 4.0)
        calls = fake_docker(monkeypatch)
        run.create_container(
            settings, run.Subject(store=settings.store("numpy-8-float32"), model=None)
        )
        pairs = pairs_of(calls[0])
        assert pairs["--memory"] == pairs["--memory-swap"] == "4g"

    @pytest.mark.parametrize("memory", ["4", "4096m", "0g", "g", "2.5g"])
    def test_only_whole_gib_are_taken(self, memory):
        with pytest.raises(ValueError, match="whole number of GiB"):
            run.with_memory(kit_settings(), memory)

    def test_a_run_with_another_limit_writes_beside_the_main_report(self):
        assert run.output_names(None) == (run.REPORT_NAME, run.SEARCHES_NAME)
        assert run.output_names("4g") == (
            "storage_spike_memory_4g.json",
            "searches_memory_4g.jsonl",
        )


class TestSettingsFile:
    def test_a_settings_file_names_the_report_and_the_records(self):
        assert run.output_names(None) == ("storage_spike.json", "searches.jsonl")
        assert run.output_names(None, "service_recall.yaml") == (
            "service_recall.json",
            "service_recall.jsonl",
        )
        assert run.output_names("4g", "service_recall.yaml") == (
            "service_recall_memory_4g.json",
            "service_recall_memory_4g.jsonl",
        )

    def test_a_container_reads_the_settings_file_the_run_reads(self, monkeypatch):
        monkeypatch.delenv("SPIKE_SETTINGS", raising=False)
        assert run.settings_env() == []
        monkeypatch.setenv("SPIKE_SETTINGS", "service_recall.yaml")
        assert run.settings_env() == ["-e", "SPIKE_SETTINGS=service_recall.yaml"]
