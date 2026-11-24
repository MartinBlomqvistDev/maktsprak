"""Tests for search_service/spike/settings.py and spike.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from search_service.spike.settings import (
    SETTINGS_ENV,
    SPIKE_FILE,
    AnnParams,
    ProbeSettings,
    QueryModelSettings,
    ReferenceSettings,
    StoreSpec,
    load_spike_settings,
    settings_file,
    variant_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def settings():
    return load_spike_settings()


class TestSpikeYaml:
    def test_the_budget_is_the_pre_registered_storage_rule(self, settings):
        assert settings.budget.p95_ms == 500
        assert settings.budget.cold_start_s == 8
        assert settings.budget.memory_gib == 2

    def test_the_container_has_one_cpu_and_2_gib(self, settings):
        assert settings.container.cpus == "1"
        assert settings.container.memory == "2g"

    def test_every_compute_pool_is_pinned_to_one_cpu(self, settings):
        assert set(settings.container.threads.values()) == {"1"}
        assert {"OPENBLAS_NUM_THREADS", "LANCE_CPU_THREADS"} <= set(settings.container.threads)

    def test_lancedbs_io_threads_keep_their_default(self, settings):
        assert "LANCE_IO_THREADS" not in settings.container.threads

    def test_both_candidate_dimensions_and_every_format_are_measured(self, settings):
        numpy = {(s.dim, s.format) for s in settings.stores if s.kind == "numpy"}
        lance = {
            (s.dim, s.format) for s in settings.stores if s.kind == "lance" and s.index == "none"
        }
        dims = (384, 1024)
        assert numpy == {(d, f) for d in dims for f in ("float32", "float16", "int8")}
        assert lance == {(d, f) for d in dims for f in ("float32", "float16")}

    def test_each_approximate_index_is_measured_in_both_dimensions(self, settings):
        indexed = {(s.index, s.dim) for s in settings.stores if s.index != "none"}
        assert indexed == {(i, d) for i in ("IVF_PQ", "IVF_HNSW_SQ") for d in (384, 1024)}

    def test_the_full_text_index_matches_the_tuned_analyzer(self, settings):
        fts = settings.lance.fts
        assert (fts.language, fts.stem, fts.remove_stop_words) == ("Swedish", True, False)
        assert fts.ascii_folding is False
        assert fts.max_token_length is None

    def test_a_store_is_found_by_name(self, settings):
        assert settings.store("numpy-384-float32").dim == 384
        with pytest.raises(KeyError):
            settings.store("no-such-store")


class TestStoreSpec:
    def test_a_numpy_store_has_no_index(self):
        with pytest.raises(ValidationError, match="exactly"):
            StoreSpec(name="x", kind="numpy", dim=8, format="float32", index="IVF_PQ")

    def test_a_lance_store_has_no_int8_vectors(self):
        with pytest.raises(ValidationError, match="float16"):
            StoreSpec(name="x", kind="lance", dim=8, format="int8")

    def test_only_an_approximate_index_is_swept(self):
        with pytest.raises(ValidationError, match="sweep"):
            StoreSpec(name="x", kind="lance", dim=8, format="float32", sweep=(AnnParams(ef=10),))

    def test_names_must_be_unique(self, settings):
        twice = settings.stores[:1] * 2
        with pytest.raises(ValidationError, match="share a name"):
            type(settings).model_validate({**settings.model_dump(), "stores": twice})

    def test_an_hnsw_sweep_reaches_the_depth(self, settings):
        shallow = StoreSpec(
            name="x", kind="lance", dim=8, format="float32", index="IVF_HNSW_SQ",
            sweep=(AnnParams(ef=settings.search.depth - 1),),
        )  # fmt: skip
        data = {**settings.model_dump(), "stores": (shallow.model_dump(),)}
        with pytest.raises(ValidationError, match="below the depth"):
            type(settings).model_validate(data)

    def test_each_leg_ranks_as_deep_as_the_hybrid_is_tuned(self, settings):
        assert settings.search.depth == 1000


class TestAnnParams:
    def test_the_label_names_what_is_set(self):
        assert AnnParams().label == "default"
        assert AnnParams(nprobes=20, refine_factor=10).label == "nprobes=20,refine_factor=10"


class TestProbeSettings:
    def test_a_grid_runs_from_first_to_last(self):
        assert ProbeSettings.values((0.5, 1.0, 0.25)) == [0.5, 0.75, 1.0]

    def test_the_grids_hold_lucenes_defaults(self, settings):
        assert 1.2 in ProbeSettings.values(settings.probe.k1_grid)
        assert 0.75 in ProbeSettings.values(settings.probe.b_grid)
        assert 0.9 in ProbeSettings.values(settings.probe.b_grid)


class TestQueryModels:
    def test_both_runtimes_are_measured_in_both_their_precisions(self, settings):
        assert settings.query_models.variants() == [
            ("torch", "fp32"),
            ("torch", "bf16"),
            ("onnx", "fp32"),
            ("onnx", "int8"),
        ]

    def test_onnx_runtime_gets_the_one_cpu(self, settings):
        assert settings.query_models.onnx.threads == 1

    def test_the_candidates_are_phase_3s_local_ones(self, settings):
        dims = {spec.name: spec.dim for spec in settings.query_models.models}
        assert dims == {"bekko-a25m": 384, "saga-embed": 1024, "arctic-l-v2": 1024}

    def test_a_variant_is_named_after_its_runtime_and_precision(self):
        assert variant_name("bekko-a25m", "onnx", "int8") == "bekko-a25m-onnx-int8"

    @pytest.mark.parametrize(
        ("runtimes", "message"),
        [
            ({"torch": ("int8",)}, "torch runs a model in fp32, bf16"),
            ({"onnx": ("bf16",)}, "onnx runs a model in fp32, int8"),
            ({"onnx": ("int8",)}, "lists fp32 first"),
        ],
    )
    def test_a_runtime_takes_only_its_own_precisions(self, settings, runtimes, message):
        data = {**settings.query_models.model_dump(), "runtimes": runtimes}
        with pytest.raises(ValidationError, match=message):
            QueryModelSettings.model_validate(data)


class TestReference:
    def test_the_cloud_run_figure_is_the_one_the_repository_records(self, settings):
        reference = settings.reference
        recorded = (REPO_ROOT / reference.cloud_run_source).read_text(encoding="utf-8")
        assert f"measured at {reference.cloud_run_cold_s:g} seconds" in recorded
        assert f"against {reference.cloud_run_warm_s:g} warm" in recorded

    def test_the_reference_sends_what_the_keep_warm_workflow_sends(self, settings):
        reference = settings.reference
        recorded = (REPO_ROOT / reference.cloud_run_source).read_text(encoding="utf-8")
        assert f'{{"text":"{reference.text}"}}' in recorded

    def test_the_reference_serves_the_inference_services_classifier(self, settings):
        dockerfile = (REPO_ROOT / "inference_service" / "Dockerfile").read_text(encoding="utf-8")
        assert f"ENV MODEL_ID={settings.reference.model}" in dockerfile
        assert f"ENV PORT={settings.reference.port}" in dockerfile

    def test_the_containers_here_have_the_services_limits_on_cloud_run(self, settings):
        reference, container = settings.reference, settings.container
        assert reference.cloud_run_cpu == container.cpus == "1"
        gib = {"2Gi": 2 * 1024**3, "2g": 2 * 1024**3}
        assert gib[reference.cloud_run_memory] == gib[container.memory]

    def test_the_second_set_of_cold_starts_is_pinned_to_a_single_cpu(self, settings):
        reference = settings.reference
        assert reference.pinned_cpu is not None and reference.pinned_cpu.isdigit()
        for several in ("0-3", "1,2", ""):
            with pytest.raises(ValidationError):
                ReferenceSettings.model_validate({**reference.model_dump(), "pinned_cpu": several})


class TestRealVectors:
    def test_real_vectors_come_with_the_questions_vectors(self):
        with pytest.raises(ValidationError, match="come together"):
            StoreSpec(name="r", kind="numpy", dim=4, format="float32", vectors_from="index/r")

    def test_only_a_numpy_store_is_built_from_real_vectors(self):
        with pytest.raises(ValidationError, match="only a numpy store"):
            StoreSpec(
                name="r", kind="lance", dim=4, format="float32", vectors_from="i", queries_from="q"
            )

    @pytest.mark.parametrize("path", ["../index", "index/../other", ".hidden", "/index"])
    def test_a_source_lies_below_the_search_data_directory(self, path):
        with pytest.raises(ValidationError):
            StoreSpec(
                name="r", kind="numpy", dim=4, format="float32", vectors_from=path, queries_from="q"
            )

    def test_a_store_of_real_vectors_says_so(self):
        spec = StoreSpec(
            name="r", kind="numpy", dim=4, format="float32", vectors_from="i/r", queries_from="q"
        )
        assert spec.real
        assert not StoreSpec(name="s", kind="numpy", dim=4, format="float32").real


class TestSettingsFiles:
    def test_a_settings_file_lies_beside_spike_yaml(self):
        assert settings_file(None) == SPIKE_FILE
        assert settings_file("service_recall.yaml") == SPIKE_FILE.with_name("service_recall.yaml")
        for name in ("../spike.yaml", "spike.yml", "spike"):
            with pytest.raises(ValueError):
                settings_file(name)

    def test_the_environment_names_the_file_every_part_reads(self, monkeypatch):
        monkeypatch.setenv(SETTINGS_ENV, "service_recall.yaml")
        assert load_spike_settings().workload.draw_filters is False
        monkeypatch.delenv(SETTINGS_ENV)
        assert load_spike_settings().workload.draw_filters is True

    @pytest.mark.parametrize("name", ["service_storage.yaml", "service_recall.yaml"])
    def test_the_service_check_holds_the_tuned_configuration_in_4_gib(self, name):
        settings = load_spike_settings(settings_file(name))
        assert (settings.budget.memory_gib, settings.container.memory) == (4, "4g")
        assert (settings.search.rrf_k, settings.search.depth) == (10, 1000)
        assert [(spec.name, spec.real) for spec in settings.stores] == [
            ("numpy-1024-float32-qwen3-8b", True)
        ]
        assert settings.query_models.models == ()
