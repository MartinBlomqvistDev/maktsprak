"""Tests for search_service/spike/query_model.py, with the smallest candidate.

Skipped without sentence-transformers, ONNX Runtime and onnx, and when the
Hugging Face cache of this machine does not hold bekko-embedding-v1-a25m at its
pinned revision: nothing is downloaded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from search_service.spike.query_model import (  # noqa: E402
    ONNX_NAME,
    PREPARED_NAME,
    TOKENIZER_NAME,
    OnnxQueryModel,
    TorchQueryModel,
    load_query_model,
    model_directory,
    prepare_models,
)
from search_service.spike.settings import load_spike_settings  # noqa: E402

SETTINGS = load_spike_settings().query_models
SPEC = SETTINGS.model("bekko-a25m")
TEXTS = [
    "Ska det byggas ny kärnkraft?",
    "Hur kortas köerna i vården?",
    "Vad vill Vänsterpartiet göra med vinsterna i välfärden?",
    "skatt pensionärer",
]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    try:
        made = prepare_models(SPEC, SETTINGS.variants(), models_dir, None, SETTINGS.onnx, TEXTS)
    except (OSError, ValueError) as error:
        pytest.skip(f"bekko-embedding-v1-a25m is not in the Hugging Face cache: {error}")
    return models_dir, {entry.label: entry for entry in made}


def load(models_dir, runtime, precision):
    directory = model_directory(models_dir, SPEC, runtime, precision)
    return load_query_model(directory, SPEC, runtime, precision, SETTINGS.onnx)


def test_every_variant_is_prepared(prepared):
    _, made = prepared
    labels = [f"bekko-a25m-{runtime}-{precision}" for runtime, precision in SETTINGS.variants()]
    assert list(made) == labels
    assert all(entry.error is None for entry in made.values()), {
        label: entry.error for label, entry in made.items()
    }


def test_the_onnx_export_reproduces_the_model(prepared):
    _, made = prepared
    check = made["bekko-a25m-onnx-fp32"].prepared.check
    assert check.same_tokens is True
    assert check.cosine_min >= 0.9999
    assert check.texts == len(TEXTS) + 1


def test_a_smaller_precision_moves_the_vectors_a_little(prepared):
    _, made = prepared
    for label in ("bekko-a25m-torch-bf16", "bekko-a25m-onnx-int8"):
        check = made[label].prepared.check
        assert 0.95 < check.cosine_min < 1.0, label
    assert made["bekko-a25m-onnx-int8"].prepared.check.same_tokens is True


def test_int8_and_bfloat16_are_smaller_on_disk(prepared):
    _, made = prepared
    size = {label: entry.prepared.bytes for label, entry in made.items()}
    assert size["bekko-a25m-onnx-int8"] < 0.4 * size["bekko-a25m-onnx-fp32"]
    assert size["bekko-a25m-torch-bf16"] < 0.7 * size["bekko-a25m-torch-fp32"]


def test_an_onnx_variant_holds_its_model_and_tokenizer(prepared):
    models_dir, _ = prepared
    for precision in ("fp32", "int8"):
        directory = model_directory(models_dir, SPEC, "onnx", precision)
        assert {ONNX_NAME, TOKENIZER_NAME, PREPARED_NAME} <= {p.name for p in directory.iterdir()}
        recorded = json.loads((directory / PREPARED_NAME).read_text(encoding="utf-8"))
        assert recorded["max_tokens"] == SPEC.max_tokens


@pytest.mark.parametrize("variant", SETTINGS.variants())
def test_a_loaded_variant_embeds_a_question_as_a_unit_vector(prepared, variant):
    models_dir, _ = prepared
    model = load(models_dir, *variant)
    vector = model.embed("Ska det byggas ny kärnkraft?")
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert abs(float(np.linalg.norm(vector)) - 1) < 1e-5


@pytest.mark.parametrize("variant", SETTINGS.variants())
def test_a_start_is_timed_in_three_parts(prepared, variant):
    models_dir, _ = prepared
    times = load(models_dir, *variant).times
    assert min(times.import_s, times.load_s, times.first_s) >= 0
    assert times.total_s == pytest.approx(times.import_s + times.load_s + times.first_s, abs=2e-3)


def test_each_runtime_is_loaded_by_its_own_class(prepared):
    models_dir, _ = prepared
    assert isinstance(load(models_dir, "torch", "fp32"), TorchQueryModel)
    assert isinstance(load(models_dir, "onnx", "int8"), OnnxQueryModel)


def test_onnx_runtime_cuts_a_long_question_where_the_model_does(prepared):
    models_dir, _ = prepared
    model = load(models_dir, "onnx", "fp32")
    assert len(model.tokens(" ".join(TEXTS * 400))) == SPEC.max_tokens


def test_a_variant_is_not_loaded_as_another(prepared):
    models_dir, _ = prepared
    directory = model_directory(models_dir, SPEC, "onnx", "int8")
    with pytest.raises(ValueError, match="not bekko-a25m in onnx fp32"):
        OnnxQueryModel(directory, SPEC, "fp32", SETTINGS.onnx)
