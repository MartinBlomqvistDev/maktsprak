"""The query model: the local embedding model a service embeds each question with.

Two runtimes are measured. With ``torch``, the service loads the model with
sentence-transformers, as phase 3's embedder does, in float32 or bfloat16. With
``onnx``, the model is exported to ONNX once, and the service runs it with ONNX
Runtime and the ``tokenizers`` library, without torch: in float32, or with its
weights quantized to int8.

A model is prepared once, as the service's image would hold it: read from the
Hugging Face cache at its pinned revision, never downloaded, and saved in each
variant. A bfloat16 model is loaded straight into bfloat16, since loading it in
float32 first and converting would need the float32 weights in memory, which
for the largest candidate is more than the whole container has. Every prepared
variant is then loaded the way the service loads it and checked against
sentence-transformers in float32, on the same texts and the same prompt: an
ONNX variant must feed the model the same tokens, and the float32 export must
give the same vectors, or the preparation fails. How far each variant's vectors
agree with float32 is recorded beside it.

A loaded model times its start in three parts: importing its runtime, loading
the model, and embedding a first text. torch, sentence-transformers and ONNX
Runtime are imported only when a model is prepared or loaded, so the rest of the
spike runs without them.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from search_service.spike.settings import (
    OnnxSettings,
    Precision,
    QueryModelSpec,
    Runtime,
    variant_name,
)

MODELS_DIR_NAME = "models"
PREPARED_NAME = "spike_model.json"
PREPARATION_NAME = "preparation.json"
ONNX_NAME = "model.onnx"
ONNX_DATA_NAME = "model.onnx.data"
TOKENIZER_NAME = "tokenizer.json"

#: The text a loaded model embeds first, which checks its dimension.
FIRST_TEXT = "kontroll"

#: A float32 export must agree with sentence-transformers at least this well on every text.
EXPORT_AGREEMENT = 0.9999

#: A protobuf, and so an ONNX file with its weights inside, holds at most 2 GiB.
PROTOBUF_LIMIT = 2 * 1024**3

_TORCH_DTYPES: dict[str, str] = {"fp32": "float32", "bf16": "bfloat16"}
_INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelCheck(_Frozen):
    """How a prepared model agreed with sentence-transformers in float32.

    Attributes:
        texts: How many texts were embedded both ways; the last is all the
            others joined, long enough to be truncated.
        same_tokens: Whether the model was fed the same tokens for every text;
            ``None`` for torch, which uses sentence-transformers' own tokenizer.
        cosine_min: The lowest cosine between a text's two vectors.
        cosine_mean: Their mean cosine.
    """

    texts: int
    same_tokens: bool | None
    cosine_min: float
    cosine_mean: float


class PreparedModel(_Frozen):
    """One query model as prepared in one runtime and one precision.

    Attributes:
        name: The model's short name.
        model: Its id on Hugging Face.
        revision: The commit its weights were read at.
        runtime: The runtime that runs it.
        precision: The precision of its weights.
        dim: The dimension of its vectors.
        max_tokens: The most tokens of a question it reads; the rest is cut.
        query_prompt: The text put before a question.
        bytes: The size of the prepared model on disk.
        check: How it agreed with sentence-transformers in float32.
    """

    name: str
    model: str
    revision: str
    runtime: Runtime
    precision: Precision
    dim: int
    max_tokens: int
    query_prompt: str
    bytes: int
    check: ModelCheck | None = None

    @property
    def label(self) -> str:
        """The model with its runtime and precision, as runs are named."""
        return variant_name(self.name, self.runtime, self.precision)


class Preparation(_Frozen):
    """What became of one variant when the models were prepared.

    Attributes:
        label: The variant, as runs are named.
        prepared: The prepared model, or ``None`` when it was not prepared.
        error: Why it was not.
    """

    label: str
    prepared: PreparedModel | None
    error: str | None


class StartTimes(_Frozen):
    """How long a query model took to start, in seconds.

    Attributes:
        import_s: Importing its runtime.
        load_s: Loading the model and its tokenizer.
        first_s: Embedding the first text.
    """

    import_s: float
    load_s: float
    first_s: float

    @property
    def total_s(self) -> float:
        """The three together."""
        return round(self.import_s + self.load_s + self.first_s, 3)


class QueryModel(Protocol):
    """A loaded query model."""

    @property
    def dim(self) -> int:
        """The dimension of its vectors."""
        ...

    @property
    def times(self) -> StartTimes:
        """How long it took to start."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed one question as a float32 unit vector."""
        ...


def model_directory(
    models_dir: Path, spec: QueryModelSpec, runtime: Runtime, precision: Precision
) -> Path:
    """Where one model is saved in one runtime and one precision."""
    return models_dir / variant_name(spec.name, runtime, precision)


def _seconds(start: float, end: float) -> float:
    return round(end - start, 3)


def _read_prepared(
    directory: Path, spec: QueryModelSpec, runtime: Runtime, precision: Precision
) -> PreparedModel:
    """Read a prepared model's description, and check it is the variant asked for."""
    prepared = PreparedModel.model_validate(
        json.loads((directory / PREPARED_NAME).read_text(encoding="utf-8"))
    )
    found = (prepared.model, prepared.revision, prepared.runtime, prepared.precision)
    if found != (spec.model, spec.revision, runtime, precision):
        raise ValueError(f"The model in {directory} is not {spec.name} in {runtime} {precision}.")
    return prepared


def _unit(vector: Any) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TorchQueryModel:
    """A prepared model run by sentence-transformers on the CPU.

    Attributes:
        spec: The model.
        precision: The precision of its weights.
        prepared: What the saved model was made from.
        times: How long it took to start.
    """

    def __init__(self, directory: Path, spec: QueryModelSpec, precision: Precision) -> None:
        started = time.perf_counter()
        import torch
        from sentence_transformers import SentenceTransformer

        imported = time.perf_counter()
        self.prepared = _read_prepared(directory, spec, "torch", precision)
        self.spec = spec
        self.precision = precision
        dtype = getattr(torch, _TORCH_DTYPES[precision])
        self.model = SentenceTransformer(
            str(directory), device="cpu", model_kwargs={"dtype": dtype}
        )
        self.model.max_seq_length = self.prepared.max_tokens
        loaded = time.perf_counter()
        produced = self.embed(FIRST_TEXT).shape[0]
        self.times = StartTimes(
            import_s=_seconds(started, imported),
            load_s=_seconds(imported, loaded),
            first_s=_seconds(loaded, time.perf_counter()),
        )
        if produced != spec.dim:
            raise ValueError(f"{spec.name} gives {produced} dimensions, not {spec.dim}.")

    @property
    def dim(self) -> int:
        """The dimension of its vectors."""
        return self.spec.dim

    def embed(self, text: str) -> np.ndarray:
        """Embed one question, with the model's prompt for a question before it."""
        vector = self.model.encode(
            [text],
            prompt=self.spec.query_prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return _unit(vector)


class OnnxQueryModel:
    """A prepared model run by ONNX Runtime and the tokenizers library, without torch.

    Attributes:
        spec: The model.
        precision: The precision of its weights.
        prepared: What the saved model was made from.
        inputs: The names of the inputs the exported model takes.
        times: How long it took to start.
    """

    def __init__(
        self,
        directory: Path,
        spec: QueryModelSpec,
        precision: Precision,
        settings: OnnxSettings,
    ) -> None:
        started = time.perf_counter()
        import onnxruntime
        from tokenizers import Tokenizer

        imported = time.perf_counter()
        self.prepared = _read_prepared(directory, spec, "onnx", precision)
        self.spec = spec
        self.precision = precision
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = settings.threads
        options.inter_op_num_threads = settings.threads
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            str(directory / ONNX_NAME), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = tuple(item.name for item in self.session.get_inputs())
        self.tokenizer = Tokenizer.from_file(str(directory / TOKENIZER_NAME))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.prepared.max_tokens)
        loaded = time.perf_counter()
        produced = self.embed(FIRST_TEXT).shape[0]
        self.times = StartTimes(
            import_s=_seconds(started, imported),
            load_s=_seconds(imported, loaded),
            first_s=_seconds(loaded, time.perf_counter()),
        )
        if produced != spec.dim:
            raise ValueError(f"{spec.name} gives {produced} dimensions, not {spec.dim}.")

    @property
    def dim(self) -> int:
        """The dimension of its vectors."""
        return self.spec.dim

    def tokens(self, text: str) -> list[int]:
        """The tokens the model is fed for one question."""
        return list(self.tokenizer.encode(self.spec.query_prompt + text).ids)

    def embed(self, text: str) -> np.ndarray:
        """Embed one question, with the model's prompt for a question before it."""
        encoding = self.tokenizer.encode(self.spec.query_prompt + text)
        values = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
        }
        feeds = {name: np.asarray([values[name]], dtype=np.int64) for name in self.inputs}
        return _unit(self.session.run(None, feeds)[0][0])


def load_query_model(
    directory: Path,
    spec: QueryModelSpec,
    runtime: Runtime,
    precision: Precision,
    onnx: OnnxSettings,
) -> TorchQueryModel | OnnxQueryModel:
    """Load a prepared model the way the service does.

    Args:
        directory: Where the model was prepared.
        spec: The model.
        runtime: The runtime that runs it.
        precision: The precision of its weights.
        onnx: How ONNX Runtime runs a model.

    Returns:
        The model, with the times of its start.
    """
    if runtime == "torch":
        return TorchQueryModel(directory, spec, precision)
    return OnnxQueryModel(directory, spec, precision, onnx)


def _directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _from_cache(spec: QueryModelSpec, precision: Precision, cache_dir: Path | None) -> Any:
    """The model from the Hugging Face cache, loaded straight into *precision*, never downloaded."""
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        spec.model,
        revision=spec.revision,
        device="cpu",
        cache_folder=None if cache_dir is None else str(cache_dir),
        local_files_only=True,
        model_kwargs={"dtype": getattr(torch, _TORCH_DTYPES[precision])},
    )


def _encoder(model: Any, names: Sequence[str]) -> Any:
    """A module that takes the tokenizer's tensors and returns sentence-transformers' embedding."""
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, *inputs: Any) -> Any:
            return self.model(dict(zip(names, inputs, strict=True)))["sentence_embedding"]

    return Encoder().eval()


def _save_onnx(source: Path, target: Path) -> None:
    """Save an exported model as one file, with its weights beside it when over 2 GiB.

    torch's exporter writes the weights of a large model as one file per tensor.
    """
    import onnx

    size = _directory_bytes(source.parent)
    model = onnx.load(str(source))
    external = size > PROTOBUF_LIMIT
    onnx.save_model(
        model,
        str(target),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=ONNX_DATA_NAME,
    )


def _export_onnx(model: Any, spec: QueryModelSpec, directory: Path, opset: int, text: str) -> None:
    """Export a float32 model to ONNX, with its tokenizer, into *directory*.

    The export traces one question through sentence-transformers' own modules,
    so the ONNX model pools as the model does. Its batch and sequence stay free.
    """
    import torch

    features = model.preprocess([text], prompt=spec.query_prompt)
    names = [name for name in _INPUT_NAMES if name in features]
    encoder = _encoder(model, names)
    directory.mkdir(parents=True, exist_ok=True)
    axes = {name: {0: "batch", 1: "sequence"} for name in names}
    with tempfile.TemporaryDirectory(dir=directory.parent) as scratch:
        exported = Path(scratch) / ONNX_NAME
        with torch.no_grad():
            torch.onnx.export(
                encoder,
                tuple(features[name] for name in names),
                str(exported),
                input_names=names,
                output_names=["embedding"],
                dynamic_axes={**axes, "embedding": {0: "batch"}},
                opset_version=opset,
                dynamo=False,
            )
        _save_onnx(exported, directory / ONNX_NAME)
    model.tokenizer.backend_tokenizer.save(str(directory / TOKENIZER_NAME))


def _quantize_onnx(source: Path, directory: Path) -> None:
    """Quantize the float32 export's weights to int8, as ONNX Runtime's dynamic quantization does.

    Activations stay float and are quantized as each operator runs; weights are
    quantized per channel.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    directory.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(source / ONNX_NAME),
        str(directory / ONNX_NAME),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    shutil.copyfile(source / TOKENIZER_NAME, directory / TOKENIZER_NAME)


def _reference_vectors(model: Any, spec: QueryModelSpec, texts: Sequence[str]) -> np.ndarray:
    """sentence-transformers' float32 vectors, one question at a time as the service embeds."""
    return np.stack(
        [
            _unit(
                model.encode(
                    [text],
                    prompt=spec.query_prompt,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0]
            )
            for text in texts
        ]
    )


def _check(
    loaded: TorchQueryModel | OnnxQueryModel,
    reference: Any,
    spec: QueryModelSpec,
    texts: Sequence[str],
    expected: np.ndarray,
) -> ModelCheck:
    """Embed *texts* with a loaded variant and compare with sentence-transformers in float32."""
    pairs = zip(texts, expected, strict=True)
    cosines = np.array([float(loaded.embed(text) @ want) for text, want in pairs])
    same_tokens = None
    if isinstance(loaded, OnnxQueryModel):
        same_tokens = all(
            loaded.tokens(text)
            == reference.preprocess([text], prompt=spec.query_prompt)["input_ids"][0].tolist()
            for text in texts
        )
    return ModelCheck(
        texts=len(texts),
        same_tokens=same_tokens,
        cosine_min=round(float(cosines.min()), 6),
        cosine_mean=round(float(cosines.mean()), 6),
    )


def _write_prepared(directory: Path, prepared: PreparedModel) -> None:
    (directory / PREPARED_NAME).write_text(
        prepared.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def prepare_models(
    spec: QueryModelSpec,
    variants: Sequence[tuple[Runtime, Precision]],
    models_dir: Path,
    cache_dir: Path | None,
    onnx: OnnxSettings,
    texts: Sequence[str],
) -> list[Preparation]:
    """Prepare one model in each variant, from the Hugging Face cache, and check each.

    A variant that cannot be prepared is reported with the reason, and the
    others are still prepared, except int8 for ONNX when the float32 export it
    is quantized from failed.

    Args:
        spec: The model.
        variants: The runtimes and precisions, in order; ONNX in float32 before int8.
        models_dir: Where the prepared models go.
        cache_dir: The Hugging Face cache to read from; ``None`` uses the default.
        onnx: How a model is exported to ONNX and run.
        texts: The questions every variant is checked on.

    Returns:
        What became of each variant, in order. A variant that failed is left
        out of *models_dir*.

    Raises:
        OSError: If the cache does not hold the model at its revision.
        ValueError: If no texts are given.
    """
    if not texts:
        raise ValueError("A model is checked on texts, and none were given.")
    checked = [*texts, " ".join(texts)]
    reference = _from_cache(spec, "fp32", cache_dir)
    max_tokens = min(spec.max_tokens, reference.max_seq_length or spec.max_tokens)
    reference.max_seq_length = max_tokens
    expected = _reference_vectors(reference, spec, checked)
    results = []
    for runtime, precision in variants:
        label = variant_name(spec.name, runtime, precision)
        directory = model_directory(models_dir, spec, runtime, precision)
        try:
            if directory.exists():
                shutil.rmtree(directory)
            if runtime == "torch":
                model = (
                    reference if precision == "fp32" else _from_cache(spec, precision, cache_dir)
                )
                directory.mkdir(parents=True)
                model.save(str(directory))
                del model
            elif precision == "fp32":
                _export_onnx(reference, spec, directory, onnx.opset, texts[0])
            else:
                source = model_directory(models_dir, spec, "onnx", "fp32")
                if not (source / PREPARED_NAME).is_file():
                    raise ValueError("the float32 export it is quantized from was not prepared")
                _quantize_onnx(source, directory)
            prepared = PreparedModel(
                name=spec.name,
                model=spec.model,
                revision=spec.revision,
                runtime=runtime,
                precision=precision,
                dim=spec.dim,
                max_tokens=max_tokens,
                query_prompt=spec.query_prompt,
                bytes=_directory_bytes(directory),
            )
            _write_prepared(directory, prepared)
            loaded = load_query_model(directory, spec, runtime, precision, onnx)
            check = _check(loaded, reference, spec, checked, expected)
            del loaded
            if check.same_tokens is False:
                raise ValueError("ONNX Runtime's tokenizer gives other tokens than the model's own")
            if (runtime, precision) == ("onnx", "fp32") and check.cosine_min < EXPORT_AGREEMENT:
                raise ValueError(
                    f"the export agrees with the model at a cosine of {check.cosine_min} at worst"
                )
            prepared = prepared.model_copy(update={"check": check})
            _write_prepared(directory, prepared)
            results.append(Preparation(label=label, prepared=prepared, error=None))
        except Exception as error:  # an exporter or ONNX Runtime raises its own kinds
            shutil.rmtree(directory, ignore_errors=True)
            reason = f"{type(error).__name__}: {error}"
            results.append(Preparation(label=label, prepared=None, error=reason))
    return results
