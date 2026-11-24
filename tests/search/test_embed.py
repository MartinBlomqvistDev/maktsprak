"""Tests for search/index/embed.py: the embedders, the input sets and the resumable runs."""

from __future__ import annotations

import base64
import json
import os

import httpx
import numpy as np
import pytest

from search.config import CandidateSettings
from search.index.embed import (
    RUN_MANIFEST,
    ApiError,
    BudgetReached,
    EmbedderSpec,
    HashEmbedder,
    OpenRouterEmbedder,
    Usage,
    build_embedder,
    embed_inputs,
    load_openrouter_key,
    normalise,
    read_inputs,
    read_query_vectors,
    read_run_manifest,
    read_shards,
    shard_path,
    write_inputs,
)

PASSAGES = [
    (0, "a:0-0", "Kärnkraften ska byggas ut."),
    (2, "b:0-0", "Skolan behöver fler lärare."),
    (3, "c:0-1", "Sjukvården har långa köer."),
    (7, "d:0-0", "Elpriset stiger när kärnkraften avvecklas."),
    (9, "e:0-0", "Lärarna ska få högre lön."),
]
QUERIES = [("natural-0001", "natural", "Ska kärnkraften byggas ut?")]


@pytest.fixture
def inputs_dir(tmp_path):
    directory = tmp_path / "inputs"
    write_inputs(directory, "subset-200", 200, PASSAGES, QUERIES, {"passages.parquet": "f" * 64})
    return directory


class TestHashEmbedder:
    def test_the_same_text_gives_the_same_unit_vector(self):
        embedder = HashEmbedder(dimension=32)
        first = embedder.embed(["Kärnkraften ska byggas ut."], "document")
        second = HashEmbedder(dimension=32).embed(["Kärnkraften ska byggas ut."], "query")
        assert first.dtype == np.float32 and first.shape == (1, 32)
        np.testing.assert_array_equal(first, second)
        assert np.linalg.norm(first[0]) == pytest.approx(1.0, abs=1e-6)

    def test_texts_that_share_words_lie_closer(self):
        vectors = HashEmbedder(dimension=256).embed(
            ["kärnkraften byggas", "kärnkraften byggas ut", "skolan lärare lön"], "document"
        )
        assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]

    def test_a_text_without_terms_still_has_a_direction(self):
        vector = HashEmbedder(dimension=8).embed(["!!!"], "document")[0]
        assert vector.tolist() == [1.0] + [0.0] * 7

    def test_usage_counts_the_texts_and_starts_over(self):
        embedder = HashEmbedder()
        embedder.embed(["a", "b"], "document")
        assert embedder.take_usage().texts == 2
        assert embedder.take_usage() == Usage()


class TestNormalise:
    def test_rows_get_length_one(self):
        rows = normalise(np.array([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(rows, axis=1), [1.0, 1.0])

    @pytest.mark.parametrize("bad", [[[0.0, 0.0]], [[np.nan, 1.0]], [[np.inf, 1.0]]])
    def test_a_zero_or_non_finite_row_raises(self, bad):
        with pytest.raises(ValueError):
            normalise(np.array(bad))


class TestInputs:
    def test_an_input_set_reads_back_as_written(self, inputs_dir):
        inputs = read_inputs(inputs_dir)
        assert inputs.positions.tolist() == [0, 2, 3, 7, 9]
        assert inputs.passage_ids == ["a:0-0", "b:0-0", "c:0-1", "d:0-0", "e:0-0"]
        assert inputs.texts[1] == "Skolan behöver fler lärare."
        assert inputs.query_ids == ["natural-0001"]
        assert inputs.manifest.target_words == 200

    def test_a_changed_text_is_caught(self, inputs_dir):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(inputs_dir / "passages.parquet")
        texts = table.column("text").to_pylist()
        texts[0] = texts[0].upper()
        table = table.set_column(2, "text", pa.array(texts))
        pq.write_table(table, inputs_dir / "passages.parquet")
        with pytest.raises(ValueError, match="does not match"):
            read_inputs(inputs_dir)

    def test_passages_out_of_order_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="ascending"):
            write_inputs(tmp_path, "x", 200, [PASSAGES[1], PASSAGES[0]], QUERIES, {})

    def test_an_id_twice_is_refused(self, tmp_path):
        twice = [PASSAGES[0], (5, "a:0-0", "igen")]
        with pytest.raises(ValueError, match="twice"):
            write_inputs(tmp_path, "x", 200, twice, QUERIES, {})


class TestRuns:
    def test_a_run_writes_its_shards_and_questions(self, inputs_dir, tmp_path):
        run = tmp_path / "run"
        manifest = embed_inputs(read_inputs(inputs_dir), HashEmbedder(16), run, shard_rows=2)
        assert manifest.complete and manifest.expected_shards == 3
        assert [record.rows for record in manifest.shards] == [2, 2, 1]
        vectors = read_shards(run, manifest)
        expected = HashEmbedder(16).embed([text for _, _, text in PASSAGES], "document")
        np.testing.assert_array_equal(vectors, expected)
        assert read_query_vectors(run, manifest).shape == (1, 16)
        assert manifest.usage.texts == len(PASSAGES) + 1

    def test_a_stopped_run_goes_on_from_the_first_missing_shard(self, inputs_dir, tmp_path):
        inputs = read_inputs(inputs_dir)
        run = tmp_path / "run"
        first = embed_inputs(inputs, HashEmbedder(16), run, shard_rows=2, max_shards=1)
        assert len(first.shards) == 1 and not first.complete
        embedder = HashEmbedder(16)
        whole = embed_inputs(inputs, embedder, run, shard_rows=2)
        assert whole.complete
        assert whole.shards[0] == first.shards[0]
        assert whole.usage.texts == len(PASSAGES) + 1
        reference = embed_inputs(inputs, HashEmbedder(16), tmp_path / "again", shard_rows=2)
        np.testing.assert_array_equal(
            read_shards(run, whole), read_shards(tmp_path / "again", reference)
        )

    def test_a_whole_run_is_not_embedded_again(self, inputs_dir, tmp_path):
        inputs = read_inputs(inputs_dir)
        run = tmp_path / "run"
        embed_inputs(inputs, HashEmbedder(16), run, shard_rows=2)
        embedder = HashEmbedder(16)
        embed_inputs(inputs, embedder, run, shard_rows=2)
        assert embedder.take_usage().texts == 0

    def test_another_embedder_is_refused(self, inputs_dir, tmp_path):
        inputs = read_inputs(inputs_dir)
        run = tmp_path / "run"
        embed_inputs(inputs, HashEmbedder(16), run, shard_rows=2, max_shards=1)
        with pytest.raises(ValueError, match="another embedder"):
            embed_inputs(inputs, HashEmbedder(32), run, shard_rows=2)

    def test_other_shard_sizes_are_refused(self, inputs_dir, tmp_path):
        inputs = read_inputs(inputs_dir)
        run = tmp_path / "run"
        embed_inputs(inputs, HashEmbedder(16), run, shard_rows=2, max_shards=1)
        with pytest.raises(ValueError, match="shards of 2 rows"):
            embed_inputs(inputs, HashEmbedder(16), run, shard_rows=3)

    def test_other_inputs_are_refused(self, inputs_dir, tmp_path):
        run = tmp_path / "run"
        embed_inputs(read_inputs(inputs_dir), HashEmbedder(16), run, shard_rows=2, max_shards=1)
        other = tmp_path / "other"
        write_inputs(other, "subset-200", 200, PASSAGES[:3], QUERIES, {})
        with pytest.raises(ValueError, match="other inputs"):
            embed_inputs(read_inputs(other), HashEmbedder(16), run, shard_rows=2)

    def test_a_damaged_shard_is_caught_on_resume_and_on_read(self, inputs_dir, tmp_path):
        inputs = read_inputs(inputs_dir)
        run = tmp_path / "run"
        manifest = embed_inputs(inputs, HashEmbedder(16), run, shard_rows=2)
        np.save(shard_path(run, 0), np.zeros((2, 16), dtype=np.float32))
        with pytest.raises(ValueError, match="no longer matches"):
            embed_inputs(inputs, HashEmbedder(16), run, shard_rows=2)
        with pytest.raises(ValueError, match="does not match"):
            read_shards(run, manifest)

    def test_vectors_of_the_wrong_shape_stop_the_run(self, inputs_dir, tmp_path):
        class Short(HashEmbedder):
            def embed(self, texts, role):
                return super().embed(texts, role)[:-1]

        with pytest.raises(ValueError, match="returned"):
            embed_inputs(read_inputs(inputs_dir), Short(16), tmp_path / "run", shard_rows=2)

    def test_vectors_not_of_length_one_stop_the_run(self, inputs_dir, tmp_path):
        class Long(HashEmbedder):
            def embed(self, texts, role):
                return super().embed(texts, role) * 2

        with pytest.raises(ValueError, match="length 1"):
            embed_inputs(read_inputs(inputs_dir), Long(16), tmp_path / "run", shard_rows=2)

    def test_the_manifest_records_the_embedder_and_the_inputs(self, inputs_dir, tmp_path):
        run = tmp_path / "run"
        embed_inputs(read_inputs(inputs_dir), HashEmbedder(16), run, shard_rows=4)
        manifest = read_run_manifest(run)
        assert manifest.embedder.kind == "hash" and manifest.embedder.dimension == 16
        assert manifest.inputs.passages == 5
        assert "numpy" in manifest.software
        assert json.loads((run / RUN_MANIFEST).read_text(encoding="utf-8"))["shard_rows"] == 4


SPEC = EmbedderSpec(
    name="remote",
    kind="openrouter",
    model="org/model",
    revision="r",
    dimension=4,
    query_prompt="Q:",
    document_prompt="",
    provider="someone",
    quantization="bf16",
)


def encoded(vector):
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def api(handler, **options):
    return OpenRouterEmbedder(
        SPEC,
        key="test-key",
        price_per_million_usd=0.01,
        cap_usd=options.pop("cap_usd", 1.0),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0.0,
        **options,
    )


class TestOpenRouterEmbedder:
    def test_a_request_pins_the_provider_and_asks_for_the_dimension(self):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append((request.headers["authorization"], body))
            data = [
                {"index": i, "object": "embedding", "embedding": encoded([1, 0, 0, 0])}
                for i in range(len(body["input"]))
            ]
            return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 7}})

        vectors = api(handler).embed(["vad tycker partierna?"], "query")
        authorization, body = seen[0]
        assert authorization == "Bearer test-key"
        assert body["model"] == "org/model" and body["dimensions"] == 4
        assert body["input"] == ["Q:vad tycker partierna?"]
        assert body["encoding_format"] == "base64"
        assert body["provider"] == {
            "order": ["someone"],
            "allow_fallbacks": False,
            "quantizations": ["bf16"],
        }
        assert vectors.tolist() == [[1.0, 0.0, 0.0, 0.0]]

    def test_longer_vectors_are_cut_and_normalised_again(self):
        def handler(request):
            body = json.loads(request.content)
            data = [
                {"index": i, "embedding": [3.0, 4.0, 0.0, 0.0, 9.0, 9.0]}
                for i in range(len(body["input"]))
            ]
            return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 1}})

        embedder = api(handler)
        vectors = embedder.embed(["a", "b"], "document")
        np.testing.assert_allclose(vectors, [[0.6, 0.8, 0.0, 0.0]] * 2, atol=1e-6)
        assert embedder.client_side_dimensions == 2

    def test_inputs_go_in_batches_and_come_back_in_order(self):
        def handler(request):
            body = json.loads(request.content)
            data = [
                {"index": i, "embedding": encoded([float(text), 1.0, 0.0, 0.0])}
                for i, text in reversed(list(enumerate(body["input"])))
            ]
            return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 2}})

        embedder = api(handler, batch_inputs=2, concurrency=2)
        vectors = embedder.embed([str(n) for n in range(5)], "document")
        ratios = vectors[:, 0] / vectors[:, 1]
        np.testing.assert_allclose(ratios, [0, 1, 2, 3, 4], atol=1e-5)
        usage = embedder.take_usage()
        assert (usage.requests, usage.tokens, usage.texts) == (3, 6, 5)

    def test_the_cost_is_the_reported_one_or_the_listed_price_if_higher(self):
        def handler(request):
            body = json.loads(request.content)
            data = [{"index": i, "embedding": [1, 0, 0, 0]} for i in range(len(body["input"]))]
            return httpx.Response(
                200, json={"data": data, "usage": {"prompt_tokens": 1_000_000, "cost": 0.002}}
            )

        embedder = api(handler)
        embedder.embed(["a"], "document")
        assert embedder.take_usage().cost_usd == pytest.approx(0.01)

    def test_rate_limits_and_server_errors_are_sent_again(self):
        answers = iter([429, 503, 200])

        def handler(request):
            status = next(answers)
            if status != 200:
                return httpx.Response(status, headers={"retry-after": "0"}, json={})
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0, 1, 0, 0]}]})

        embedder = api(handler)
        assert embedder.embed(["a"], "document").tolist() == [[0.0, 1.0, 0.0, 0.0]]
        assert embedder.take_usage().requests == 3

    def test_a_refusal_is_not_sent_again(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(402, json={"error": {"message": "Insufficient credits"}})

        with pytest.raises(ApiError, match="402"):
            api(handler).embed(["a"], "document")
        assert len(calls) == 1

    def test_too_few_embeddings_are_refused(self):
        def handler(request):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]})

        with pytest.raises(ApiError, match="1 embeddings for 2 inputs"):
            api(handler).embed(["a", "b"], "document")

    def test_a_short_vector_is_refused(self):
        def handler(request):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

        with pytest.raises(ApiError, match="2 values"):
            api(handler).embed(["a"], "document")

    def test_no_request_starts_once_the_cap_is_spent(self):
        def handler(request):
            body = json.loads(request.content)
            data = [{"index": i, "embedding": [1, 0, 0, 0]} for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": data, "usage": {"cost": 0.6}})

        embedder = api(handler, cap_usd=1.0, batch_inputs=1, concurrency=1)
        with pytest.raises(BudgetReached):
            embedder.embed(["a", "b", "c"], "document")
        assert embedder.spent_usd == pytest.approx(1.2)

    def test_texts_sent_singly_keep_their_order_and_count_their_attempts(self):
        sent = []

        def handler(request):
            body = json.loads(request.content)
            sent.append(body["input"])
            if body["input"] == ["Q:b"] and sent.count(["Q:b"]) == 1:
                return httpx.Response(429, json={"error": "slow down"})
            value = {"Q:a": 1.0, "Q:b": 2.0, "Q:c": 3.0}[body["input"][0]]
            data = [{"index": 0, "embedding": encoded([value, 1.0, 0.0, 0.0])}]
            return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 3}})

        embedder = api(handler)
        vectors, answered = embedder.embed_singly(["a", "b", "c"], "query")
        assert sent == [["Q:a"], ["Q:b"], ["Q:b"], ["Q:c"]]
        np.testing.assert_allclose(vectors[:, 0] / vectors[:, 1], [1, 2, 3], atol=1e-5)
        assert [one.attempts for one in answered] == [1, 2, 1]
        assert all(one.seconds >= 0 for one in answered)
        usage = embedder.take_usage()
        assert (usage.texts, usage.requests, usage.tokens) == (3, 4, 9)


class TestBuild:
    def test_an_api_model_needs_a_key(self):
        candidate = CandidateSettings(
            name="remote",
            kind="openrouter",
            model="org/model",
            revision="r",
            dimension=4,
            query_prompt="",
            document_prompt="",
            provider="someone",
            quantization="bf16",
            price_per_million_usd=0.01,
            licensed=True,
        )
        with pytest.raises(ValueError, match="needs a key"):
            build_embedder(candidate)
        assert isinstance(build_embedder(candidate, key="k"), OpenRouterEmbedder)

    def test_the_key_comes_from_the_environment_first(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('OPENROUTER_API_KEY="from-file"\n', encoding="utf-8")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert load_openrouter_key(env) == "from-file"
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert load_openrouter_key(env) == "from-env"

    def test_a_missing_key_stops(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            load_openrouter_key(tmp_path / ".env")


@pytest.mark.skipif(
    not os.environ.get("MAKTSPRAK_TEST_LOCAL_MODEL"),
    reason="Set MAKTSPRAK_TEST_LOCAL_MODEL to run a downloaded local model.",
)
def test_a_local_model_embeds_with_its_prompts():
    pytest.importorskip("sentence_transformers")
    from search.config import load_settings

    candidate = load_settings().vectors.candidate(os.environ["MAKTSPRAK_TEST_LOCAL_MODEL"])
    embedder = build_embedder(candidate, device="cpu")
    vectors = embedder.embed(["Ska kärnkraften byggas ut?", "Skolan behöver lärare."], "query")
    assert vectors.shape == (2, candidate.dimension)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-3)
