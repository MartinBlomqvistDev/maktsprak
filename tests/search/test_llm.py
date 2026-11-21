"""Tests for search/eval/llm.py, without the SDK and without a network."""

from __future__ import annotations

import os

import pytest

from search.config import Prices
from search.eval.llm import KEY_NAME, Usage, cost_usd, load_key, message_params

OPUS = Prices(input=4.0, output=20.0)


class TestCost:
    def test_input_and_output_at_list_price(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
        assert cost_usd(usage, OPUS) == pytest.approx(4.0 + 2.0)

    def test_a_batch_costs_half(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
        assert cost_usd(usage, OPUS, batch_discount=0.5) == pytest.approx(3.0)

    def test_cache_writes_and_reads_have_their_own_price(self):
        usage = Usage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
        assert cost_usd(usage, OPUS) == pytest.approx(4.0 * 1.25 + 4.0 * 0.1)


class TestKey:
    def test_the_key_is_read_from_the_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv(KEY_NAME, raising=False)
        env = tmp_path / ".env"
        env.write_text('OTHER=1\nANTHROPIC_API_KEY="sk-test"\r\n', encoding="utf-8")
        load_key(env)
        assert os.environ[KEY_NAME] == "sk-test"

    def test_a_key_already_in_the_environment_is_kept(self, tmp_path, monkeypatch):
        monkeypatch.setenv(KEY_NAME, "from-env")
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
        load_key(env)
        assert os.environ[KEY_NAME] == "from-env"

    def test_no_key_anywhere_stops(self, tmp_path, monkeypatch):
        monkeypatch.delenv(KEY_NAME, raising=False)
        with pytest.raises(SystemExit):
            load_key(tmp_path / ".env")


class TestParams:
    def test_a_request_asks_for_json_at_an_effort(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        params = message_params(
            model="claude-opus-5-5",
            max_tokens=100,
            effort="low",
            system="system",
            user="user",
            schema=schema,
        )
        assert params["messages"] == [{"role": "user", "content": "user"}]
        assert params["output_config"] == {
            "effort": "low",
            "format": {"type": "json_schema", "schema": schema},
        }
        assert "thinking" not in params and "temperature" not in params
