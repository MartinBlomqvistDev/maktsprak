"""Calls to Anthropic's Messages API, one at a time or as a message batch.

This is the only module that imports Anthropic's SDK, and it imports it only
when a call is made, so everything else in the package runs, and is tested,
without the SDK or a key. A request is a plain dictionary of the parameters the
Messages API takes; the answer comes back as a :class:`Completion`, whatever
went wrong on the way.

The key is read from the environment, as the SDK does. :func:`load_key` puts it
there from a ``.env`` file without ever printing it.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from search.config import Prices

logger = logging.getLogger("llm")

KEY_NAME = "ANTHROPIC_API_KEY"

#: The SDK retries rate limits, overloads and server errors this many times itself.
SDK_RETRIES = 5

#: Seconds between two looks at a message batch that has not ended.
BATCH_POLL_SECONDS = 30

#: Price multipliers on the input price for tokens written to and read from the
#: prompt cache with the default five-minute lifetime.
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.1


class Usage(BaseModel):
    """The tokens one request used, as the API reports them.

    Attributes:
        input_tokens: Input tokens outside the prompt cache.
        output_tokens: Output tokens, thinking included.
        cache_creation_input_tokens: Input tokens written to the cache.
        cache_read_input_tokens: Input tokens read from the cache.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def cost_usd(usage: Usage, prices: Prices, batch_discount: float = 0.0) -> float:
    """What one request cost.

    Args:
        usage: Its tokens.
        prices: The model's list prices.
        batch_discount: The share taken off when the request ran in a message batch.

    Returns:
        The cost in US dollars.
    """
    input_cost = prices.input * (
        usage.input_tokens
        + CACHE_WRITE_FACTOR * usage.cache_creation_input_tokens
        + CACHE_READ_FACTOR * usage.cache_read_input_tokens
    )
    total = (input_cost + prices.output * usage.output_tokens) / 1_000_000
    return total * (1 - batch_discount)


class Completion(BaseModel):
    """What came back for one request.

    Attributes:
        custom_id: The id the caller gave the request.
        model: The model that answered, as the response names it.
        stop_reason: Why the model stopped, for example ``end_turn``,
            ``max_tokens`` or ``refusal``.
        text: The text of the first text block, which with a JSON schema is the JSON.
        usage: The tokens used.
        batch_id: The message batch the request ran in, if any.
        error: What went wrong when no response came back.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str
    model: str | None = None
    stop_reason: str | None = None
    text: str | None = None
    usage: Usage = Usage()
    batch_id: str | None = None
    error: str | None = None


def load_key(env_file: Path) -> None:
    """Put the API key from a ``.env`` file into the environment, unless it is already there.

    The file is read line by line for ``ANTHROPIC_API_KEY=...``; quotes around
    the value and a Windows line end are dropped. The value is never printed.

    Args:
        env_file: The ``.env`` file.

    Raises:
        SystemExit: If neither the environment nor the file holds the key.
    """
    if os.environ.get(KEY_NAME):
        return
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == KEY_NAME:
                os.environ[KEY_NAME] = value.strip().strip('"').strip("'")
                return
    raise SystemExit(f"{KEY_NAME} is neither in the environment nor in {env_file}.")


def message_params(
    *,
    model: str,
    max_tokens: int,
    effort: str,
    system: str,
    user: str,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the parameters of one Messages API request that must answer in JSON.

    Thinking is left to the model's default, which on the current models is
    adaptive, and its depth is set by *effort*.

    Args:
        model: The model id.
        max_tokens: The most output tokens, thinking included.
        effort: ``low``, ``medium``, ``high``, ``xhigh`` or ``max``.
        system: The system prompt.
        user: The user message.
        schema: The JSON schema the answer must follow.

    Returns:
        The request's parameters.
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": dict(schema)},
        },
    }


def _completion_from_message(custom_id: str, message: Any, batch_id: str | None) -> Completion:
    """Turn the SDK's message into a :class:`Completion`."""
    text = next((block.text for block in message.content if block.type == "text"), None)
    usage = message.usage
    return Completion(
        custom_id=custom_id,
        model=message.model,
        stop_reason=message.stop_reason,
        text=text,
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        ),
        batch_id=batch_id,
    )


def _client() -> Any:
    """Build the SDK's client, which reads the key from the environment."""
    import anthropic

    return anthropic.Anthropic(max_retries=SDK_RETRIES)


def complete_one(
    custom_id: str, params: Mapping[str, Any], client: Any | None = None
) -> Completion:
    """Send one request and wait for its answer.

    Args:
        custom_id: The caller's id for the request.
        params: The request's parameters.
        client: The SDK client; a new one when ``None``.

    Returns:
        The completion. An error after the SDK's own retries comes back in
        ``error``, not as an exception.
    """
    import anthropic

    client = client or _client()
    try:
        message = client.messages.create(**params)
    except anthropic.APIStatusError as error:
        return Completion(custom_id=custom_id, error=f"HTTP {error.status_code}: {error.message}")
    except anthropic.APIConnectionError as error:
        return Completion(custom_id=custom_id, error=f"connection: {error}")
    return _completion_from_message(custom_id, message, None)


def complete_many(
    requests: Mapping[str, Mapping[str, Any]], workers: int = 4
) -> dict[str, Completion]:
    """Send requests one at a time each, a few in parallel, and wait for all answers.

    Args:
        requests: The requests' parameters, by custom id.
        workers: How many requests are in flight at once.

    Returns:
        The completions, by custom id.
    """
    client = _client()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            custom_id: pool.submit(complete_one, custom_id, params, client)
            for custom_id, params in requests.items()
        }
        return {custom_id: future.result() for custom_id, future in futures.items()}


def run_batch(
    requests: Mapping[str, Mapping[str, Any]],
    on_submitted: Callable[[str], None] | None = None,
    batch_id: str | None = None,
    poll_seconds: float = BATCH_POLL_SECONDS,
) -> dict[str, Completion]:
    """Run requests as one message batch, at half the list price, and wait for it to end.

    Args:
        requests: The requests' parameters, by custom id. Ignored when *batch_id* is given.
        on_submitted: Called with the batch's id as soon as it exists, so a
            caller can record it and pick the batch up again after an interruption.
        batch_id: A batch submitted earlier, to wait for instead of submitting a new one.
        poll_seconds: Seconds between two looks at the batch.

    Returns:
        The completions, by custom id. A request that errored, expired or was
        cancelled comes back with ``error`` set.
    """
    client = _client()
    if batch_id is None:
        batch = client.messages.batches.create(
            requests=[
                {"custom_id": custom_id, "params": dict(params)}
                for custom_id, params in requests.items()
            ]
        )
        batch_id = batch.id
        if on_submitted is not None:
            on_submitted(batch_id)
        logger.info("Batch %s submitted with %d requests", batch_id, len(requests))
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        logger.info(
            "Batch %s: %d processing, %d succeeded", batch_id, counts.processing, counts.succeeded
        )
        time.sleep(poll_seconds)

    completions: dict[str, Completion] = {}
    for entry in client.messages.batches.results(batch_id):
        result = entry.result
        if result.type == "succeeded":
            completions[entry.custom_id] = _completion_from_message(
                entry.custom_id, result.message, batch_id
            )
        else:
            detail = result.type
            if result.type == "errored":
                detail = f"errored: {result.error.type}"
            completions[entry.custom_id] = Completion(
                custom_id=entry.custom_id, batch_id=batch_id, error=detail
            )
    return completions
