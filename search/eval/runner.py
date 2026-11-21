"""Running a generation: the waves of passage questions and the unanswerable requests.

A run lives in its own directory under the search-data directory. Its logs are
the record of everything the model was asked and answered; :mod:`search.eval.assemble`
builds the question sets from them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from search.config import GenerationSettings, UnanswerableKind
from search.corpus.table import PassageTable
from search.eval.generation import (
    PROMPT_OF_SET,
    SCHEMA_OF_PROMPT,
    Attempt,
    Prompt,
    RareWords,
    append_log,
    judge_completion,
    load_passage_texts,
    plan_wave,
    read_log,
    render_request,
    replay,
)
from search.eval.llm import Completion, cost_usd, message_params
from search.eval.sampling import Frame, Slot
from search.eval.unanswerable import SCHEMAS, UnanswerableRequest

logger = logging.getLogger("generation")

ATTEMPTS_NAME = "attempts.jsonl"
UNANSWERABLE_NAME = "unanswerable.jsonl"
PENDING_NAME = "pending_batch.txt"

#: A run stops after this many waves, whatever is left.
MAX_WAVES = 25

#: Before a budgeted run has logged a cost, each request is assumed to cost this
#: much, which is more than one passage question costs at list price.
PRIOR_REQUEST_USD = 0.03

Sender = Callable[[Mapping[str, Mapping[str, Any]]], dict[str, Completion]]


def now() -> datetime:
    """The current time in UTC, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass(frozen=True)
class RunContext:
    """What a run needs besides its slots.

    Attributes:
        directory: Where the run's logs live.
        table: The passages.
        frame: The frame the slots were drawn from.
        speeches_path: ``anforanden.parquet``, for the passages' text.
        rare_words: Which words of a passage a question may not reuse.
        prompts: The prompts, by name.
        generation: The model, its effort and its prices.
    """

    directory: Path
    table: PassageTable
    frame: Frame
    speeches_path: Path
    rare_words: RareWords
    prompts: Mapping[str, Prompt]
    generation: GenerationSettings

    def cost(self, completion: Completion) -> float:
        """What a completion cost, at the batch price when it ran in a batch."""
        discount = self.generation.batch_discount if completion.batch_id else 0.0
        return cost_usd(completion.usage, self.generation.prices[self.generation.model], discount)


def run_waves(
    context: RunContext,
    slots: Sequence[Slot],
    send: Sender,
    excluded: set[int],
    max_passages: int,
    budget_usd: float | None = None,
    spent_before: float = 0.0,
) -> list[Attempt]:
    """Send waves of requests until every slot is filled or has run out of passages.

    Args:
        context: The run's context.
        slots: The slots.
        send: Sends a wave's requests and returns the completions.
        excluded: Passages no slot may take.
        max_passages: The most passages one slot may try.
        budget_usd: The most the run may cost, counting *spent_before*; no
            wave starts once it is reached, and a wave is cut to fit what is left.
        spent_before: What earlier runs sharing the budget have spent.

    Returns:
        The whole log of the run.
    """
    log_path = context.directory / ATTEMPTS_NAME
    pending_path = context.directory / PENDING_NAME
    for wave in range(1, MAX_WAVES + 1):
        log = read_log(log_path)
        states = replay(slots, log, context.table)
        planned = plan_wave(states, context.frame, excluded, max_passages)
        if not planned:
            break
        spent = spent_before + sum(attempt.cost_usd for attempt in log)
        if budget_usd is not None:
            per_request = _cost_per_request(log, PRIOR_REQUEST_USD)
            room = int((budget_usd - spent) // per_request)
            if room < 1:
                logger.info("The budget of %.2f USD is reached after %.4f USD", budget_usd, spent)
                break
            planned = planned[:room]
        passage_ids = [context.table.passage_ids[item.passage] for item in planned]
        texts = load_passage_texts(context.speeches_path, passage_ids)
        requests: dict[str, dict[str, Any]] = {}
        details: dict[str, tuple[Slot, str, Prompt, set[str], bool]] = {}
        for item, passage_id in zip(planned, passage_ids, strict=True):
            text = texts[passage_id]
            prompt = context.prompts[PROMPT_OF_SET[item.slot.question_set]]
            system, user = render_request(
                prompt, text, context.rare_words.banned(text), item.retry_words
            )
            requests[item.custom_id] = message_params(
                model=context.generation.model,
                max_tokens=context.generation.max_tokens,
                effort=context.generation.effort,
                system=system,
                user=user,
                schema=SCHEMA_OF_PROMPT[prompt.name],
            )
            details[item.custom_id] = (
                item.slot,
                passage_id,
                prompt,
                context.rare_words.rare_stems(text),
                item.retry,
            )
        logger.info("Wave %d: %d requests", wave, len(requests))
        completions = send(requests)
        entries = []
        for custom_id, completion in completions.items():
            slot, passage_id, prompt, rare, retry = details[custom_id]
            verdict, response = judge_completion(completion, prompt.name, rare)
            entries.append(
                Attempt(
                    custom_id=custom_id,
                    slot_id=slot.slot_id,
                    passage_id=passage_id,
                    retry=retry,
                    prompt=prompt.name,
                    prompt_sha256=prompt.sha256,
                    outcome=verdict.outcome,
                    problem=verdict.problem,
                    reused=verdict.reused,
                    response=response,
                    model=completion.model,
                    usage=completion.usage,
                    cost_usd=context.cost(completion),
                    batch_id=completion.batch_id,
                    created=now(),
                )
            )
        append_log(log_path, entries)
        pending_path.unlink(missing_ok=True)
    return read_log(log_path)


def _cost_per_request(log: Sequence[Attempt], prior_usd: float) -> float:
    """The highest cost of a logged request, a cautious guess at the next one.

    Before any request has cost anything, *prior_usd* stands in.
    """
    costs = [attempt.cost_usd for attempt in log if attempt.cost_usd > 0]
    return max(costs) if costs else prior_usd


def run_unanswerable(
    context: RunContext,
    asked: Mapping[UnanswerableKind, int],
    send: Sender,
) -> list[UnanswerableRequest]:
    """Ask for unanswerable questions, one request per kind not yet answered in this run.

    Args:
        context: The run's context.
        asked: How many questions to ask for, by kind.
        send: Sends the requests and returns the completions.

    Returns:
        The run's whole log of unanswerable requests.
    """
    log_path = context.directory / UNANSWERABLE_NAME
    done = {request.kind for request in read_log(log_path, UnanswerableRequest) if request.response}
    requests: dict[str, dict[str, Any]] = {}
    for kind, count in asked.items():
        if kind in done:
            continue
        prompt = context.prompts[kind]
        requests[f"unanswerable-{kind.replace('_', '-')}"] = message_params(
            model=context.generation.model,
            max_tokens=context.generation.max_tokens,
            effort=context.generation.effort,
            system=prompt.system.format(count=count),
            user=prompt.user.format(count=count),
            schema=SCHEMAS[kind],
        )
    if not requests:
        return read_log(log_path, UnanswerableRequest)
    kinds = {f"unanswerable-{kind.replace('_', '-')}": kind for kind in asked}
    completions = send(requests)
    entries = [
        UnanswerableRequest(
            custom_id=custom_id,
            kind=kinds[custom_id],
            asked=asked[kinds[custom_id]],
            prompt_sha256=context.prompts[kinds[custom_id]].sha256,
            response=completion.text,
            model=completion.model,
            stop_reason=completion.stop_reason,
            error=completion.error,
            usage=completion.usage,
            cost_usd=context.cost(completion),
            batch_id=completion.batch_id,
            created=now(),
        )
        for custom_id, completion in completions.items()
    ]
    append_log(log_path, entries)
    return read_log(log_path, UnanswerableRequest)


def speaker_names(speeches_path: Path) -> list[str]:
    """Every distinct speaker in the corpus as published, lower-case."""
    column = pq.read_table(speeches_path, columns=["speaker_raw"]).column("speaker_raw")
    return sorted({name.lower() for name in column.to_pylist()})
