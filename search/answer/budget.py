"""The budget guard: once the month's budget for the model is spent, the baseline answers.

The guard counts what the language model's requests cost in the current
calendar month, in UTC, in a ledger that every instance of the service shares.
While that is below the monthly budget the language model selects; from then
on the baseline does, and the answer says so. A request that fails twice is
answered by the baseline as well.

Each request to the model is checked against the ledger as it stands just
before the request is sent, and its cost is added as soon as it returns, in an
update the ledger refuses when another writer came first, so no instance's
count overwrites another's. The month's spend can pass the budget only by the
requests already sent when the budget is reached: one per instance, since an
instance runs one selector at a time. A request that came back without its
usage after it may have been sent, because the connection broke or timed out,
is counted at the most it could have cost.

When the ledger cannot be read, or holds anything but a month and a sum, the
guard refuses the model and the baseline answers. When a cost cannot be written
after a paid answer, the answer still stands, and that guard refuses the model
from then on.

This is the soft cap in the service. The hard cap belongs at the provider: a
spending limit on the workspace of the service's key in Anthropic's console,
the only cap that holds if this code is wrong.

A ledger is an object in Cloud Storage, written only on the generation it was
read at, for the service; a file, compared and replaced under a lock, for one
process; or memory.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from search.answer import selector as selector_module
from search.answer.assemble import assemble
from search.answer.models import Answer, Candidate, Selected, Speech
from search.answer.selector import BaselineSelector, ClaudeSelector, failed, fall_back
from search.eval.llm import Completion, Usage, cost_usd
from search.eval.runner import now

logger = logging.getLogger(__name__)

#: How many requests one question may send: one, and once more if it fails.
ATTEMPTS = 2

#: How many times an update is tried while other writers keep coming first.
WRITE_ATTEMPTS = 8


class LedgerError(RuntimeError):
    """The ledger could not be read or written, or holds something else."""


class Ledger(Protocol):
    """Where the month's spend is kept: read with a version, written only on that version."""

    def read(self) -> tuple[bytes | None, Any]:
        """The ledger's content, or ``None`` when there is none yet, and its version."""
        ...

    def write(self, data: bytes, version: Any) -> bool:
        """Replace the content if it is still at *version*.

        Returns:
            ``False`` when another writer came first, and nothing was written.
        """
        ...


class MemoryLedger:
    """A ledger in memory, for one process; a restart starts the month again."""

    def __init__(self) -> None:
        self._data: bytes | None = None
        self._version = 0
        self._lock = threading.Lock()

    def read(self) -> tuple[bytes | None, int]:
        """The content and its version."""
        with self._lock:
            return self._data, self._version

    def write(self, data: bytes, version: int) -> bool:
        """Replace the content if no one has written it since *version*."""
        with self._lock:
            if version != self._version:
                return False
            self._data, self._version = data, self._version + 1
            return True


_FILE_LOCKS: dict[Path, threading.Lock] = {}
_FILE_LOCKS_LOCK = threading.Lock()


class FileLedger:
    """A JSON file, for one process: compared and replaced under a lock per file.

    The file's directory must exist; the ledger never makes it, so that a
    missing mount fails loudly instead of counting in the container.

    Attributes:
        path: The file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        with _FILE_LOCKS_LOCK:
            self._lock = _FILE_LOCKS.setdefault(path.resolve(), threading.Lock())

    def read(self) -> tuple[bytes | None, bytes | None]:
        """The content, which is also its version."""
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return None, None
        except OSError as error:
            raise LedgerError(f"The ledger {self.path} cannot be read.") from error
        return data, data

    def write(self, data: bytes, version: bytes | None) -> bool:
        """Replace the file if it still holds *version*."""
        with self._lock:
            if self.read()[0] != version:
                return False
            temporary = self.path.with_name(self.path.name + ".writing")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, self.path)
            except OSError as error:
                raise LedgerError(f"The ledger {self.path} cannot be written.") from error
            return True


class GcsLedger:
    """An object in Cloud Storage, written only on the generation it was read at.

    The service account's token comes from the metadata server of the instance
    the service runs on.

    Attributes:
        bucket: The bucket.
        name: The object's name.
    """

    API = "https://storage.googleapis.com"
    TOKEN_URL = (
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    )

    def __init__(
        self, bucket: str, name: str, client: httpx.Client | None = None, timeout_s: float = 5.0
    ) -> None:
        self.bucket = bucket
        self.name = name
        self._client = client or httpx.Client(timeout=timeout_s)
        self._token: str | None = None
        self._expires = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_url(cls, url: str, client: httpx.Client | None = None) -> GcsLedger:
        """The ledger at ``gs://bucket/name``.

        Raises:
            ValueError: If the URL names no bucket and object.
        """
        bucket, _, name = url.removeprefix("gs://").partition("/")
        if not url.startswith("gs://") or not bucket or not name:
            raise ValueError(f"{url} is not gs://bucket/object.")
        return cls(bucket, name, client)

    def _authorization(self) -> dict[str, str]:
        with self._lock:
            if self._token is None or time.monotonic() >= self._expires:
                try:
                    response = self._client.get(
                        self.TOKEN_URL, headers={"Metadata-Flavor": "Google"}
                    )
                    response.raise_for_status()
                    body = response.json()
                    self._token = str(body["access_token"])
                    self._expires = time.monotonic() + float(body["expires_in"]) - 60
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                    raise LedgerError("No token for the ledger's bucket.") from error
            return {"Authorization": f"Bearer {self._token}"}

    def read(self) -> tuple[bytes | None, int]:
        """The object's content and generation; generation 0 when there is no object yet."""
        url = f"{self.API}/storage/v1/b/{self.bucket}/o/{quote(self.name, safe='')}"
        try:
            response = self._client.get(url, params={"alt": "media"}, headers=self._authorization())
        except httpx.HTTPError as error:
            raise LedgerError("The ledger cannot be read.") from error
        if response.status_code == 404:
            return None, 0
        if response.status_code != 200:
            raise LedgerError(f"The ledger cannot be read: HTTP {response.status_code}.")
        try:
            return response.content, int(response.headers["x-goog-generation"])
        except (KeyError, ValueError) as error:
            raise LedgerError("The ledger came back without its generation.") from error

    def write(self, data: bytes, version: int) -> bool:
        """Write the object if it is still at generation *version*."""
        try:
            response = self._client.post(
                f"{self.API}/upload/storage/v1/b/{self.bucket}/o",
                params={"uploadType": "media", "name": self.name, "ifGenerationMatch": version},
                content=data,
                headers={**self._authorization(), "Content-Type": "application/json"},
            )
        except httpx.HTTPError as error:
            raise LedgerError("The ledger cannot be written.") from error
        if response.status_code == 412:
            return False
        if response.status_code != 200:
            raise LedgerError(f"The ledger cannot be written: HTTP {response.status_code}.")
        return True


def ledger_from(value: Ledger | Path | str | None) -> Ledger:
    """The ledger *value* names: ``gs://bucket/object``, a file, a ledger, or memory."""
    if value is None:
        return MemoryLedger()
    if isinstance(value, str) and value.startswith("gs://"):
        return GcsLedger.from_url(value)
    if isinstance(value, str | Path):
        return FileLedger(Path(value))
    return value


class BudgetGuard:
    """What the language model has cost this month, against the monthly budget.

    Every reading goes to the ledger, so instances that share it count together.

    Attributes:
        monthly_usd: The budget for one calendar month, in US dollars.
        ledger: Where the count is kept.
    """

    def __init__(
        self,
        monthly_usd: float,
        ledger: Ledger | Path | str | None = None,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.monthly_usd = monthly_usd
        self.ledger = ledger_from(ledger)
        self._clock = clock

    def _read(self) -> tuple[str, float, Any]:
        data, version = self.ledger.read()
        if data is None:
            return "", 0.0, version
        try:
            saved = json.loads(data)
            month, spent = str(saved["month"]), float(saved["spent_usd"])
        except (ValueError, KeyError, TypeError) as error:
            raise LedgerError("The ledger does not hold a month and a sum.") from error
        if not math.isfinite(spent) or spent < 0:
            raise LedgerError("The ledger's sum is not a cost.")
        return month, spent, version

    def spent(self) -> float:
        """What the language model has cost in the current month.

        Raises:
            LedgerError: If the ledger cannot be read.
        """
        month, spent, _ = self._read()
        return spent if month == self._clock().strftime("%Y-%m") else 0.0

    def allows(self) -> bool:
        """Whether the month's budget still has room for a request.

        Raises:
            LedgerError: If the ledger cannot be read.
        """
        return self.spent() < self.monthly_usd

    def record(self, cost_usd: float) -> None:
        """Add the cost of a request to the current month, in one update of the ledger.

        Raises:
            LedgerError: If the ledger cannot be read or written, or other
                writers keep coming first.
        """
        for _ in range(WRITE_ATTEMPTS):
            month, spent, version = self._read()
            current = self._clock().strftime("%Y-%m")
            total = (spent if month == current else 0.0) + cost_usd
            data = (json.dumps({"month": current, "spent_usd": total}) + "\n").encode("utf-8")
            if self.ledger.write(data, version):
                return
        raise LedgerError("Other writers kept coming first.")


def most_cost(selector: ClaudeSelector, params: Mapping[str, Any]) -> float:
    """The most one request can cost: every byte of it an input token, and the most output.

    A token holds at least one byte of the text it stands for, so the request's
    length in bytes bounds its input tokens.
    """
    request_bytes = len(json.dumps(params, ensure_ascii=False).encode("utf-8"))
    usage = Usage(input_tokens=request_bytes, output_tokens=selector.max_tokens)
    return cost_usd(usage, selector.prices)


class GuardedSelector:
    """The selector the page runs: the language model within the budget, the baseline beyond it.

    Attributes:
        claude: The language model's selector.
        baseline: The baseline selector.
        guard: The budget guard.
    """

    def __init__(
        self,
        claude: ClaudeSelector,
        baseline: BaselineSelector,
        guard: BudgetGuard,
        client: Any | None = None,
    ) -> None:
        self.claude = claude
        self.baseline = baseline
        self.guard = guard
        self._client = client
        self._unrecorded = False

    def _allows(self) -> bool:
        if self._unrecorded:
            return False
        try:
            return self.guard.allows()
        except LedgerError as error:
            logger.error("The budget's ledger cannot be read, so the baseline answers: %s", error)
            return False

    def cost(self, completion: Completion, params: Mapping[str, Any]) -> float:
        """What one request cost, or the most it could have cost when it came back without usage.

        A request the API refused with a status generated nothing. One that
        broke off or timed out may have been answered and billed.
        """
        usage = completion.usage
        if usage.input_tokens or usage.output_tokens:
            return cost_usd(usage, self.claude.prices)
        if completion.error is not None and completion.error.startswith("HTTP "):
            return 0.0
        return most_cost(self.claude, params)

    def _count(self, completion: Completion, params: Mapping[str, Any]) -> None:
        try:
            self.guard.record(self.cost(completion, params))
        except LedgerError as error:
            self._unrecorded = True
            logger.error(
                "A cost could not be written to the budget's ledger, so the baseline answers "
                "from now on: %s",
                error,
            )

    def select(self, question: str, candidates: Sequence[Candidate]) -> Selected:
        """Answer one question with the language model, or with the baseline and why.

        The budget is checked before each request, and each request's cost is
        counted as it returns.

        Args:
            question: The question.
            candidates: Its candidates, best first.

        Returns:
            The answer. When the baseline answered, ``fallback`` says why.
        """
        if not candidates:
            return self.claude.select(question, candidates)
        params = self.claude.request(question, candidates)
        completions: list[Completion] = []
        refused = False
        while len(completions) < ATTEMPTS:
            if not self._allows():
                refused = True
                break
            # Sent through the selector's module, as ClaudeSelector.select sends
            # it, so that one stand-in for the request serves both.
            completion = selector_module.complete_one("answer", params, self._client)
            completions.append(completion)
            self._count(completion, params)
            if not failed(completion):
                break
        selected = (
            self.claude.read(completions)
            if completions
            else Selected(selector="claude", output=None)
        )
        if selected.output is None:
            reason = "budget" if refused else "failed"
            return fall_back(selected, self.baseline.select(question, candidates), reason)
        return selected


def answer(
    question: str,
    candidates: Sequence[Candidate],
    speeches: Mapping[str, Speech],
    selector: GuardedSelector | BaselineSelector,
) -> Answer:
    """Answer one question: select, then assemble.

    Args:
        question: The question.
        candidates: Its candidates, best first.
        speeches: The speeches the candidates belong to, by key.
        selector: The selector.

    Returns:
        The answer.
    """
    return assemble(selector.select(question, candidates), candidates, speeches)
