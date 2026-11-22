"""The budget guard: once the month's budget for the model is spent, the baseline answers.

This is the soft cap in the service. The hard cap belongs at the provider: a
spending limit on the project's own key in Anthropic's console, the only cap
that holds if this code is wrong. The guard counts what the language model's requests cost in the
current calendar month, in UTC. While that is below the monthly budget the
language model selects; from then on the baseline does, and the answer says
so. A request that fails twice is answered by the baseline as well.

The guard keeps its count in memory, or in a small JSON file when one is
given, so that a restart does not reset the month.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from search.answer.assemble import assemble
from search.answer.models import Answer, Candidate, Selected, Speech
from search.answer.selector import BaselineSelector, ClaudeSelector, fall_back
from search.eval.runner import now


class BudgetGuard:
    """What the language model has cost this month, against the monthly budget.

    Attributes:
        monthly_usd: The budget for one calendar month, in US dollars.
        ledger: The file the count is kept in, or ``None`` to keep it in memory.
    """

    def __init__(
        self,
        monthly_usd: float,
        ledger: Path | None = None,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.monthly_usd = monthly_usd
        self.ledger = ledger
        self._clock = clock
        self._lock = threading.Lock()
        self._month = ""
        self._spent = 0.0
        if ledger is not None and ledger.is_file():
            saved = json.loads(ledger.read_text(encoding="utf-8"))
            self._month = str(saved["month"])
            self._spent = float(saved["spent_usd"])

    def _roll(self) -> None:
        """Start the count again when a new month has begun."""
        month = self._clock().strftime("%Y-%m")
        if month != self._month:
            self._month = month
            self._spent = 0.0

    def spent(self) -> float:
        """What the language model has cost in the current month."""
        with self._lock:
            self._roll()
            return self._spent

    def allows(self) -> bool:
        """Whether the month's budget still has room for a request."""
        return self.spent() < self.monthly_usd

    def record(self, cost_usd: float) -> None:
        """Add the cost of a request to the current month."""
        with self._lock:
            self._roll()
            self._spent += cost_usd
            if self.ledger is not None:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                self.ledger.write_text(
                    json.dumps({"month": self._month, "spent_usd": self._spent}) + "\n",
                    encoding="utf-8",
                )


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

    def select(self, question: str, candidates: Sequence[Candidate]) -> Selected:
        """Answer one question with the language model, or with the baseline and why.

        Args:
            question: The question.
            candidates: Its candidates, best first.

        Returns:
            The answer. When the baseline answered, ``fallback`` says why.
        """
        if not candidates:
            return self.claude.select(question, candidates)
        if not self.guard.allows():
            return fall_back(
                Selected(selector="claude", output=None),
                self.baseline.select(question, candidates),
                "budget",
            )
        selected = self.claude.select(question, candidates, self._client)
        self.guard.record(selected.cost_usd)
        if selected.output is None:
            return fall_back(selected, self.baseline.select(question, candidates), "failed")
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
