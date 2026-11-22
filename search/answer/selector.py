"""The two selectors: the baseline, without a language model, and Claude.

Both take a question and its candidates and answer in the same form
(:class:`search.answer.models.SelectorOutput`), as JSON text the assembler
reads. Neither writes any text of its own.

The baseline selector quotes the first candidate, from the first to the last
of its paragraphs that hold a word of the question outside Snowball's stop
words, compared after stemming; when no paragraph holds one, it quotes the
whole passage. It abstains when the first candidate scores below a share of
the most the question's terms can score: ``k1 + 1`` times the sum of their
idf, a term the index lacks counting with the idf of a term no passage holds.
The share is chosen on the development split by :func:`choose_threshold`.

The language model gets the question and the candidates with numbered
paragraphs, and answers in structured output. A request that fails, is
refused or is cut off is sent once more; when that fails too, the selector
returns no output, and the caller lets the baseline answer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from search.answer.models import Candidate, Selected
from search.config import Prices
from search.eval.generation import Prompt, load_prompt
from search.eval.llm import Completion, Usage, complete_one, cost_usd, message_params
from search.index.analyzer import Analyzer
from search.index.lexical import Bm25Params, LexicalIndex

#: The directory of the selector's prompt, ``prompt.md``.
PROMPT_DIR: Path = Path(__file__).resolve().parent
PROMPT_NAME = "prompt"

#: The words of a question the baseline looks for in a paragraph: stemmed, stop words dropped.
QUESTION_WORDS = Analyzer(stem=True, stopwords=True)

#: An answer that quotes nothing.
ABSTENTION = json.dumps({"answerable": False, "selections": []})

#: Stop reasons after which a request is sent once more.
RETRIED_STOPS = frozenset({"refusal", "max_tokens"})

#: The JSON schema of the language model's answer. It is the same for every
#: question, so the API compiles it once; the assembler checks the ids.
SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "passage_id": {"type": "string"},
                    "first_paragraph": {"type": "integer"},
                    "last_paragraph": {"type": "integer"},
                },
                "required": ["passage_id", "first_paragraph", "last_paragraph"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "selections"],
    "additionalProperties": False,
}


def load_selector_prompt(directory: Path = PROMPT_DIR) -> Prompt:
    """Read the selector's prompt, ``prompt.md``, with its hash."""
    return load_prompt(PROMPT_NAME, directory)


def selection_output(passage_id: str, first: int, last: int) -> str:
    """An answer that quotes one range of one passage, as JSON text."""
    return json.dumps(
        {
            "answerable": True,
            "selections": [
                {"passage_id": passage_id, "first_paragraph": first, "last_paragraph": last}
            ],
        }
    )


def trimmed_span(question: str, paragraphs: Sequence[str]) -> tuple[int, int]:
    """The range the baseline quotes from a passage, numbered from 1.

    Args:
        question: The question.
        paragraphs: The passage's paragraphs.

    Returns:
        The first and the last paragraph that hold a word of the question
        outside the stop words, compared after stemming, or the whole passage
        when none does.
    """
    words = set(QUESTION_WORDS.terms(question))
    holding = [
        number
        for number, paragraph in enumerate(paragraphs, start=1)
        if words & set(QUESTION_WORDS.terms(paragraph))
    ]
    if not holding:
        return 1, len(paragraphs)
    return holding[0], holding[-1]


class BaselineSelector:
    """Quote the best candidate's best paragraphs, or abstain on a low score.

    Attributes:
        index: The BM25 index the candidates were ranked in.
        params: The ``k1`` and ``b`` they were ranked with.
        threshold: The share of the question's highest possible score below
            which the selector abstains.
    """

    name = "baseline"

    def __init__(self, index: LexicalIndex, params: Bm25Params, threshold: float) -> None:
        self.index = index
        self.params = params
        self.threshold = threshold

    def ceiling(self, question: str) -> float:
        """The most a passage can score for *question*: ``k1 + 1`` times the sum of its terms' idf.

        Each distinct term counts once, as the index scores it; a term no
        passage holds counts with the idf of a document frequency of 0.
        """
        terms = dict.fromkeys(self.index.analyzer.terms(question))
        return (self.params.k1 + 1) * sum(self.index.idf(term) for term in terms)

    def relative_score(self, question: str, candidates: Sequence[Candidate]) -> float | None:
        """The first candidate's score as a share of the ceiling, or ``None`` without candidates."""
        if not candidates:
            return None
        ceiling = self.ceiling(question)
        return candidates[0].score / ceiling if ceiling > 0 else 0.0

    def select(self, question: str, candidates: Sequence[Candidate]) -> Selected:
        """Answer one question.

        Args:
            question: The question.
            candidates: Its candidates, best first.

        Returns:
            The answer, which quotes at most the first candidate.
        """
        score = self.relative_score(question, candidates)
        if score is None or score < self.threshold:
            return Selected(selector="baseline", output=ABSTENTION)
        best = candidates[0]
        first, last = trimmed_span(question, best.paragraphs)
        return Selected(selector="baseline", output=selection_output(best.passage_id, first, last))


def choose_threshold(
    answerable: Sequence[float | None], unanswerable: Sequence[float | None]
) -> tuple[float, float]:
    """Pick the baseline's threshold by balanced accuracy on a split.

    A question scores ``None`` when it has no candidates; it is abstained on
    at every threshold. The values tried are 0 and every score observed: a
    value between two observed scores abstains on the same questions as the
    next one up. Balanced accuracy is compared exactly, as a fraction, and a
    tie goes to the lowest value.

    Args:
        answerable: The relative score of each answerable question.
        unanswerable: The relative score of each unanswerable question.

    Returns:
        The threshold and the balanced accuracy it reaches.

    Raises:
        ValueError: If either list is empty.
    """
    if not answerable or not unanswerable:
        raise ValueError("The threshold needs answerable and unanswerable questions.")
    observed = {score for score in (*answerable, *unanswerable) if score is not None}
    best: tuple[float, Fraction] | None = None
    for threshold in sorted({0.0, *observed}):
        abstains = sum(score is None or score < threshold for score in unanswerable)
        answers = sum(score is not None and score >= threshold for score in answerable)
        accuracy = (Fraction(abstains, len(unanswerable)) + Fraction(answers, len(answerable))) / 2
        if best is None or accuracy > best[1]:
            best = (threshold, accuracy)
    assert best is not None
    return best[0], float(best[1])


def render_candidates(candidates: Sequence[Candidate]) -> str:
    """The candidates as the prompt shows them: an id, the speech's fields, numbered paragraphs."""
    blocks = []
    for candidate in candidates:
        lines = [
            f'<utdrag id="{candidate.passage_id}">',
            f"Talare: {candidate.speaker}",
            f"Parti: {candidate.party}",
            f"Datum: {candidate.date.isoformat()}",
        ]
        if candidate.title:
            lines.append(f"Debatt: {candidate.title}")
        lines.extend(
            f"[{number}] {paragraph}"
            for number, paragraph in enumerate(candidate.paragraphs, start=1)
        )
        lines.append("</utdrag>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def failed(completion: Completion) -> bool:
    """Whether a request failed, was refused or was cut off, and is sent once more."""
    return (
        completion.error is not None
        or completion.stop_reason in RETRIED_STOPS
        or completion.text is None
    )


class ClaudeSelector:
    """Selection by a language model, in structured output.

    Attributes:
        prompt: The prompt, with its hash.
        model: The model id.
        effort: How much the model thinks.
        max_tokens: The most output tokens, thinking included.
        prices: The model's list prices.
    """

    name = "claude"

    def __init__(
        self, prompt: Prompt, model: str, effort: str, max_tokens: int, prices: Prices
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.prices = prices

    def request(self, question: str, candidates: Sequence[Candidate]) -> dict[str, Any]:
        """The parameters of the request for one question."""
        return message_params(
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            system=self.prompt.system,
            user=self.prompt.user.format(
                question=question, candidates=render_candidates(candidates)
            ),
            schema=SELECTOR_SCHEMA,
        )

    def read(self, completions: Sequence[Completion], batch_discount: float = 0.0) -> Selected:
        """Turn the completions of one question, in the order sent, into the selector's answer.

        The last completion decides. When it failed, the answer has no output,
        and the caller lets the baseline answer.

        Args:
            completions: One completion, or two when the first failed.
            batch_discount: The share taken off the price, for a message batch.

        Returns:
            The answer, with the tokens and cost of every request.
        """
        last = completions[-1]
        usage = Usage(
            input_tokens=sum(item.usage.input_tokens for item in completions),
            output_tokens=sum(item.usage.output_tokens for item in completions),
            cache_creation_input_tokens=sum(
                item.usage.cache_creation_input_tokens for item in completions
            ),
            cache_read_input_tokens=sum(item.usage.cache_read_input_tokens for item in completions),
        )
        cost = sum(cost_usd(item.usage, self.prices, batch_discount) for item in completions)
        return Selected(
            selector="claude",
            output=None if failed(last) else last.text,
            model=last.model,
            stop_reason=last.stop_reason,
            error=last.error,
            attempts=len(completions),
            usage=usage,
            cost_usd=cost,
        )

    def select(
        self, question: str, candidates: Sequence[Candidate], client: Any | None = None
    ) -> Selected:
        """Answer one question with one request, sent once more if it fails.

        A question without candidates is abstained on without a request.

        Args:
            question: The question.
            candidates: Its candidates, best first.
            client: The SDK client; a new one when ``None``.

        Returns:
            The answer; without output when both requests failed.
        """
        if not candidates:
            return Selected(selector="claude", output=ABSTENTION)
        params = self.request(question, candidates)
        completions = [complete_one("answer", params, client)]
        if failed(completions[0]):
            completions.append(complete_one("answer", params, client))
        return self.read(completions)


def fall_back(failure: Selected, baseline: Selected, reason: str) -> Selected:
    """The baseline's answer in place of the language model's, keeping what the requests cost.

    Args:
        failure: The language model's answer, without output or not sent.
        baseline: The baseline's answer to the same question.
        reason: ``budget`` or ``failed``.

    Returns:
        The baseline's output, marked with the reason and carrying the
        language model's requests, tokens and cost.
    """
    return Selected.model_validate(
        {
            **failure.model_dump(),
            "selector": "baseline",
            "output": baseline.output,
            "fallback": reason,
        }
    )
