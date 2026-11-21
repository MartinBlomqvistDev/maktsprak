"""Writing questions with a language model: prompts, checks and the order of the work.

Generation runs in waves. In each wave every slot that still lacks a question
gets one request, for the passage it is working on. The answer is checked in
code, and the outcome is appended to a log, which is the only state: a run that
stops can be started again and goes on from the log.

A slot keeps its passage for one more request when the question reused a rare
word, broke a rule of form or was cut off at the token limit; a request after a
rare word names the words. A second failure on the same passage, a passage the
model finds no question in, or a refusal moves the slot to the next passage of
its stratum (see :mod:`search.eval.sampling`), up to a fixed number of passages
per slot. An API error repeats the same request in the next wave.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, ValidationError

from search.corpus.table import PassageTable
from search.eval.llm import Completion, Usage
from search.eval.sampling import Frame, Slot, candidates
from search.index.analyzer import STEMMING, stem, tokenize
from search.index.lexical import LexicalIndex

PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

#: The prompt each kind of slot is written from. A filtered question is a natural question.
PROMPT_OF_SET: Mapping[str, str] = {"natural": "natural", "filtered": "natural", "stance": "stance"}

#: A question has at most this many words, split on whitespace.
MAX_QUESTION_WORDS = 25

#: A keyword question has from two to four search terms.
KEYWORDS_MIN, KEYWORDS_MAX = 2, 4

#: The opening a stance question must have, by direction.
STANCE_OPENINGS: Mapping[str, str] = {
    "against": "Vilka argument har förts fram mot ",
    "for": "Vilka argument har förts fram för ",
}

#: The direction words the model answers with, in Swedish, mapped to the stored direction.
STANCE_DIRECTIONS: Mapping[str, str] = {"mot": "against", "för": "for"}

#: An API error on the same request this many times moves the slot on.
MAX_ERRORS_PER_PASSAGE = 3

#: The outcomes after which a slot asks once more about the same passage.
RETRIED_OUTCOMES = frozenset({"rare", "invalid", "truncated"})

Outcome = Literal["filled", "rare", "unusable", "invalid", "refused", "truncated", "error"]

_YEAR = re.compile(r"\b(?:19|20)\d\d\b")


@dataclass(frozen=True)
class Prompt:
    """A prompt file: the system prompt, the user message and the line added on a retry.

    Attributes:
        name: The file's name without ``.md``.
        system: The system prompt, which may hold ``{placeholders}``.
        user: The user message, with ``{placeholders}``.
        retry: The line added to the user message on a retry; may be empty.
        sha256: The hash of the file's bytes, stored with every question written from it.
    """

    name: str
    system: str
    user: str
    retry: str
    sha256: str


def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> Prompt:
    """Read a prompt file of three sections: system, user and retry.

    Each section opens with a line of its own: ``=== system ===``,
    ``=== user ===`` and ``=== retry ===``, in that order. Windows line ends are
    read as Unix ones, and the hash is taken of the text with Unix line ends, so
    a checkout that converts line ends gives the same prompt and the same hash.

    Args:
        name: The file's name without ``.md``.
        directory: Where the prompt files are.

    Returns:
        The prompt.

    Raises:
        ValueError: If the sections are missing or out of order.
    """
    text = (directory / f"{name}.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    markers = ["=== system ===", "=== user ===", "=== retry ==="]
    try:
        starts = [lines.index(marker) for marker in markers]
    except ValueError as error:
        raise ValueError(f"{name}.md must have the sections {markers}.") from error
    if starts != sorted(starts) or starts[0] != 0:
        raise ValueError(f"{name}.md must open with {markers[0]} and keep the sections in order.")
    ends = [*starts[1:], len(lines)]
    system, user, retry = (
        "\n".join(lines[start + 1 : end]).strip() for start, end in zip(starts, ends, strict=True)
    )
    return Prompt(
        name=name,
        system=system,
        user=user,
        retry=retry,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


class RareWords:
    """Which words of a passage a question written from it may not reuse.

    A word is rare when its stem occurs in fewer passages than a share of the
    corpus, counted in the index built with stemming and without stop words.
    The check compares stems, so another inflection of a rare word is caught too.

    Attributes:
        index: The stemmed index, used for the document frequencies.
        below: A stem held by fewer passages than this is rare.
    """

    def __init__(self, index: LexicalIndex, share: float) -> None:
        if index.analyzer != STEMMING:
            raise ValueError("Rare words are counted in the index with stemming and no stop words.")
        self.index = index
        self.below = share * index.passages

    def rare_stems(self, text: str) -> set[str]:
        """The rare stems of a text."""
        return {
            term
            for term in STEMMING.terms(text)
            if self.index.document_frequency(term) < self.below
        }

    def banned(self, text: str) -> list[str]:
        """The rare words of a text as written there, lower-case, each form once, in order."""
        rare = self.rare_stems(text)
        return list(dict.fromkeys(token for token in tokenize(text) if stem(token) in rare))

    @staticmethod
    def reused(question: str, rare: set[str]) -> list[str]:
        """The words of a question whose stems are among *rare*, as written in the question."""
        return list(dict.fromkeys(token for token in tokenize(question) if stem(token) in rare))


#: The answer to the natural-question prompt.
PASSAGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "reason": {"type": "string"},
        "question": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["usable", "reason", "question", "keywords"],
    "additionalProperties": False,
}

#: The answer to the stance prompt.
STANCE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "reason": {"type": "string"},
        "direction": {"type": "string", "enum": ["för", "mot", "ingen"]},
        "question": {"type": "string"},
    },
    "required": ["usable", "reason", "direction", "question"],
    "additionalProperties": False,
}

SCHEMA_OF_PROMPT: Mapping[str, Mapping[str, Any]] = {
    "natural": PASSAGE_SCHEMA,
    "stance": STANCE_SCHEMA,
}


class PassageAnswer(BaseModel):
    """What the model answers to the natural-question prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    usable: bool
    reason: str
    question: str
    keywords: list[str]


class StanceAnswer(BaseModel):
    """What the model answers to the stance prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    usable: bool
    reason: str
    direction: Literal["för", "mot", "ingen"]
    question: str


@dataclass(frozen=True)
class Verdict:
    """The code's judgement of one answer.

    Attributes:
        outcome: ``filled``, ``rare``, ``unusable`` or ``invalid``.
        problem: What was wrong, for any outcome but ``filled``.
        reused: The rare words the question reused, for ``rare``.
    """

    outcome: Outcome
    problem: str | None = None
    reused: tuple[str, ...] = ()


def is_single_word(word: str) -> bool:
    """Tell whether a keyword is one word: no whitespace, and a letter or digit in it.

    A word is a run of characters between whitespace, as everywhere else in the
    corpus, so a hyphenated word such as "a-kassan" or "EU-medborgare" is one word.
    """
    return len(word.split()) == 1 and word == word.strip() and bool(tokenize(word))


def _question_problem(question: str) -> str | None:
    """Say what is wrong with the form of a question, or ``None``."""
    if not question or question != question.strip() or "\n" in question:
        return "the question is empty or not one line"
    if not question.endswith("?"):
        return "the question does not end with a question mark"
    if len(question.split()) > MAX_QUESTION_WORDS:
        return f"the question has more than {MAX_QUESTION_WORDS} words"
    return None


def check_passage_answer(answer: PassageAnswer, rare: set[str]) -> Verdict:
    """Check a natural question and its keywords.

    Args:
        answer: The model's answer.
        rare: The rare stems of the passage.

    Returns:
        The verdict.
    """
    if not answer.usable:
        return Verdict("unusable", answer.reason)
    problem = _question_problem(answer.question)
    if problem:
        return Verdict("invalid", problem)
    keywords = answer.keywords
    if not KEYWORDS_MIN <= len(keywords) <= KEYWORDS_MAX:
        return Verdict("invalid", f"{len(keywords)} keywords, not {KEYWORDS_MIN} to {KEYWORDS_MAX}")
    if not all(is_single_word(word) for word in keywords):
        return Verdict("invalid", "a keyword is not a single word")
    if len({word.lower() for word in keywords}) != len(keywords):
        return Verdict("invalid", "a keyword is repeated")
    reused = RareWords.reused(answer.question, rare)
    if reused:
        return Verdict("rare", "the question reuses rare words", tuple(reused))
    return Verdict("filled")


def check_stance_answer(answer: StanceAnswer, rare: set[str]) -> Verdict:
    """Check a stance question.

    Args:
        answer: The model's answer.
        rare: The rare stems of the passage.

    Returns:
        The verdict.
    """
    if not answer.usable or answer.direction == "ingen":
        return Verdict("unusable", answer.reason)
    problem = _question_problem(answer.question)
    if problem:
        return Verdict("invalid", problem)
    opening = STANCE_OPENINGS[STANCE_DIRECTIONS[answer.direction]]
    if not answer.question.startswith(opening) or len(answer.question) <= len(opening) + 1:
        return Verdict("invalid", f"the question does not open with {opening.strip()!r}")
    reused = RareWords.reused(answer.question, rare)
    if reused:
        return Verdict("rare", "the question reuses rare words", tuple(reused))
    return Verdict("filled")


class Attempt(BaseModel):
    """One request for a slot and what came of it; a line of the generation log.

    Attributes:
        custom_id: The request's id, unique within a run.
        slot_id: The slot it was for.
        passage_id: The passage it was written from.
        retry: Whether the request was the second one about its passage.
        prompt: The prompt file's name.
        prompt_sha256: The prompt file's hash.
        outcome: What came of it.
        problem: What was wrong, when something was.
        reused: The rare words the question reused, for ``rare``.
        response: The model's answer, as returned.
        model: The model that answered.
        usage: The tokens used.
        cost_usd: What the request cost.
        batch_id: The message batch it ran in, if any.
        created: When the answer was recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str
    slot_id: str
    passage_id: str
    retry: bool
    prompt: str
    prompt_sha256: str
    outcome: Outcome
    problem: str | None = None
    reused: tuple[str, ...] = ()
    response: str | None = None
    model: str | None = None
    usage: Usage = Usage()
    cost_usd: float = 0.0
    batch_id: str | None = None
    created: datetime


def read_log(path: Path, model: type[BaseModel] = Attempt) -> list[Any]:
    """Read a JSON Lines log; a missing file is an empty log."""
    if not path.is_file():
        return []
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_log(path: Path, entries: Iterable[BaseModel]) -> None:
    """Append entries to a JSON Lines log, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")


@dataclass
class SlotState:
    """Where one slot stands, rebuilt from the log.

    Attributes:
        slot: The slot.
        filled: The attempt that filled it, once one has.
        passages: The passages it has tried, in order, as index positions.
        current: The passage it is still working on, if any.
        retrying: Whether the next request is the second one about the current passage.
        retry_words: The rare words the next request must name, on a retry.
        errors: API errors in a row on the current request.
        attempts: How many requests it has made.
    """

    slot: Slot
    filled: Attempt | None = None
    passages: list[int] = field(default_factory=list)
    current: int | None = None
    retrying: bool = False
    retry_words: tuple[str, ...] = ()
    errors: int = 0
    attempts: int = 0

    def _drop_current(self) -> None:
        self.current, self.retrying, self.retry_words, self.errors = None, False, (), 0


def replay(
    slots: Sequence[Slot], attempts: Iterable[Attempt], table: PassageTable
) -> dict[str, SlotState]:
    """Rebuild every slot's state from the log.

    Args:
        slots: The slots of the run.
        attempts: The log, in the order it was written.
        table: The passages, to turn ids into positions.

    Returns:
        The state of each slot, by slot id.
    """
    states = {slot.slot_id: SlotState(slot) for slot in slots}
    for attempt in attempts:
        state = states[attempt.slot_id]
        position = table.position(attempt.passage_id)
        state.attempts += 1
        if position not in state.passages:
            state.passages.append(position)
        state.current = position
        if attempt.outcome == "filled":
            state.filled = attempt
            state._drop_current()
        elif attempt.outcome == "error":
            state.errors += 1
            if state.errors >= MAX_ERRORS_PER_PASSAGE:
                state._drop_current()
        elif attempt.outcome in RETRIED_OUTCOMES and not attempt.retry:
            state.retrying, state.retry_words, state.errors = True, attempt.reused, 0
        else:
            state._drop_current()
    return states


@dataclass(frozen=True)
class Planned:
    """One request to send in the next wave.

    Attributes:
        custom_id: The request's id.
        slot: The slot it is for.
        passage: The passage's index position.
        retry: Whether it is the second request about the passage.
        retry_words: The rare words to name, on a retry after a rare word.
    """

    custom_id: str
    slot: Slot
    passage: int
    retry: bool
    retry_words: tuple[str, ...]


def plan_wave(
    states: Mapping[str, SlotState],
    frame: Frame,
    excluded: set[int],
    max_passages: int,
) -> list[Planned]:
    """Decide the next wave: one request for every slot that still lacks a question.

    A slot without a current passage takes the next passage of its stratum
    that no slot has tried and that is not excluded. Slots choose in frame
    order, so the plan depends only on the log.

    Args:
        states: Every slot's state.
        frame: The frame the slots were drawn from.
        excluded: Passages no slot may take, such as those of the pilot.
        max_passages: The most passages one slot may try.

    Returns:
        The requests, in frame order of their slots.
    """
    used = set(excluded)
    for state in states.values():
        used.update(state.passages)
    planned = []
    for state in sorted(states.values(), key=lambda state: state.slot.frame_position):
        if state.filled is not None:
            continue
        if state.current is None:
            if len(state.passages) >= max_passages:
                continue
            following = (
                position
                for position in candidates(frame, state.slot.frame_position)
                if position not in used
            )
            state.current = next(following, None)
            if state.current is None:
                continue
            used.add(state.current)
        planned.append(
            Planned(
                custom_id=f"{state.slot.slot_id}-{state.attempts + 1}",
                slot=state.slot,
                passage=state.current,
                retry=state.retrying,
                retry_words=state.retry_words,
            )
        )
    return planned


#: The outcomes that come from the API rather than from the code's checks, and
#: that judging an answer again cannot change.
API_OUTCOMES = frozenset({"error", "refused", "truncated"})


def rejudge(
    attempts: Sequence[Attempt], texts: Mapping[str, str], rare_words: RareWords
) -> list[Attempt]:
    """Judge every logged answer again with the checks as they stand now.

    Args:
        attempts: The log, in the order it was written.
        texts: The text of each passage the log names, by passage id.
        rare_words: Which words of a passage a question may not reuse.

    Returns:
        The log with each answer's outcome, problem and reused words judged
        again. An API error, a refusal or an answer cut off at the token limit
        stays as it was.
    """
    judged = []
    for attempt in attempts:
        if attempt.outcome in API_OUTCOMES or attempt.response is None:
            judged.append(attempt)
            continue
        completion = Completion(
            custom_id=attempt.custom_id,
            model=attempt.model,
            stop_reason="end_turn",
            text=attempt.response,
        )
        rare = rare_words.rare_stems(texts[attempt.passage_id])
        verdict, _ = judge_completion(completion, attempt.prompt, rare)
        judged.append(
            attempt.model_copy(
                update={
                    "outcome": verdict.outcome,
                    "problem": verdict.problem,
                    "reused": verdict.reused,
                }
            )
        )
    return judged


def simulate(
    slots: Sequence[Slot],
    attempts: Sequence[Attempt],
    frame: Frame,
    table: PassageTable,
    excluded: set[int],
    max_passages: int,
    max_waves: int = 25,
) -> tuple[list[Attempt], list[Planned]]:
    """Run the waves again with the log in place of the API.

    Each request the plan makes is answered by the first unused logged attempt
    for the same slot, passage and kind of request, first or second. Run on a
    log judged again by :func:`rejudge`, this gives the questions the run would
    have produced under the checks as they stand now, as far as the log holds
    the answers.

    Args:
        slots: The slots of the run.
        attempts: The log, judged as it should count.
        frame: The frame the slots were drawn from.
        table: The passages.
        excluded: Passages no slot may take.
        max_passages: The most passages one slot may try.
        max_waves: The most waves to run.

    Returns:
        The attempts that count, in the order the waves use them, and the
        requests the log holds no answer for. When that second list is not
        empty, the run is incomplete until those requests are made.
    """
    waiting: dict[tuple[str, str, bool], list[Attempt]] = {}
    for attempt in attempts:
        waiting.setdefault((attempt.slot_id, attempt.passage_id, attempt.retry), []).append(attempt)
    used: list[Attempt] = []
    for _ in range(max_waves):
        planned = plan_wave(replay(slots, used, table), frame, excluded, max_passages)
        if not planned:
            return used, []
        missing = []
        for item in planned:
            key = (item.slot.slot_id, table.passage_ids[item.passage], item.retry)
            queue = waiting.get(key, [])
            if queue:
                used.append(queue.pop(0))
            else:
                missing.append(item)
        if missing:
            return used, missing
    return used, []


def render_request(
    prompt: Prompt, passage_text: str, banned: Sequence[str], retry_words: Sequence[str]
) -> tuple[str, str]:
    """Fill in a passage prompt.

    Args:
        prompt: The prompt.
        passage_text: The passage, paragraphs separated by a blank line.
        banned: The passage's rare words.
        retry_words: The rare words an earlier question reused, on a retry.

    Returns:
        The system prompt and the user message.
    """
    user = prompt.user.format(
        passage=passage_text, banned=", ".join(banned) if banned else "(inga)"
    )
    if retry_words:
        user += "\n\n" + prompt.retry.format(reused=", ".join(retry_words))
    return prompt.system, user


def judge_completion(
    completion: Completion, prompt_name: str, rare: set[str]
) -> tuple[Verdict, str | None]:
    """Turn what came back into a verdict, before the question's own checks or including them.

    Args:
        completion: What came back.
        prompt_name: ``natural`` or ``stance``.
        rare: The rare stems of the passage.

    Returns:
        The verdict, and the model's answer as text when there was one.
    """
    if completion.error is not None:
        return Verdict("error", completion.error), None
    if completion.stop_reason == "refusal":
        return Verdict("refused", "the model declined the request"), completion.text
    if completion.stop_reason == "max_tokens":
        return Verdict("truncated", "the answer was cut off at the token limit"), completion.text
    if completion.text is None:
        return Verdict("invalid", "the answer held no text"), None
    try:
        if prompt_name == "stance":
            return check_stance_answer(
                StanceAnswer.model_validate_json(completion.text), rare
            ), completion.text
        return check_passage_answer(
            PassageAnswer.model_validate_json(completion.text), rare
        ), completion.text
    except ValidationError as error:
        return Verdict(
            "invalid", f"the answer does not follow the schema: {error.error_count()} errors"
        ), completion.text


def passage_span(passage_id: str) -> tuple[str, int, int]:
    """Split a passage key into its speech and its first and last paragraph."""
    speech, _, span = passage_id.rpartition(":")
    first, _, last = span.partition("-")
    return speech, int(first), int(last)


def load_passage_texts(speeches_path: Path, passage_ids: Iterable[str]) -> dict[str, str]:
    """Read the text of some passages, paragraphs separated by a blank line.

    Args:
        speeches_path: ``anforanden.parquet``.
        passage_ids: The passages.

    Returns:
        Each passage's text, by passage id.
    """
    spans = {passage_id: passage_span(passage_id) for passage_id in passage_ids}
    wanted = sorted({speech for speech, _, _ in spans.values()})
    if not wanted:
        return {}
    table = pq.read_table(
        speeches_path,
        columns=["anf_id", "paragraphs"],
        filters=pc.field("anf_id").isin(wanted),
    )
    paragraphs = dict(
        zip(table.column("anf_id").to_pylist(), table.column("paragraphs").to_pylist(), strict=True)
    )
    return {
        passage_id: "\n\n".join(paragraphs[speech][first : last + 1])
        for passage_id, (speech, first, last) in spans.items()
    }


def years_in(text: str) -> list[int]:
    """The four-digit years from 1900 to 2099 that a text names, in order."""
    return [int(match) for match in _YEAR.findall(text)]


def response_json(text: str | None) -> Any:
    """Parse a model's JSON answer, or return ``None`` when it is missing or broken."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
