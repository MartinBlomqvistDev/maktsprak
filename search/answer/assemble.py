"""The assembler: the only code that turns a selector's answer into quotations.

A selector names passages and paragraphs; it never supplies text. The
assembler checks each selection against the question's own candidates and
quotes the paragraphs from the corpus, so a quotation is always whole
paragraphs of one speech as the corpus holds them, under that speech's own
speaker, party and link.

The output is read in one form only (:class:`SelectorOutput`, strict). An
output of any other form, or with any other field, is rejected whole, and so
are the selections of an answer marked unanswerable. Each selection is then
accepted only if:

1. it is among the first :data:`MAX_QUOTES` of the output,
2. its passage is among the question's candidates,
3. its first paragraph comes no later than its last,
4. its range lies within the passage,
5. the speech still holds the text the candidate was read from, and
6. it overlaps no selection accepted before it.

The checks run in that order, and the first that fails names the rejection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from search.answer.models import (
    MAX_QUOTES,
    Answer,
    Candidate,
    Quote,
    Rejected,
    Rejection,
    Selected,
    Selection,
    SelectorOutput,
    Speech,
)
from search.models import text_sha256


def _rejection(
    position: int,
    selection: Selection,
    candidates: Mapping[str, Candidate],
    speeches: Mapping[str, Speech],
    taken: Sequence[tuple[str, int, int]],
    max_quotes: int,
) -> Rejection | None:
    """The first check a selection fails, or ``None`` when it passes them all."""
    if position >= max_quotes:
        return "beyond_cap"
    candidate = candidates.get(selection.passage_id)
    if candidate is None:
        return "unknown_passage"
    first, last = selection.first_paragraph, selection.last_paragraph
    if first > last:
        return "reversed_range"
    if first < 1 or last > len(candidate.paragraphs):
        return "outside_passage"
    speech = speeches.get(candidate.anf_id)
    if (
        speech is None
        or len(speech.paragraphs) <= candidate.last
        or text_sha256(speech.paragraphs) != candidate.text_sha256
    ):
        return "stale_text"
    start, end = candidate.first + first - 1, candidate.first + last - 1
    for anf_id, taken_start, taken_end in taken:
        if anf_id == candidate.anf_id and start <= taken_end and taken_start <= end:
            return "overlap"
    return None


def assemble(
    selected: Selected,
    candidates: Sequence[Candidate],
    speeches: Mapping[str, Speech],
    max_quotes: int = MAX_QUOTES,
) -> Answer:
    """Check a selector's answer and quote what it may quote.

    Args:
        selected: What the selector returned for the question.
        candidates: The question's candidates, the only passages it may quote.
        speeches: The corpus's speeches, by key; the quoted text is read here.
        max_quotes: The most quotations an answer holds.

    Returns:
        The answer, with every rejected selection recorded.

    Raises:
        ValueError: If two candidates share a passage id.
    """
    by_id = {candidate.passage_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Two candidates of one question share a passage id.")
    if selected.output is None:
        return Answer(
            selected=selected,
            answerable=None,
            quotes=(),
            rejected=(Rejected(position=None, reason="malformed"),),
        )
    try:
        output = SelectorOutput.model_validate_json(selected.output, strict=True)
    except ValidationError:
        return Answer(
            selected=selected,
            answerable=None,
            quotes=(),
            rejected=(Rejected(position=None, reason="malformed"),),
        )
    if not output.answerable:
        return Answer(
            selected=selected,
            answerable=False,
            quotes=(),
            rejected=tuple(
                Rejected(position=position, reason="marked_unanswerable")
                for position in range(len(output.selections))
            ),
        )

    quotes: list[Quote] = []
    rejected: list[Rejected] = []
    taken: list[tuple[str, int, int]] = []
    for position, selection in enumerate(output.selections):
        reason = _rejection(position, selection, by_id, speeches, taken, max_quotes)
        if reason is not None:
            rejected.append(Rejected(position=position, reason=reason))
            continue
        candidate = by_id[selection.passage_id]
        speech = speeches[candidate.anf_id]
        start = candidate.first + selection.first_paragraph - 1
        end = candidate.first + selection.last_paragraph - 1
        taken.append((candidate.anf_id, start, end))
        quotes.append(
            Quote(
                passage_id=candidate.passage_id,
                rank=candidate.rank,
                first_paragraph=selection.first_paragraph,
                last_paragraph=selection.last_paragraph,
                anf_id=speech.anf_id,
                party=speech.party,
                speaker=speech.speaker,
                date=speech.date,
                title=speech.title,
                url=speech.url,
                paragraphs=speech.paragraphs[start : end + 1],
            )
        )
    return Answer(
        selected=selected, answerable=True, quotes=tuple(quotes), rejected=tuple(rejected)
    )
