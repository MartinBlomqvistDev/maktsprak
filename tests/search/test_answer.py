"""Tests for search/answer/: the candidates, the assembler, the selectors and the budget guard.

Nothing here needs a key or a network: the language model's requests go
through a stand-in for ``complete_one``.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import search.answer.selector as selector_module
from search.answer.assemble import assemble
from search.answer.budget import BudgetGuard, GuardedSelector, answer
from search.answer.candidates import (
    StaleIndexError,
    build_candidates,
    check_index,
    load_speeches,
    rank_passages,
)
from search.answer.models import MAX_QUOTES, Candidate, Selected, Speech
from search.answer.selector import (
    ABSTENTION,
    SELECTOR_SCHEMA,
    BaselineSelector,
    ClaudeSelector,
    choose_threshold,
    failed,
    fall_back,
    load_selector_prompt,
    render_candidates,
    trimmed_span,
)
from search.config import Prices
from search.corpus.store import write_speeches
from search.corpus.table import PassageTable, load_passage_table
from search.eval.llm import Completion, Usage
from search.index.analyzer import Analyzer
from search.index.lexical import (
    Bm25Params,
    IndexManifest,
    LexicalIndex,
    build_index,
    file_sha256,
    passage_ids_sha256,
    passage_texts,
    tokenize_corpus,
)
from tests.search.corpus_kit import speech, write_corpus

OPUS = Prices(input=4.0, output=20.0)
PARAMS = Bm25Params(k1=1.2, b=0.9)
STEMMED = Analyzer(stem=True, stopwords=False)

#: Three speeches, each packed into one passage.
CORPUS = [
    speech(
        "H209100",
        1,
        [
            "Kärnkraften ska byggas ut för att klara elförsörjningen.",
            "Vattenkraften räcker inte till.",
            "Sol och vind blir för dyrt för industrin.",
            "Därför vill vi ha fler reaktorer.",
        ],
        party="M",
        title="Energipolitik",
        speaker="Statsrådet Per Persson (M)",
    ),
    speech(
        "H209100",
        2,
        ["Avveckling av kärnkraften är rätt väg.", "Sol och vind ger billig el."],
        party="MP",
        title="Energipolitik",
    ),
    speech(
        "GR0950",
        1,
        ["Skolan behöver fler lärare och mindre klasser.", "Lärarna ska få högre lön."],
        party="S",
        day=date(2004, 3, 2),
    ),
]


class Corpus:
    """The corpus on disk, its index and what reads it."""

    def __init__(self, directory: Path) -> None:
        self.speeches_path, self.passages_path = write_corpus(directory, CORPUS)
        self.table: PassageTable = load_passage_table(self.passages_path)
        texts = passage_texts(self.speeches_path, self.passages_path)
        self.index: LexicalIndex = build_index(tokenize_corpus(texts), STEMMED)
        self.manifest = IndexManifest(
            analyzer=STEMMED,
            passages=self.index.passages,
            passage_ids_sha256=passage_ids_sha256(self.table.passage_ids),
            passages_sha256=file_sha256(self.passages_path),
            speeches_sha256=file_sha256(self.speeches_path),
            terms=len(self.index.terms),
            postings=len(self.index.doc_ids),
            tokens=int(self.index.doc_lengths.sum()),
            mean_passage_terms=round(self.index.mean_length, 3),
        )
        self.speeches: dict[str, Speech] = load_speeches(
            self.speeches_path, [item.anf_id for item in CORPUS]
        )

    def candidates(self, question: str, allowed=None) -> list[Candidate]:
        hits = rank_passages(self.index, question, PARAMS, allowed)
        return build_candidates(hits, self.table, self.speeches)


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    return Corpus(tmp_path)


def claude(output: object) -> Selected:
    """What the language model returned, as JSON text."""
    text = output if isinstance(output, str) or output is None else json.dumps(output)
    return Selected(selector="claude", output=text)


def picks(*selections: tuple[str, int, int], answerable: bool = True) -> dict:
    return {
        "answerable": answerable,
        "selections": [
            {"passage_id": passage, "first_paragraph": first, "last_paragraph": last}
            for passage, first, last in selections
        ],
    }


ENERGY = "H209100-1:0-3"
GREEN = "H209100-2:0-1"
SCHOOL = "GR0950-1:0-1"


class TestCandidates:
    def test_candidates_are_ranked_from_one_with_their_speech(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        assert [candidate.rank for candidate in candidates] == [1, 2]
        assert {candidate.passage_id for candidate in candidates} == {ENERGY, GREEN}
        first = next(candidate for candidate in candidates if candidate.passage_id == ENERGY)
        assert first.speaker == "Statsrådet Per Persson (M)"
        assert first.party == "M" and first.title == "Energipolitik"
        assert first.paragraphs == tuple(CORPUS[0].paragraphs)
        assert first.url == CORPUS[0].url

    def test_the_filter_keeps_out_other_passages(self, corpus):
        candidates = corpus.candidates("kärnkraften", corpus.table.party_mask("MP"))
        assert [candidate.passage_id for candidate in candidates] == [GREEN]

    def test_a_speech_is_hashed_as_read(self, corpus):
        assert corpus.speeches["GR0950-1"].text_sha256 == CORPUS[2].text_sha256

    def test_an_index_of_this_corpus_is_accepted(self, corpus):
        check_index(corpus.manifest, corpus.table, corpus.speeches_path, corpus.passages_path)

    def test_an_index_of_another_speeches_file_is_refused(self, corpus):
        changed = [CORPUS[0], CORPUS[1], speech("GR0950", 1, ["Skolan ska ha fler lärare."])]
        write_speeches(corpus.speeches_path, [changed])
        with pytest.raises(StaleIndexError, match="anforanden.parquet differs"):
            check_index(corpus.manifest, corpus.table, corpus.speeches_path, corpus.passages_path)

    def test_an_index_of_another_passage_count_is_refused(self, corpus):
        manifest = corpus.manifest.model_copy(update={"passages": 4})
        with pytest.raises(StaleIndexError, match="4 passages in the index"):
            check_index(manifest, corpus.table, corpus.speeches_path, corpus.passages_path)

    def test_a_hit_whose_speech_is_too_short_is_refused(self, corpus):
        hits = rank_passages(corpus.index, "kärnkraften", PARAMS)
        short = corpus.speeches["H209100-1"].model_copy(update={"paragraphs": ("Kort.",)})
        speeches = {**corpus.speeches, "H209100-1": short}
        with pytest.raises(StaleIndexError):
            build_candidates(hits, corpus.table, speeches)


class TestAssembler:
    def test_a_quotation_is_the_corpus_text_under_its_own_speech(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        result = assemble(claude(picks((ENERGY, 2, 3))), candidates, corpus.speeches)
        assert result.rejected == () and result.answerable is True
        [quote] = result.quotes
        assert quote.paragraphs == tuple(CORPUS[0].paragraphs[1:3])
        assert quote.text == "\n\n".join(CORPUS[0].paragraphs[1:3])
        assert (quote.anf_id, quote.party, quote.url) == ("H209100-1", "M", CORPUS[0].url)
        assert (quote.first_paragraph, quote.last_paragraph) == (2, 3)

    def test_an_invented_id_is_rejected(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        result = assemble(claude(picks(("H209100-9:0-1", 1, 1))), candidates, corpus.speeches)
        assert result.quotes == ()
        assert [item.reason for item in result.rejected] == ["unknown_passage"]

    def test_an_id_from_another_question_is_rejected(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        assert SCHOOL not in {candidate.passage_id for candidate in candidates}
        result = assemble(claude(picks((SCHOOL, 1, 2))), candidates, corpus.speeches)
        assert [item.reason for item in result.rejected] == ["unknown_passage"]

    @pytest.mark.parametrize(
        ("first", "last", "reason"),
        [
            (1, 5, "outside_passage"),
            (0, 1, "outside_passage"),
            (-1, 2, "outside_passage"),
            (3, 2, "reversed_range"),
        ],
    )
    def test_a_range_outside_the_passage_is_rejected(self, corpus, first, last, reason):
        candidates = corpus.candidates("kärnkraften")
        result = assemble(claude(picks((ENERGY, first, last))), candidates, corpus.speeches)
        assert result.quotes == ()
        assert [item.reason for item in result.rejected] == [reason]

    @pytest.mark.parametrize(
        "output",
        [
            {**picks((ENERGY, 1, 1)), "reason": "Talaren vill bygga ut."},
            {
                "answerable": True,
                "selections": [
                    {
                        "passage_id": ENERGY,
                        "first_paragraph": 1,
                        "last_paragraph": 1,
                        "text": "Påhittad text.",
                    }
                ],
            },
            {"answerable": True},
            {
                "answerable": True,
                "selections": [{"passage_id": ENERGY, "first_paragraph": "1", "last_paragraph": 1}],
            },
            {
                "answerable": True,
                "selections": [{"passage_id": ENERGY, "first_paragraph": 1.0, "last_paragraph": 1}],
            },
            {"answerable": "true", "selections": []},
            "not json",
            None,
        ],
    )
    def test_an_output_of_another_form_is_rejected_whole(self, corpus, output):
        candidates = corpus.candidates("kärnkraften")
        result = assemble(claude(output), candidates, corpus.speeches)
        assert result.quotes == () and result.answerable is None
        assert [(item.position, item.reason) for item in result.rejected] == [(None, "malformed")]

    def test_a_speech_changed_since_its_candidate_was_read_is_rejected(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        changed = corpus.speeches["H209100-1"].model_copy(
            update={"paragraphs": ("Kärnkraften ska avvecklas.", *CORPUS[0].paragraphs[1:])}
        )
        speeches = {**corpus.speeches, "H209100-1": changed}
        result = assemble(claude(picks((ENERGY, 1, 1), (GREEN, 1, 1))), candidates, speeches)
        assert [item.reason for item in result.rejected] == ["stale_text"]
        assert [quote.passage_id for quote in result.quotes] == [GREEN]

    def test_a_speech_missing_from_the_corpus_is_rejected(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        speeches = {key: value for key, value in corpus.speeches.items() if key != "H209100-1"}
        result = assemble(claude(picks((ENERGY, 1, 1))), candidates, speeches)
        assert [item.reason for item in result.rejected] == ["stale_text"]

    def test_selections_of_an_unanswerable_answer_are_rejected(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        output = picks((ENERGY, 1, 1), (GREEN, 1, 2), answerable=False)
        result = assemble(claude(output), candidates, corpus.speeches)
        assert result.quotes == () and result.answerable is False
        assert [(item.position, item.reason) for item in result.rejected] == [
            (0, "marked_unanswerable"),
            (1, "marked_unanswerable"),
        ]

    def test_an_overlapping_selection_is_rejected_and_an_adjacent_one_kept(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        output = picks((ENERGY, 1, 2), (ENERGY, 2, 3), (ENERGY, 3, 4))
        result = assemble(claude(output), candidates, corpus.speeches)
        assert [(quote.first_paragraph, quote.last_paragraph) for quote in result.quotes] == [
            (1, 2),
            (3, 4),
        ]
        assert [(item.position, item.reason) for item in result.rejected] == [(1, "overlap")]

    def test_a_selection_beyond_the_cap_is_rejected(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        output = picks(*[(ENERGY, number, number) for number in range(1, 5)], (GREEN, 1, 1))
        result = assemble(claude(output), candidates, corpus.speeches)
        assert len(result.quotes) == MAX_QUOTES
        assert [(item.position, item.reason) for item in result.rejected] == [(4, "beyond_cap")]

    def test_no_quotation_is_an_abstention(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        assert assemble(claude(picks()), candidates, corpus.speeches).abstained
        assert assemble(claude(ABSTENTION), candidates, corpus.speeches).abstained

    def test_two_candidates_with_one_id_are_refused(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        with pytest.raises(ValueError):
            assemble(claude(picks()), [candidates[0], candidates[0]], corpus.speeches)


class TestBaseline:
    def test_it_quotes_from_the_first_to_the_last_paragraph_holding_a_question_word(self):
        paragraphs = CORPUS[0].paragraphs
        assert trimmed_span("Vad sägs om vattenkraften och industrin?", paragraphs) == (2, 3)

    def test_stop_words_of_the_question_are_not_looked_for(self):
        paragraphs = CORPUS[0].paragraphs
        assert trimmed_span("Vad vill de om reaktorer?", paragraphs) == (4, 4)

    def test_without_a_question_word_it_quotes_the_whole_passage(self):
        assert trimmed_span("Hur går det för fotbollen?", CORPUS[0].paragraphs) == (1, 4)

    def test_the_ceiling_is_k1_plus_one_times_the_idf_of_each_term(self, corpus):
        selector = BaselineSelector(corpus.index, PARAMS, threshold=0.5)
        terms = dict.fromkeys(STEMMED.terms("kärnkraften kärnkraften zebra"))
        expected = (PARAMS.k1 + 1) * sum(corpus.index.idf(term) for term in terms)
        assert selector.ceiling("kärnkraften kärnkraften zebra") == pytest.approx(expected)
        assert corpus.index.document_frequency("zebra") == 0

    def test_a_score_below_the_threshold_abstains_and_one_above_quotes(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        score = BaselineSelector(corpus.index, PARAMS, 0.0).relative_score(
            "kärnkraften", candidates
        )
        assert 0 < score < 1
        above = BaselineSelector(corpus.index, PARAMS, threshold=score)
        below = BaselineSelector(corpus.index, PARAMS, threshold=score + 1e-9)
        quoted = assemble(above.select("kärnkraften", candidates), candidates, corpus.speeches)
        assert [quote.passage_id for quote in quoted.quotes] == [candidates[0].passage_id]
        assert assemble(
            below.select("kärnkraften", candidates), candidates, corpus.speeches
        ).abstained

    def test_without_candidates_it_abstains(self, corpus):
        selector = BaselineSelector(corpus.index, PARAMS, threshold=0.0)
        assert selector.select("fotboll", []).output == ABSTENTION
        assert selector.relative_score("fotboll", []) is None

    def test_the_threshold_with_the_highest_balanced_accuracy_wins(self):
        threshold, accuracy = choose_threshold([0.9, 0.5, None], [0.2, 0.6])
        assert threshold == 0.9 and accuracy == pytest.approx(2 / 3)

    def test_a_tie_goes_to_the_lowest_threshold(self):
        threshold, accuracy = choose_threshold([0.5, 0.9], [0.3, 0.7])
        assert threshold == 0.5 and accuracy == pytest.approx(0.75)

    def test_a_question_without_candidates_is_abstained_on_at_every_threshold(self):
        threshold, accuracy = choose_threshold([0.4], [None])
        assert threshold == 0.0 and accuracy == 1.0

    def test_the_threshold_needs_both_kinds_of_question(self):
        with pytest.raises(ValueError):
            choose_threshold([0.5], [])


def completion(text: str | None = None, stop: str = "end_turn", error: str | None = None):
    return Completion(
        custom_id="answer",
        model="claude-opus-5-5",
        stop_reason=None if error else stop,
        text=text,
        usage=Usage(input_tokens=7000, output_tokens=500),
        error=error,
    )


@pytest.fixture
def requests_sent(monkeypatch):
    """Replace the request with a stand-in that returns the queued completions in order."""
    queue: list[Completion] = []
    sent: list[dict] = []

    def complete_one(custom_id, params, client=None):
        sent.append(params)
        return queue.pop(0)

    monkeypatch.setattr(selector_module, "complete_one", complete_one)
    return queue, sent


def claude_selector() -> ClaudeSelector:
    return ClaudeSelector(load_selector_prompt(), "claude-opus-5-5", "low", 8000, OPUS)


class TestClaudeSelector:
    def test_the_prompt_has_its_three_sections(self):
        prompt = load_selector_prompt()
        assert "{question}" in prompt.user and "{candidates}" in prompt.user
        assert prompt.retry == "" and len(prompt.sha256) == 64

    def test_the_request_shows_each_candidate_with_numbered_paragraphs(self, corpus):
        candidates = corpus.candidates("kärnkraften")
        params = claude_selector().request("Ska kärnkraften byggas ut?", candidates)
        user = params["messages"][0]["content"]
        assert user.startswith("Fråga: Ska kärnkraften byggas ut?\n\n")
        assert f'<utdrag id="{ENERGY}">' in user
        assert "Talare: Statsrådet Per Persson (M)\nParti: M\nDatum: 2015-05-12" in user
        assert "Debatt: Energipolitik" in user
        assert "[4] Därför vill vi ha fler reaktorer." in user
        assert params["output_config"] == {
            "effort": "low",
            "format": {"type": "json_schema", "schema": SELECTOR_SCHEMA},
        }
        assert params["model"] == "claude-opus-5-5" and params["max_tokens"] == 8000

    def test_a_debate_without_a_heading_has_no_heading_line(self, corpus):
        candidates = corpus.candidates("lärare", corpus.table.party_mask("S"))
        assert "Debatt:" not in render_candidates(candidates)

    def test_the_schema_is_the_same_for_every_question(self, corpus):
        selector = claude_selector()
        first = selector.request("kärnkraften", corpus.candidates("kärnkraften"))
        second = selector.request("lärare", corpus.candidates("lärare"))
        assert first["output_config"] == second["output_config"]

    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            (completion(text="{}"), False),
            (completion(text="{}", stop="refusal"), True),
            (completion(text="{", stop="max_tokens"), True),
            (completion(error="HTTP 529: overloaded"), True),
            (completion(text=None), True),
        ],
    )
    def test_a_failed_request_is_told_apart(self, item, expected):
        assert failed(item) is expected

    def test_a_failed_request_is_sent_once_more(self, corpus, requests_sent):
        queue, sent = requests_sent
        output = json.dumps(picks((ENERGY, 1, 1)))
        queue.extend([completion(stop="refusal"), completion(text=output)])
        selected = claude_selector().select("kärnkraften", corpus.candidates("kärnkraften"))
        assert len(sent) == 2 and sent[0] == sent[1]
        assert selected.output == output and selected.attempts == 2
        assert selected.usage.input_tokens == 14_000
        assert selected.cost_usd == pytest.approx(2 * (7000 * 4 + 500 * 20) / 1e6)

    def test_two_failures_leave_no_output(self, corpus, requests_sent):
        queue, _ = requests_sent
        queue.extend([completion(error="HTTP 500: error"), completion(stop="max_tokens", text="{")])
        selected = claude_selector().select("kärnkraften", corpus.candidates("kärnkraften"))
        assert selected.output is None and selected.stop_reason == "max_tokens"

    def test_no_request_is_sent_without_candidates(self, requests_sent):
        _, sent = requests_sent
        assert claude_selector().select("fotboll", []).output == ABSTENTION
        assert sent == []

    def test_a_batch_costs_less(self):
        selected = claude_selector().read([completion(text="{}")], batch_discount=0.5)
        assert selected.cost_usd == pytest.approx((7000 * 4 + 500 * 20) / 1e6 / 2)

    def test_the_baseline_answers_in_place_and_the_cost_is_kept(self):
        failure = claude_selector().read([completion(error="x"), completion(error="y")])
        baseline = Selected(selector="baseline", output=ABSTENTION)
        result = fall_back(failure, baseline, "failed")
        assert (result.selector, result.fallback, result.output) == (
            "baseline",
            "failed",
            ABSTENTION,
        )
        assert result.attempts == 2 and result.cost_usd == failure.cost_usd


def fixed_clock(year: int, month: int):
    return lambda: datetime(year, month, 15, 12, tzinfo=timezone.utc)


class TestBudget:
    def test_the_guard_allows_until_the_budget_is_spent(self):
        guard = BudgetGuard(2.0, clock=fixed_clock(2026, 11))
        guard.record(1.99)
        assert guard.allows()
        guard.record(0.02)
        assert not guard.allows()

    def test_a_new_month_starts_from_nothing(self):
        months = iter([(2026, 11), (2026, 11), (2026, 12)])
        guard = BudgetGuard(2.0, clock=lambda: datetime(*next(months), 1, tzinfo=timezone.utc))
        guard.record(5.0)
        assert not guard.allows()
        assert guard.allows()

    def test_the_ledger_keeps_the_month_across_a_restart(self, tmp_path):
        ledger = tmp_path / "budget.json"
        BudgetGuard(2.0, ledger, clock=fixed_clock(2026, 11)).record(1.5)
        again = BudgetGuard(2.0, ledger, clock=fixed_clock(2026, 11))
        assert again.spent() == pytest.approx(1.5)
        assert BudgetGuard(2.0, ledger, clock=fixed_clock(2026, 12)).spent() == 0.0

    def test_within_the_budget_the_language_model_answers(self, corpus, requests_sent):
        queue, _ = requests_sent
        queue.append(completion(text=json.dumps(picks((ENERGY, 1, 1)))))
        guard = BudgetGuard(2.0, clock=fixed_clock(2026, 11))
        baseline = BaselineSelector(corpus.index, PARAMS, threshold=0.0)
        guarded = GuardedSelector(claude_selector(), baseline, guard)
        candidates = corpus.candidates("kärnkraften")
        result = answer("kärnkraften", candidates, corpus.speeches, guarded)
        assert result.selected.selector == "claude" and result.selected.fallback is None
        assert [quote.passage_id for quote in result.quotes] == [ENERGY]
        assert guard.spent() == pytest.approx((7000 * 4 + 500 * 20) / 1e6)

    def test_beyond_the_budget_the_baseline_answers_and_says_so(self, corpus, requests_sent):
        _, sent = requests_sent
        guard = BudgetGuard(2.0, clock=fixed_clock(2026, 11))
        guard.record(2.0)
        baseline = BaselineSelector(corpus.index, PARAMS, threshold=0.0)
        guarded = GuardedSelector(claude_selector(), baseline, guard)
        candidates = corpus.candidates("kärnkraften")
        result = answer("kärnkraften", candidates, corpus.speeches, guarded)
        assert (result.selected.selector, result.selected.fallback) == ("baseline", "budget")
        assert sent == [] and result.quotes

    def test_after_two_failures_the_baseline_answers(self, corpus, requests_sent):
        queue, _ = requests_sent
        queue.extend([completion(error="HTTP 500: a"), completion(error="HTTP 500: b")])
        guard = BudgetGuard(2.0, clock=fixed_clock(2026, 11))
        baseline = BaselineSelector(corpus.index, PARAMS, threshold=0.0)
        guarded = GuardedSelector(claude_selector(), baseline, guard)
        candidates = corpus.candidates("kärnkraften")
        result = answer("kärnkraften", candidates, corpus.speeches, guarded)
        assert (result.selected.selector, result.selected.fallback) == ("baseline", "failed")
        assert result.selected.attempts == 2 and guard.spent() > 0
