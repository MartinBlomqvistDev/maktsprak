"""Tests for search/eval/quote_judge.py and search/eval/answer_metrics.py."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from search.config import Prices
from search.eval.answer_metrics import (
    QuestionOutcome,
    QuoteOutcome,
    abstention,
    accuracy_difference,
    choose_round,
    party_shares,
    precision_difference,
    quote_precision,
    ungraded,
)
from search.eval.generation import Prompt, load_prompt
from search.eval.llm import Completion, Usage
from search.eval.quote_judge import (
    JUDGE_PROMPT_SHA256,
    GradingPair,
    QuoteGrade,
    check_judge_prompt,
    grades_by_key,
    grading_requests,
    pending,
    read_grades,
    text_sha256,
)
from search.parties import PARTY_ORDER

OPUS = Prices(input=4.0, output=20.0)
WHEN = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)


def pair(
    question_id: str = "natural-0001", text: str = "Kärnkraften ska byggas ut."
) -> GradingPair:
    """A question and a text for the judge."""
    return GradingPair(question_id=question_id, question="Ska kärnkraften byggas ut?", text=text)


def grade_line(item: GradingPair, grade: int | None) -> QuoteGrade:
    return QuoteGrade(
        question_id=item.question_id,
        text_sha256=text_sha256(item.text),
        grade=grade,
        prompt_sha256=JUDGE_PROMPT_SHA256,
        created=WHEN,
    )


class TestQuoteJudge:
    def test_the_judge_prompt_is_the_calibrated_one(self):
        check_judge_prompt(load_prompt("judge"))

    def test_another_judge_prompt_is_refused(self):
        other = Prompt(name="judge", system="s", user="u", retry="", sha256="0" * 64)
        with pytest.raises(ValueError):
            check_judge_prompt(other)

    def test_a_pair_is_keyed_by_its_question_and_the_hash_of_its_text(self):
        item = pair()
        assert item.key == f"natural-0001:{text_sha256(item.text)}"
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", item.custom_id)

    def test_a_graded_pair_is_not_sent_again(self):
        first, second = pair(text="A."), pair(text="B.")
        assert pending([first, second, first], [grade_line(first, 2)]) == [second]

    def test_an_ungraded_pair_is_sent_once_more_and_then_left(self):
        item = pair()
        assert pending([item], [grade_line(item, None)]) == [item]
        assert pending([item], [grade_line(item, None), grade_line(item, None)]) == []

    def test_the_first_usable_grade_counts(self):
        item = pair()
        log = [grade_line(item, None), grade_line(item, 1), grade_line(item, 2)]
        assert grades_by_key(log) == {item.key: 1}

    def test_the_judge_sees_the_text_as_its_passage(self):
        prompt = load_prompt("judge")
        requests = grading_requests([pair()], prompt, "claude-opus-5-5", "low", 4000)
        [params] = requests.values()
        assert params["messages"][0]["content"] == prompt.user.format(
            question="Ska kärnkraften byggas ut?", passage="Kärnkraften ska byggas ut."
        )
        assert params["output_config"]["effort"] == "low"

    def test_completions_become_grades(self):
        prompt = load_prompt("judge")
        graded, missing = pair(text="A."), pair(text="B.")
        completions = {
            graded.custom_id: Completion(
                custom_id=graded.custom_id,
                model="claude-opus-5-5",
                stop_reason="end_turn",
                text='{"reason": "Svarar.", "grade": 2}',
                usage=Usage(input_tokens=1000, output_tokens=100),
                batch_id="msgbatch_1",
            )
        }
        lines = read_grades([graded, missing], completions, prompt, OPUS, 0.5, WHEN)
        assert [line.grade for line in lines] == [2, None]
        assert lines[0].cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6 / 2)
        assert lines[0].batch_id == "msgbatch_1" and lines[1].error == "no result"


def quote(grade: int | None, party: str = "S") -> QuoteOutcome:
    return QuoteOutcome(
        passage_id="H1-1:0-1",
        first_paragraph=1,
        last_paragraph=1,
        party=party,
        text_sha256="0" * 64,
        grade=grade,
    )


def outcome(number: int, answerable: bool, *grades: int | None) -> QuestionOutcome:
    return QuestionOutcome(
        question_id=f"q-{number:03d}",
        question_set="natural" if answerable else "unanswerable",
        answerable=answerable,
        selector="claude",
        quotes=tuple(quote(grade) for grade in grades),
    )


#: Four answerable questions and two unanswerable ones.
FIRST = [
    outcome(1, True, 2, 2),
    outcome(2, True, 2),
    outcome(3, True),
    outcome(4, True, 1, None),
    outcome(5, False),
    outcome(6, False, 2),
]


class TestMetrics:
    def test_balanced_accuracy_is_the_mean_of_the_two_shares(self):
        shares = abstention(FIRST)
        assert shares.abstains_unanswerable == 0.5
        assert shares.answers_answerable == 0.75
        assert shares.needless_abstention == 0.25
        assert shares.balanced_accuracy == pytest.approx(0.625)

    def test_precision_counts_only_answerable_questions_and_ungraded_is_not_2(self):
        assert quote_precision(FIRST) == pytest.approx(3 / 5)
        assert ungraded(FIRST) == 1

    def test_no_quotation_has_no_precision(self):
        assert quote_precision([outcome(1, True), outcome(2, False)]) is None

    def test_abstention_needs_both_kinds(self):
        with pytest.raises(ValueError):
            abstention([outcome(1, True)])

    def test_a_selector_against_itself_differs_by_nothing(self):
        assert precision_difference(FIRST, FIRST, seed=1, resamples=200) == (0.0, 0.0, 0.0)
        assert accuracy_difference(FIRST, FIRST, seed=1, resamples=200) == (0.0, 0.0, 0.0)

    def test_the_precision_difference_pools_each_selectors_quotations(self):
        second = [
            outcome(1, True, 1),
            outcome(2, True, 2),
            outcome(3, True, 1),
            outcome(4, True, 1),
            outcome(5, False, 1),
            outcome(6, False),
        ]
        point, low, high = precision_difference(FIRST, second, seed=7, resamples=2000)
        assert point == pytest.approx(3 / 5 - 1 / 4)
        assert low <= point <= high

    def test_the_accuracy_difference_keeps_each_kind_apart(self):
        everything = [outcome(n, item.answerable, 2) for n, item in enumerate(FIRST, start=1)]
        point, low, high = accuracy_difference(FIRST, everything, seed=7, resamples=2000)
        assert point == pytest.approx(0.625 - 0.5)
        assert low <= point <= high

    def test_a_paired_comparison_needs_the_same_questions(self):
        with pytest.raises(ValueError):
            accuracy_difference(FIRST, FIRST[::-1], seed=1, resamples=10)

    def test_party_shares_follow_the_party_order(self):
        shares = party_shares(["S", "S", "M", "SD"])
        assert list(shares) == list(PARTY_ORDER)
        assert shares["S"] == 0.5 and shares["V"] == 0.0

    def test_the_round_with_the_highest_mean_goes_on_and_a_tie_to_the_later(self):
        assert choose_round([(1, 0.8, 0.7), (2, 0.9, 0.7), (3, 0.7, 0.8)]) == 2
        assert choose_round([(1, 0.8, 0.8), (2, 0.9, 0.7)]) == 2
        with pytest.raises(ValueError):
            choose_round([])
