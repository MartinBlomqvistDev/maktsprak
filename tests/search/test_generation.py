"""Tests for search/eval/generation.py: prompts, rare words, checks and the order of work."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from search.corpus.table import PassageTable
from search.eval.generation import (
    PROMPTS_DIR,
    Attempt,
    PassageAnswer,
    RareWords,
    StanceAnswer,
    append_log,
    check_passage_answer,
    check_stance_answer,
    is_single_word,
    judge_completion,
    load_passage_texts,
    load_prompt,
    passage_span,
    plan_wave,
    read_log,
    rejudge,
    render_request,
    replay,
    simulate,
    years_in,
)
from search.eval.llm import Completion
from search.eval.sampling import Slot, build_frame
from search.index.analyzer import STEMMING, Analyzer
from search.index.lexical import build_index, tokenize_corpus
from tests.search.corpus_kit import SPEECHES, write_corpus

WHEN = datetime(2026, 11, 21, tzinfo=timezone.utc)


class TestPrompts:
    TEXT = "=== system ===\nSys {count}\n=== user ===\nUse {passage}\n=== retry ===\nAgain\n"

    def test_a_prompt_file_has_its_three_sections(self, tmp_path):
        (tmp_path / "x.md").write_bytes(self.TEXT.encode("utf-8"))
        prompt = load_prompt("x", tmp_path)
        assert (prompt.system, prompt.user, prompt.retry) == (
            "Sys {count}",
            "Use {passage}",
            "Again",
        )
        assert prompt.sha256 == hashlib.sha256(self.TEXT.encode("utf-8")).hexdigest()

    def test_windows_line_ends_give_the_same_prompt_and_hash(self, tmp_path):
        (tmp_path / "unix.md").write_bytes(self.TEXT.encode("utf-8"))
        (tmp_path / "windows.md").write_bytes(self.TEXT.replace("\n", "\r\n").encode("utf-8"))
        unix, windows = load_prompt("unix", tmp_path), load_prompt("windows", tmp_path)
        assert (windows.system, windows.user, windows.retry, windows.sha256) == (
            unix.system,
            unix.user,
            unix.retry,
            unix.sha256,
        )

    def test_sections_out_of_order_are_refused(self, tmp_path):
        (tmp_path / "x.md").write_text(
            "=== user ===\nU\n=== system ===\nS\n=== retry ===\n", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            load_prompt("x", tmp_path)

    @pytest.mark.parametrize(
        ("name", "placeholders"),
        [
            ("natural", ["{passage}", "{banned}"]),
            ("stance", ["{passage}", "{banned}"]),
            ("off_topic", ["{count}"]),
            ("unknown_speaker", ["{count}"]),
            ("sd_before_2010", ["{count}"]),
            ("after_index", ["{count}"]),
        ],
    )
    def test_the_shipped_prompts_load_with_their_placeholders(self, name, placeholders):
        prompt = load_prompt(name, PROMPTS_DIR)
        assert all(holder in prompt.system + prompt.user for holder in placeholders)
        assert "—" not in prompt.system + prompt.user + prompt.retry

    def test_a_retry_names_the_reused_words(self):
        prompt = load_prompt("natural", PROMPTS_DIR)
        _, first = render_request(prompt, "Text.", [], [])
        assert first.endswith("(inga)")
        _, again = render_request(prompt, "Text.", ["jantelagen"], ["jantelagen"])
        assert "jantelagen, som inte får stå i frågan" in again


@pytest.fixture
def rare_words() -> RareWords:
    """Rare words counted over four short passages: a stem in fewer than two is rare."""
    corpus = tokenize_corpus(
        [
            ("a:0-0", "Jantelagen styr skolan"),
            ("b:0-0", "skolan och lärarna"),
            ("c:0-0", "lärarna och eleverna"),
            ("d:0-0", "eleverna och skolan"),
        ]
    )
    return RareWords(build_index(corpus, STEMMING), share=0.5)


class TestRareWords:
    def test_a_stem_in_fewer_passages_than_the_share_is_rare(self, rare_words):
        assert rare_words.rare_stems("Jantelagen styr skolan") == {"jantelag", "styr"}

    def test_banned_words_are_the_rare_forms_as_written(self, rare_words):
        assert rare_words.banned("Jantelagen styr skolan, jantelagen!") == ["jantelagen", "styr"]

    def test_another_inflection_of_a_rare_word_is_caught(self, rare_words):
        rare = rare_words.rare_stems("Jantelagen styr skolan")
        assert RareWords.reused("Vad betyder jantelag för skolan?", rare) == ["jantelag"]

    def test_only_the_stemmed_index_without_stop_words_counts(self):
        index = build_index(tokenize_corpus([("a:0-0", "x")]), Analyzer(stem=True, stopwords=True))
        with pytest.raises(ValueError):
            RareWords(index, 0.5)


def passage_answer(**changes) -> PassageAnswer:
    fields = {
        "usable": True,
        "reason": "Svarar.",
        "question": "Varför behövs fler lärare i skolan?",
        "keywords": ["lärare", "skola"],
    }
    fields.update(changes)
    return PassageAnswer(**fields)


class TestPassageChecks:
    def test_a_good_answer_fills_the_slot(self):
        assert check_passage_answer(passage_answer(), set()).outcome == "filled"

    def test_an_unusable_passage_is_unusable(self):
        verdict = check_passage_answer(passage_answer(usable=False, question=""), set())
        assert verdict.outcome == "unusable"

    @pytest.mark.parametrize(
        "changes",
        [
            {"question": "Varför behövs fler lärare"},
            {"question": " ".join(["ord"] * 26) + "?"},
            {"keywords": ["lärare"]},
            {"keywords": ["a", "b", "c", "d", "e"]},
            {"keywords": ["andra chans", "komvux"]},
            {"keywords": ["Lärare", "lärare"]},
        ],
    )
    def test_a_broken_rule_of_form_is_invalid(self, changes):
        assert check_passage_answer(passage_answer(**changes), set()).outcome == "invalid"

    def test_a_reused_rare_word_is_caught_by_its_stem(self):
        verdict = check_passage_answer(passage_answer(), {"lär"})
        assert verdict.outcome == "rare"
        assert verdict.reused == ("lärare",)

    def test_a_hyphenated_keyword_is_one_word(self):
        answer = passage_answer(keywords=["a-kassan", "EU-medborgare", "ersättning"])
        assert check_passage_answer(answer, set()).outcome == "filled"

    @pytest.mark.parametrize(
        ("word", "single"),
        [
            ("a-kassan", True),
            ("2003", True),
            ("hela landet", False),
            (" skola", False),
            ("-", False),
        ],
    )
    def test_a_single_word_has_no_whitespace_and_a_letter_or_digit(self, word, single):
        assert is_single_word(word) is single


def stance_answer(**changes) -> StanceAnswer:
    fields = {
        "usable": True,
        "reason": "Uttalad.",
        "direction": "mot",
        "question": "Vilka argument har förts fram mot att lägga ned kärnkraften?",
    }
    fields.update(changes)
    return StanceAnswer(**fields)


class TestStanceChecks:
    def test_a_good_stance_question_fills_the_slot(self):
        assert check_stance_answer(stance_answer(), set()).outcome == "filled"

    def test_no_stance_is_unusable(self):
        answer = stance_answer(usable=False, direction="ingen", question="")
        assert check_stance_answer(answer, set()).outcome == "unusable"

    def test_the_opening_must_match_the_direction(self):
        answer = stance_answer(direction="för")
        assert check_stance_answer(answer, set()).outcome == "invalid"

    def test_a_reused_rare_word_is_caught(self):
        assert check_stance_answer(stance_answer(), {"kärnkraft"}).outcome == "rare"


class TestCompletions:
    def test_what_comes_back_is_judged_in_order(self):
        assert judge_completion(Completion(custom_id="x", error="HTTP 500"), "natural", set())[
            0
        ].outcome == ("error")
        refused = Completion(custom_id="x", stop_reason="refusal", text=None)
        assert judge_completion(refused, "natural", set())[0].outcome == "refused"
        cut = Completion(custom_id="x", stop_reason="max_tokens", text="{")
        assert judge_completion(cut, "natural", set())[0].outcome == "truncated"
        broken = Completion(custom_id="x", stop_reason="end_turn", text='{"usable": true}')
        assert judge_completion(broken, "natural", set())[0].outcome == "invalid"
        good = Completion(
            custom_id="x", stop_reason="end_turn", text=passage_answer().model_dump_json()
        )
        verdict, text = judge_completion(good, "natural", set())
        assert verdict.outcome == "filled" and json.loads(text)["usable"] is True


def tiny_table() -> PassageTable:
    """Twelve passages in two strata of six: 2015 and S, 2016 and S, all of one length."""
    return PassageTable(
        passage_ids=[f"H{number}-1:0-0" for number in range(12)],
        anf_ids=[f"H{number}-1" for number in range(12)],
        parties=np.array(["S"] * 12, dtype="<U2"),
        years=np.array([2015] * 6 + [2016] * 6, dtype=np.int16),
        words=np.array([100] * 12, dtype=np.int32),
    )


def attempt(slot: Slot, passage: int, outcome: str, retry=False, reused=()) -> Attempt:
    return Attempt(
        custom_id=f"{slot.slot_id}-x",
        slot_id=slot.slot_id,
        passage_id=f"H{passage}-1:0-0",
        retry=retry,
        prompt="natural",
        prompt_sha256="0" * 64,
        outcome=outcome,
        reused=reused,
        created=WHEN,
    )


class TestWaves:
    @pytest.fixture
    def setting(self):
        table = tiny_table()
        frame = build_frame(table, seed=1)
        slots = [
            Slot(slot_id="natural-0001", question_set="natural", split="dev", frame_position=0),
            Slot(slot_id="natural-0002", question_set="natural", split="test", frame_position=6),
        ]
        return table, frame, slots

    def test_the_first_wave_asks_about_the_drawn_passages(self, setting):
        table, frame, slots = setting
        planned = plan_wave(replay(slots, [], table), frame, set(), 12)
        assert [item.passage for item in planned] == [int(frame.order[0]), int(frame.order[6])]
        assert [item.custom_id for item in planned] == ["natural-0001-1", "natural-0002-1"]
        assert not any(item.retry for item in planned)

    def test_a_rare_word_keeps_the_passage_once_and_names_the_words(self, setting):
        table, frame, slots = setting
        first = int(frame.order[0])
        log = [attempt(slots[0], first, "rare", reused=("jantelagen",))]
        planned = plan_wave(replay(slots[:1], log, table), frame, set(), 12)
        assert planned[0].passage == first
        assert planned[0].retry and planned[0].retry_words == ("jantelagen",)
        log.append(attempt(slots[0], first, "rare", retry=True, reused=("jantelagen",)))
        planned = plan_wave(replay(slots[:1], log, table), frame, set(), 12)
        assert planned[0].passage == int(frame.order[1])
        assert not planned[0].retry

    def test_an_unusable_passage_moves_the_slot_on_within_its_stratum(self, setting):
        table, frame, slots = setting
        log = [attempt(slots[0], int(frame.order[0]), "unusable")]
        planned = plan_wave(replay(slots[:1], log, table), frame, set(), 12)
        assert planned[0].passage == int(frame.order[1])
        assert table.years[planned[0].passage] == 2015

    def test_an_api_error_repeats_the_request_until_three(self, setting):
        table, frame, slots = setting
        first = int(frame.order[0])
        log = [attempt(slots[0], first, "error"), attempt(slots[0], first, "error")]
        assert plan_wave(replay(slots[:1], log, table), frame, set(), 12)[0].passage == first
        log.append(attempt(slots[0], first, "error"))
        assert plan_wave(replay(slots[:1], log, table), frame, set(), 12)[0].passage != first

    def test_a_filled_slot_asks_nothing_more(self, setting):
        table, frame, slots = setting
        log = [attempt(slots[0], int(frame.order[0]), "filled")]
        planned = plan_wave(replay(slots, log, table), frame, set(), 12)
        assert [item.slot.slot_id for item in planned] == ["natural-0002"]

    def test_excluded_passages_and_the_limit_are_respected(self, setting):
        table, frame, slots = setting
        excluded = {int(frame.order[0]), int(frame.order[1])}
        planned = plan_wave(replay(slots[:1], [], table), frame, excluded, 12)
        assert planned[0].passage == int(frame.order[2])
        log = [attempt(slots[0], int(frame.order[0]), "unusable")]
        assert plan_wave(replay(slots[:1], log, table), frame, set(), 1) == []

    def test_two_slots_never_take_the_same_passage(self):
        table = tiny_table()
        frame = build_frame(table, seed=1)
        slots = [
            Slot(slot_id=f"natural-000{n}", question_set="natural", split="dev", frame_position=n)
            for n in (1, 2)
        ]
        log = [attempt(slots[0], int(frame.order[1]), "unusable")]
        planned = plan_wave(replay(slots, log, table), frame, set(), 12)
        assert len({item.passage for item in planned}) == 2


class TestJudgedAgain:
    """A check corrected after a run counts as if it had held from the start."""

    @pytest.fixture
    def setting(self):
        table = tiny_table()
        frame = build_frame(table, seed=1)
        slot = Slot(slot_id="natural-0001", question_set="natural", split="dev", frame_position=0)
        return table, frame, slot

    def logged(self, slot, passage: int, outcome: str, keywords, retry=False) -> Attempt:
        answer = passage_answer(keywords=keywords).model_dump_json()
        return attempt(slot, passage, outcome, retry=retry).model_copy(update={"response": answer})

    def test_an_answer_turned_away_wrongly_now_fills_its_slot(self, setting):
        table, frame, slot = setting
        first, second = int(frame.order[0]), int(frame.order[1])
        log = [
            self.logged(slot, first, "invalid", ["a-kassan", "lön"]),
            self.logged(slot, first, "invalid", ["a-kassan", "lön"], retry=True),
            self.logged(slot, second, "filled", ["skola", "lärare"]),
        ]
        texts = {entry.passage_id: "Lönerna och a-kassan." for entry in log}
        index = build_index(tokenize_corpus([("x:0-0", "lönerna och a-kassan")]), STEMMING)
        judged = rejudge(log, texts, RareWords(index, share=0.01))
        assert [entry.outcome for entry in judged] == ["filled", "filled", "filled"]
        counted, missing = simulate([slot], judged, frame, table, set(), 12)
        assert missing == []
        assert [entry.passage_id for entry in counted] == [log[0].passage_id]

    def test_an_api_outcome_is_not_judged_again(self, setting):
        table, _, slot = setting
        error = attempt(slot, 0, "error")
        index = build_index(tokenize_corpus([("x:0-0", "x")]), STEMMING)
        assert rejudge([error], {}, RareWords(index, share=0.5)) == [error]

    def test_a_request_the_log_cannot_answer_is_named(self, setting):
        table, frame, slot = setting
        log = [self.logged(slot, int(frame.order[0]), "unusable", ["a", "b"])]
        counted, missing = simulate([slot], log, frame, table, set(), 12)
        assert len(counted) == 1
        assert [item.passage for item in missing] == [int(frame.order[1])]


class TestLogsAndTexts:
    def test_a_log_reads_back_in_order(self, tmp_path):
        slot = Slot(slot_id="natural-0001", question_set="natural", split="dev", frame_position=0)
        entries = [attempt(slot, 1, "unusable"), attempt(slot, 2, "filled")]
        append_log(tmp_path / "log.jsonl", entries[:1])
        append_log(tmp_path / "log.jsonl", entries[1:])
        assert read_log(tmp_path / "log.jsonl") == entries
        assert read_log(tmp_path / "missing.jsonl") == []

    def test_a_passage_text_is_its_paragraphs_with_a_blank_line_between(self, tmp_path):
        speeches_path, _ = write_corpus(tmp_path)
        key = f"{SPEECHES[0].anf_id}:0-1"
        assert load_passage_texts(speeches_path, [key]) == {
            key: "\n\n".join(SPEECHES[0].paragraphs)
        }
        assert passage_span("HC09139-36:2-4") == ("HC09139-36", 2, 4)

    def test_years_are_found_in_text(self):
        assert years_in("Valen 2006 och 2030, inte 1500 eller 20300.") == [2006, 2030]
