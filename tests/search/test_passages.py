"""Tests for the packing in search/corpus/passages.py and the passage Parquet IO."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from search.config import load_settings
from search.corpus import passages as passages_module
from search.corpus.passages import (
    LARGEST_KEPT,
    PERIOD_FIRST_YEARS,
    Band,
    CorpusTotals,
    GroupTally,
    PassagesPerSpeech,
    TrialTarget,
    WordsPerPassage,
    build_passages,
    check_cover,
    count_words,
    pack,
    paragraph_word_counts,
    partition,
    passage_paragraphs,
    percentile,
    period_label,
    period_of,
)
from search.corpus.store import iter_passages, iter_speeches, write_passages, write_speeches
from search.models import Anforande, Passage, deep_link, passage_id, session_bounds, text_sha256


def paragraph(words: int, mark: str = "ord") -> str:
    """Build a paragraph of *words* words."""
    return " ".join(f"{mark}{index}" for index in range(words))


def speech_of(
    *word_counts: int,
    anf_nummer: int = 1,
    party: str = "S",
    day: date = date(2015, 5, 12),
    person_id: str | None = "0123456789012",
) -> Anforande:
    """Build a speech whose paragraphs hold the given numbers of words, at least one each."""
    paragraphs = tuple(
        paragraph(count, mark=f"p{index}w") for index, count in enumerate(word_counts)
    )
    dok_id = "H209100"
    session_year = day.year if day.month >= 9 else day.year - 1
    return Anforande(
        anf_id=f"{dok_id}-{anf_nummer}",
        dok_id=dok_id,
        rm=f"{session_year}/{(session_year + 1) % 100:02d}",
        dok_nummer=100,
        anf_nummer=anf_nummer,
        date=day,
        speaker_raw="Anna Andersson (S)",
        person_id=person_id,
        party=party,
        is_reply=False,
        title=None,
        activity=None,
        rel_dok_id=None,
        layout="html",
        paragraphs=paragraphs,
        notes=(),
        text_sha256=text_sha256(paragraphs),
        url=deep_link(dok_id, anf_nummer),
    )


def totals_of(speeches: list[Anforande]) -> CorpusTotals:
    """Count speeches, paragraphs and words the way the corpus build does."""
    return CorpusTotals(
        speeches=len(speeches),
        paragraphs=sum(len(speech.paragraphs) for speech in speeches),
        words=sum(count_words(p) for speech in speeches for p in speech.paragraphs),
    )


class TestPartition:
    def test_a_one_paragraph_speech_is_one_passage(self):
        assert partition([37], 200) == [(0, 0)]

    def test_a_speech_shorter_than_the_target_is_one_passage(self):
        assert partition([30, 40, 50], 200) == [(0, 2)]

    def test_a_speech_just_under_one_and_a_half_targets_is_one_passage(self):
        assert partition([149, 150], 200) == [(0, 1)]

    def test_a_half_rounds_up_to_the_next_passage(self):
        assert partition([150, 150], 200) == [(0, 0), (1, 1)]

    def test_a_long_speech_of_even_paragraphs_is_cut_into_equal_runs(self):
        spans = partition([25] * 40, 200)
        assert spans == [(0, 7), (8, 15), (16, 23), (24, 31), (32, 39)]

    def test_no_speech_ends_in_a_stub(self):
        # Filled greedily towards 200, 31 paragraphs of 10 words would end in a passage of 110.
        spans = partition([10] * 31, 200)
        sizes = [10 * (last - first + 1) for first, last in spans]
        assert sizes == [150, 160]

    def test_a_long_paragraph_in_the_middle_stands_alone(self):
        spans = partition([100, 100, 1000, 100, 100], 400)
        assert spans == [(0, 1), (2, 2), (3, 3), (4, 4)]

    def test_a_long_paragraph_is_never_split_even_when_the_target_is_small(self):
        assert partition([50, 50, 1000, 50, 50], 200) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    def test_fewer_paragraphs_than_wanted_passages_gives_one_passage_per_paragraph(self):
        assert partition([500], 100) == [(0, 0)]
        assert partition([300, 300], 50) == [(0, 0), (1, 1)]

    def test_a_tie_between_two_boundaries_goes_to_the_earlier_one(self):
        # 50 words, a target of 25: two passages, ideal cut at 25 words. The boundaries
        # after two and after three paragraphs, at 20 and 30 words, are equally near.
        assert partition([10, 10, 10, 10, 10], 25) == [(0, 1), (2, 4)]

    def test_the_nearer_boundary_wins_when_there_is_no_tie(self):
        # Ideal cut at 50 words; the boundaries lie at 30 and 60 words.
        assert partition([30, 30, 40], 50) == [(0, 1), (2, 2)]

    def test_a_cut_is_moved_rather_than_leaving_a_passage_empty(self):
        # Three passages wanted, with ideal cuts at 334 and 669 words. Both lie nearest the
        # boundary after the long first paragraph, but the second cut must leave the first
        # passage a paragraph of its own and so moves one boundary on.
        assert partition([1000, 1, 1, 1], 400) == [(0, 0), (1, 1), (2, 3)]

    def test_a_paragraph_of_one_word_is_a_valid_count(self):
        assert partition([1, 1, 1], 200) == [(0, 2)]
        assert partition([1, 1], 1) == [(0, 0), (1, 1)]

    @pytest.mark.parametrize("counts", [[0], [30, 0, 30], [0, 300, 300], [-1, 50]])
    def test_a_paragraph_without_words_raises(self, counts):
        with pytest.raises(ValueError, match="at least one word to be packed"):
            partition(counts, 200)

    def test_the_result_depends_only_on_the_counts_and_the_target(self):
        counts = [12, 80, 33, 5, 140, 61, 9]
        assert partition(counts, 150) == partition(list(counts), 150)
        assert partition(tuple(counts), 150) == partition(counts, 150)

    def test_a_target_below_one_word_raises(self):
        with pytest.raises(ValueError, match="at least one word"):
            partition([10], 0)

    def test_a_speech_without_paragraphs_raises(self):
        with pytest.raises(ValueError, match="at least one paragraph"):
            partition([], 200)

    @pytest.mark.parametrize("target", [1, 50, 120, 200, 320, 1000])
    def test_every_paragraph_belongs_to_exactly_one_passage_in_order(self, target):
        generator = random.Random(2026)
        for _ in range(300):
            counts = [
                generator.choice([1, 3, 20, 80, 250, 700]) for _ in range(generator.randint(1, 40))
            ]
            spans = partition(counts, target)
            assert spans[0][0] == 0
            assert spans[-1][1] == len(counts) - 1
            for (_, last), (first, _) in zip(spans, spans[1:], strict=False):
                assert first == last + 1
            assert all(first <= last for first, last in spans)
            wanted = max(1, (2 * sum(counts) + target) // (2 * target))
            assert len(spans) == min(wanted, len(counts))


class TestPack:
    def test_a_passage_is_keyed_by_its_speech_and_its_paragraph_range(self):
        speech = speech_of(25, 25, 25, 25, 25, 25, 25, 25)
        passages = pack(speech, 100)
        assert [passage.passage_id for passage in passages] == [
            "H209100-1:0-3",
            "H209100-1:4-7",
        ]
        assert passage_id("HC09139-36", 0, 2) == "HC09139-36:0-2"

    def test_a_passage_counts_the_words_of_its_paragraphs(self):
        speech = speech_of(10, 20, 30, 40)
        passages = pack(speech, 50)
        assert [passage.words for passage in passages] == [60, 40]

    def test_the_passages_carry_the_fields_a_search_filters_on(self):
        speech = speech_of(10, 10, party="KD", day=date(2016, 2, 3), person_id=None)
        (passage,) = pack(speech, 200)
        assert (passage.party, passage.date, passage.person_id) == ("KD", date(2016, 2, 3), None)

    def test_the_same_speech_and_target_always_give_the_same_passages(self):
        speech = speech_of(*[17, 90, 41, 5, 230, 60, 12, 75, 33])
        assert pack(speech, 120) == pack(speech, 120)

    def test_every_paragraph_is_in_exactly_one_passage(self):
        speech = speech_of(*[17, 90, 41, 5, 230, 60, 12, 75, 33])
        passages = pack(speech, 120)
        rebuilt = tuple(p for passage in passages for p in passage_paragraphs(speech, passage))
        assert rebuilt == speech.paragraphs
        check_cover(speech, passages)

    def test_a_long_speech_of_even_paragraphs_packs_to_equal_passages(self):
        speech = speech_of(*[25] * 40)
        assert [passage.words for passage in pack(speech, 200)] == [200] * 5

    def test_a_target_below_one_word_raises(self):
        with pytest.raises(ValueError, match="at least one word"):
            pack(speech_of(10), 0)

    def test_a_speech_of_one_word_paragraphs_is_packed_whole(self):
        speech = speech_of(1, 1, 1)
        passages = pack(speech, 200)
        assert [(p.first, p.last, p.words) for p in passages] == [(0, 2, 3)]
        check_cover(speech, passages)

    def test_the_word_counts_of_a_speech_follow_its_paragraphs(self):
        assert paragraph_word_counts(speech_of(3, 1, 7)) == [3, 1, 7]
        assert count_words("  Fru   talman!\n Tack. ") == 3


class TestPassageParagraphs:
    def test_returns_the_whole_paragraphs_of_the_passage(self):
        speech = speech_of(25, 25, 25, 25)
        first, second = pack(speech, 50)
        assert passage_paragraphs(speech, first) == speech.paragraphs[0:2]
        assert passage_paragraphs(speech, second) == speech.paragraphs[2:4]

    def test_a_passage_of_another_speech_raises(self):
        passages = pack(speech_of(25, 25), 50)
        with pytest.raises(ValueError, match="does not belong"):
            passage_paragraphs(speech_of(25, 25, anf_nummer=2), passages[0])

    def test_a_passage_reaching_past_the_speech_raises(self):
        passages = pack(speech_of(25, 25, 25, 25), 50)
        with pytest.raises(ValueError, match="reaches past"):
            passage_paragraphs(speech_of(25, 25), passages[-1])

    def test_a_passage_packed_from_other_text_raises(self):
        passages = pack(speech_of(25, 25), 50)
        with pytest.raises(ValueError, match="counts 50 words; its paragraphs hold 51"):
            passage_paragraphs(speech_of(25, 26), passages[0])


class TestCheckCover:
    def test_a_gap_between_passages_is_found(self):
        speech = speech_of(25, 25, 25, 25)
        first, second = pack(speech, 50)
        gap = Passage(
            passage_id=passage_id(speech.anf_id, 3, 3),
            anf_id=speech.anf_id,
            first=3,
            last=3,
            words=25,
            party=speech.party,
            date=speech.date,
            person_id=speech.person_id,
        )
        with pytest.raises(ValueError, match="where 2 was expected"):
            check_cover(speech, [first, gap])

    def test_passages_that_stop_short_are_found(self):
        speech = speech_of(25, 25, 25, 25)
        first, _ = pack(speech, 50)
        with pytest.raises(ValueError, match="end at paragraph 1 of 4"):
            check_cover(speech, [first])

    def test_an_overlap_is_found(self):
        speech = speech_of(25, 25, 25, 25)
        first, _ = pack(speech, 50)
        with pytest.raises(ValueError, match="where 2 was expected"):
            check_cover(speech, [first, first])

    def test_another_party_than_the_speech_is_found(self):
        speech = speech_of(25, 25)
        (passage,) = pack(speech, 200)
        other = passage.model_copy(update={"party": "M"})
        with pytest.raises(ValueError, match="other filter fields"):
            check_cover(speech, [other])

    def test_another_speaker_than_the_speech_is_found(self):
        speech = speech_of(25, 25)
        (passage,) = pack(speech, 200)
        other = passage.model_copy(update={"person_id": "9876543210987"})
        with pytest.raises(ValueError, match="other filter fields"):
            check_cover(speech, [other])

    def test_another_date_than_the_speech_is_found(self):
        speech = speech_of(25, 25)
        (passage,) = pack(speech, 200)
        other = passage.model_copy(update={"date": speech.date + timedelta(days=1)})
        with pytest.raises(ValueError, match="other filter fields"):
            check_cover(speech, [other])

    def test_a_passage_of_another_speech_is_found(self):
        speech = speech_of(25, 25)
        (passage,) = pack(speech_of(25, 25, anf_nummer=2), 200)
        with pytest.raises(ValueError, match="H209100-2:0-1 does not belong to speech H209100-1"):
            check_cover(speech, [passage])

    def test_a_word_count_that_differs_from_the_paragraphs_is_found(self):
        speech = speech_of(25, 25)
        (passage,) = pack(speech, 200)
        other = passage.model_copy(update={"words": 49})
        with pytest.raises(ValueError, match="counts 49 words; its paragraphs hold 50"):
            check_cover(speech, [other])

    def test_a_passage_that_reaches_past_the_last_paragraph_is_found(self):
        speech = speech_of(25, 25)
        beyond = Passage(
            passage_id=passage_id(speech.anf_id, 0, 2),
            anf_id=speech.anf_id,
            first=0,
            last=2,
            words=75,
            party=speech.party,
            date=speech.date,
            person_id=speech.person_id,
        )
        with pytest.raises(ValueError, match="reaches past the 2 paragraphs"):
            check_cover(speech, [beyond])


class TestPassageParquet:
    def passages(self) -> list[Passage]:
        return [
            p
            for speech in (
                speech_of(25, 25, 25, 25, anf_nummer=1),
                speech_of(40, anf_nummer=2, party="SD", person_id=None),
            )
            for p in pack(speech, 50)
        ]

    def test_passages_survive_a_round_trip_unchanged(self, tmp_path):
        passages = self.passages()
        path = tmp_path / "search" / "passages.parquet"
        written = write_passages(path, [passages[:1], [], passages[1:]])
        assert written == len(passages)
        assert list(iter_passages(path)) == passages

    def test_a_batch_source_that_fails_leaves_no_file_behind(self, tmp_path):
        path = tmp_path / "passages.parquet"

        def batches():
            yield self.passages()
            raise RuntimeError("stopped halfway")

        with pytest.raises(RuntimeError, match="stopped halfway"):
            write_passages(path, batches())
        assert not path.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_failed_write_leaves_the_previous_file_untouched(self, tmp_path):
        path = tmp_path / "passages.parquet"
        write_passages(path, [self.passages()])
        before = path.read_bytes()

        def batches():
            yield self.passages()[:1]
            raise RuntimeError("stopped halfway")

        with pytest.raises(RuntimeError):
            write_passages(path, batches())
        assert path.read_bytes() == before


class TestBuildPassages:
    def corpus(self) -> list[Anforande]:
        return [
            speech_of(25, 25, 25, 25, anf_nummer=1, party="S", day=date(2005, 3, 1)),
            speech_of(30, anf_nummer=2, party="M", day=date(2012, 3, 1)),
            speech_of(500, 40, 40, anf_nummer=3, party="SD", day=date(2020, 3, 1)),
            speech_of(10, 10, anf_nummer=4, party="S", day=date(2020, 4, 1)),
        ]

    def build(self, tmp_path: Path, speeches: list[Anforande] | None = None, **changes):
        speeches = speeches if speeches is not None else self.corpus()
        source = tmp_path / "anforanden.parquet"
        write_speeches(source, [speeches])
        arguments = {
            "speeches_path": source,
            "destination": tmp_path / "passages.parquet",
            "target_words": 100,
            "trial_targets": (50, 400),
            "corpus": totals_of(speeches),
            "inputs_sha256": "abc",
        }
        arguments.update(changes)
        return build_passages(**arguments)

    def test_the_passages_cover_every_paragraph_and_word_of_the_corpus(self, tmp_path):
        report = self.build(tmp_path)
        speeches = list(iter_speeches(tmp_path / "anforanden.parquet"))
        passages = list(iter_passages(tmp_path / "passages.parquet"))

        assert report.cover.exact
        assert report.cover.build_report == totals_of(speeches)
        assert report.cover.read == report.cover.in_passages == totals_of(speeches)
        assert len(passages) == report.passages
        for speech in speeches:
            mine = [p for p in passages if p.anf_id == speech.anf_id]
            check_cover(speech, mine)

    def test_the_report_counts_speeches_and_passages(self, tmp_path):
        report = self.build(tmp_path)
        assert report.target_words == 100
        assert report.inputs_sha256 == "abc"
        assert report.speeches == 4
        # One passage each for the speeches of 100, 30 and 20 words; the speech of 580 words
        # wants six passages but has three paragraphs, so it gets three.
        assert report.passages == 6

    def test_the_report_groups_passages_by_party_and_by_period(self, tmp_path):
        report = self.build(tmp_path)
        assert sum(tally.passages for tally in report.passages_by_party.values()) == report.passages
        assert report.passages_by_party == {
            "S": GroupTally(passages=2, words=120),
            "M": GroupTally(passages=1, words=30),
            "SD": GroupTally(passages=3, words=580),
        }
        assert list(report.passages_by_period) == ["2003-2009", "2010-2017", "2018-2020"]
        assert report.passages_by_period["2003-2009"].words == 100
        assert report.passages_by_period["2010-2017"].words == 30
        assert report.passages_by_period["2018-2020"].words == 600

    def test_the_last_period_ends_in_2026_when_the_latest_speech_is_from_2026(self, tmp_path):
        speeches = [*self.corpus(), speech_of(15, anf_nummer=5, day=date(2026, 11, 18))]
        report = self.build(tmp_path, speeches)
        assert list(report.passages_by_period) == ["2003-2009", "2010-2017", "2018-2026"]

    def test_a_speech_of_the_session_2026_27_dated_2027_is_packed(self, tmp_path):
        late = speech_of(15, anf_nummer=5, day=date(2027, 6, 15))
        assert late.rm == "2026/27"
        self.build(tmp_path, [*self.corpus(), late])
        passages = list(iter_passages(tmp_path / "passages.parquet"))
        assert [p.passage_id for p in passages if p.anf_id == late.anf_id] == ["H209100-5:0-0"]

    def test_the_last_period_ends_in_2027_when_the_latest_speech_is_from_2027(self, tmp_path):
        late = speech_of(15, anf_nummer=5, day=date(2027, 6, 15))
        report = self.build(tmp_path, [*self.corpus(), late])
        assert list(report.passages_by_period) == ["2003-2009", "2010-2017", "2018-2027"]
        # The speeches of 580 and 20 words from 2020, in four passages, and the one from 2027.
        assert report.passages_by_period["2018-2027"] == GroupTally(passages=5, words=615)

    def test_the_last_period_ends_with_the_latest_year_wherever_that_speech_stands(self, tmp_path):
        late = speech_of(15, anf_nummer=5, day=date(2027, 6, 15))
        report = self.build(tmp_path, [late, *self.corpus()])
        assert list(report.passages_by_period) == ["2003-2009", "2010-2017", "2018-2027"]

    def test_a_passage_with_another_speaker_stops_the_build_and_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        def with_another_speaker(speech: Anforande, target_words: int) -> tuple[Passage, ...]:
            return tuple(
                passage.model_copy(update={"person_id": "9876543210987"})
                for passage in pack(speech, target_words)
            )

        monkeypatch.setattr(passages_module, "pack", with_another_speaker)
        with pytest.raises(ValueError, match="H209100-1:0-3 carries other filter fields"):
            self.build(tmp_path)
        assert not (tmp_path / "passages.parquet").exists()
        assert not list(tmp_path.glob("*.part"))

    def test_passages_that_leave_out_a_paragraph_stop_the_build_and_keep_the_earlier_file(
        self, tmp_path, monkeypatch
    ):
        self.build(tmp_path)
        earlier = (tmp_path / "passages.parquet").read_bytes()

        def without_the_last_passage(speech: Anforande, target_words: int) -> tuple[Passage, ...]:
            packed = pack(speech, target_words)
            return packed[:-1] if len(packed) > 1 else packed

        monkeypatch.setattr(passages_module, "pack", without_the_last_passage)
        with pytest.raises(ValueError, match="The passages of H209100-3 end at paragraph 1 of 3"):
            self.build(tmp_path)
        assert (tmp_path / "passages.parquet").read_bytes() == earlier
        assert not list(tmp_path.glob("*.part"))

    def test_the_report_summarises_passage_sizes(self, tmp_path):
        report = self.build(tmp_path)
        # The passages hold 20, 30, 40, 40, 100 and 500 words, in 1, 1, 3 and 1 per speech.
        assert report.words_per_passage == WordsPerPassage(
            min=20, p10=20, median=40.0, mean=121.7, p90=100, p99=100, max=500
        )
        assert report.passages_per_speech == PassagesPerSpeech(mean=1.5, median=1.0, p90=1, max=3)

    def test_the_report_counts_the_passages_beyond_each_size_limit(self, tmp_path):
        report = self.build(tmp_path)
        assert set(report.size_bands) == {"under_50_words", "over_320_words", "over_512_words"}
        assert report.size_bands["under_50_words"] == Band(passages=4, share=0.666667)
        assert report.size_bands["over_320_words"] == Band(passages=1, share=0.166667)
        assert report.size_bands["over_512_words"] == Band(passages=0, share=0.0)

    def test_the_report_names_the_largest_passages_largest_first(self, tmp_path):
        report = self.build(tmp_path)
        largest = report.largest_passages
        assert largest[0].passage_id == "H209100-3:0-0"
        assert (largest[0].words, largest[0].paragraphs) == (500, 1)
        assert [entry.words for entry in largest] == sorted(
            (entry.words for entry in largest), reverse=True
        )
        assert len(largest) <= LARGEST_KEPT

    def test_only_the_ten_largest_are_kept_and_ties_are_ordered_by_key(self, tmp_path):
        speeches = [speech_of(60, anf_nummer=number) for number in range(1, 16)]
        report = self.build(tmp_path, speeches, target_words=60, trial_targets=())
        keys = [entry.passage_id for entry in report.largest_passages]
        assert len(keys) == LARGEST_KEPT
        assert keys == sorted(keys)
        assert keys == sorted(f"H209100-{number}:0-0" for number in range(1, 16))[:LARGEST_KEPT]

    def test_the_report_counts_what_other_targets_would_give_without_writing_them(self, tmp_path):
        report = self.build(tmp_path)
        assert report.trial_targets == (
            TrialTarget(target_words=50, passages=7, median_words=40.0),
            TrialTarget(target_words=400, passages=4, median_words=65.0),
        )
        assert {path.name for path in tmp_path.glob("*.parquet")} == {
            "anforanden.parquet",
            "passages.parquet",
        }

    def test_the_same_inputs_give_the_same_report(self, tmp_path):
        first = self.build(tmp_path)
        second = self.build(tmp_path)
        assert first == second
        assert first.model_dump_json() == second.model_dump_json()

    def test_totals_that_differ_from_the_build_report_stop_the_build_and_write_nothing(
        self, tmp_path
    ):
        wrong = totals_of(self.corpus()).model_copy(update={"words": 1})
        with pytest.raises(ValueError, match="do not cover the corpus"):
            self.build(tmp_path, corpus=wrong)
        assert not (tmp_path / "passages.parquet").exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_speech_dated_before_the_first_period_stops_the_build_and_is_named(self, tmp_path):
        speeches = [speech_of(10, anf_nummer=7, day=date(2002, 10, 1))]
        with pytest.raises(ValueError, match="H209100-7: 2002-10-01 lies before 2003"):
            self.build(tmp_path, speeches)
        assert not (tmp_path / "passages.parquet").exists()
        assert not list(tmp_path.glob("*.part"))

    def test_a_corpus_without_speeches_stops_the_build_and_keeps_the_earlier_file(self, tmp_path):
        self.build(tmp_path)
        earlier = (tmp_path / "passages.parquet").read_bytes()
        with pytest.raises(ValueError, match="reported no speeches"):
            self.build(tmp_path, [])
        assert (tmp_path / "passages.parquet").read_bytes() == earlier
        assert not list(tmp_path.glob("*.part"))


class TestReportHelpers:
    def test_a_percentile_is_the_value_at_the_floored_index(self):
        values = list(range(1, 101))
        assert percentile(values, 10) == 10
        assert percentile(values, 50) == 50
        assert percentile(values, 99) == 99
        assert percentile([7], 99) == 7
        assert percentile([1, 2], 90) == 1

    @pytest.mark.parametrize(
        ("day", "first_year"),
        [
            (date(2003, 9, 16), 2003),
            (date(2009, 12, 31), 2003),
            (date(2010, 1, 1), 2010),
            (date(2017, 12, 31), 2010),
            (date(2018, 1, 1), 2018),
            (date(2026, 11, 18), 2018),
            (date(2027, 1, 20), 2018),
            (date(2027, 9, 30), 2018),
        ],
    )
    def test_a_date_falls_in_the_period_of_its_year(self, day, first_year):
        assert period_of(day) == first_year

    def test_a_date_before_the_first_period_raises(self):
        with pytest.raises(ValueError, match="2002-12-31 lies before 2003"):
            period_of(date(2002, 12, 31))

    def test_the_sessions_in_the_shipped_settings_run_from_the_first_period_into_the_last(
        self, tmp_path
    ):
        settings = load_settings(data_root=tmp_path)
        first_day, _ = session_bounds(settings.corpus.first_session)
        _, last_day = session_bounds(settings.source.last_session)
        assert period_of(first_day) == PERIOD_FIRST_YEARS[0]
        assert period_of(last_day) == PERIOD_FIRST_YEARS[-1]

    @pytest.mark.parametrize("latest_year", [2026, 2027])
    def test_the_first_two_periods_keep_their_labels_whatever_the_latest_year(self, latest_year):
        assert period_label(2003, latest_year) == "2003-2009"
        assert period_label(2010, latest_year) == "2010-2017"

    @pytest.mark.parametrize(("latest_year", "label"), [(2026, "2018-2026"), (2027, "2018-2027")])
    def test_the_label_of_the_last_period_ends_with_the_latest_year(self, latest_year, label):
        assert period_label(2018, latest_year) == label
