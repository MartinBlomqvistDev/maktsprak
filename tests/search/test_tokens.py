"""Tests for search/corpus/tokens.py and search/pipeline/count_tokens.py.

They build a tiny tokenizer in memory, and nothing here reaches the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from search.corpus.tokens import (
    INPUT_LIMIT_TOKENS,
    LengthSpread,
    model_type_of,
    percentile,
    serialised_sha256_of,
    spread_of,
    token_lengths,
)

tokenizers = pytest.importorskip("tokenizers")

VOCAB = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "herr": 3, "talman": 4, "tack": 5}


def word_level(
    with_special_tokens: bool = False, extra_word: str | None = None
) -> tokenizers.Tokenizer:
    """Build a tokenizer that gives one token per whitespace-separated word.

    Args:
        with_special_tokens: Whether the tokenizer wraps a text in ``[CLS]`` and ``[SEP]``.
        extra_word: A word to add to the vocabulary.

    Returns:
        The tokenizer, with truncation and padding switched off.
    """
    vocab = VOCAB if extra_word is None else {**VOCAB, extra_word: len(VOCAB)}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    if with_special_tokens:
        tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
        )
    return tokenizer


class TestTokenLengths:
    def test_counts_one_token_per_word_with_this_tokenizer(self):
        lengths = list(
            token_lengths(["herr talman", "tack", "herr talman tack tack"], word_level())
        )
        assert lengths == [2, 1, 4]

    def test_a_word_outside_the_vocabulary_is_one_token(self):
        assert list(token_lengths(["herr okänt"], word_level())) == [2]

    def test_an_empty_text_has_no_tokens(self):
        assert list(token_lengths([""], word_level())) == [0]

    def test_special_tokens_are_left_out(self):
        tokenizer = word_level(with_special_tokens=True)
        assert tokenizer.encode("herr talman").ids == [1, 3, 4, 2]
        assert list(token_lengths(["herr talman"], tokenizer)) == [2]

    def test_order_is_kept_across_batches(self):
        texts = ["herr", "herr herr", "herr herr herr", "tack", "tack tack"]
        assert list(token_lengths(texts, word_level(), batch_size=2)) == [1, 2, 3, 1, 2]

    def test_takes_any_iterable_of_texts(self):
        texts = (text for text in ["herr", "tack tack"])
        assert list(token_lengths(texts, word_level(), batch_size=1)) == [1, 2]

    def test_a_tokenizer_that_truncates_is_refused(self):
        tokenizer = word_level()
        tokenizer.enable_truncation(max_length=2)
        with pytest.raises(ValueError, match="truncates or pads"):
            list(token_lengths(["herr talman tack"], tokenizer))

    def test_a_tokenizer_that_pads_is_refused(self):
        tokenizer = word_level()
        tokenizer.enable_padding(length=8)
        with pytest.raises(ValueError, match="truncates or pads"):
            list(token_lengths(["herr"], tokenizer))

    def test_a_tokenizer_with_both_switched_off_is_accepted(self):
        tokenizer = word_level()
        tokenizer.enable_truncation(max_length=2)
        tokenizer.enable_padding(length=8)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        assert list(token_lengths(["herr talman tack"], tokenizer)) == [3]

    def test_a_batch_size_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            list(token_lengths(["herr"], word_level(), batch_size=0))


class TestPercentile:
    VALUES = list(range(1, 11))

    def test_uses_the_nearest_rank(self):
        assert percentile(self.VALUES, 0.5) == 5
        assert percentile(self.VALUES, 0.9) == 9
        assert percentile(self.VALUES, 0.99) == 10

    def test_the_ends_are_the_smallest_and_largest_values(self):
        assert percentile(self.VALUES, 0.0) == 1
        assert percentile(self.VALUES, 1.0) == 10

    def test_one_value_is_every_percentile(self):
        assert percentile([7], 0.5) == 7
        assert percentile([7], 0.99) == 7

    def test_an_even_count_takes_the_lower_middle_as_the_median(self):
        assert percentile([1, 2, 3, 4], 0.5) == 2

    def test_no_values_is_an_error(self):
        with pytest.raises(ValueError, match="at least one"):
            percentile([], 0.5)

    @pytest.mark.parametrize("share", [-0.1, 1.1])
    def test_a_share_outside_zero_to_one_is_an_error(self, share):
        with pytest.raises(ValueError, match="between 0 and 1"):
            percentile(self.VALUES, share)


class TestSpreadOf:
    def test_summarises_unsorted_counts(self):
        spread = spread_of([10, 1, 4, 2, 3, 5, 6, 7, 8, 9])
        assert spread == LengthSpread(median=5, p90=9, p99=10, maximum=10, over_input_limit=0)

    def test_counts_the_texts_over_the_input_limit(self):
        lengths = [3, INPUT_LIMIT_TOKENS, INPUT_LIMIT_TOKENS + 1, 900]
        assert spread_of(lengths).over_input_limit == 2

    def test_a_text_of_511_or_512_tokens_is_not_over_the_limit_as_counted_without_special_tokens(
        self,
    ):
        lengths = [INPUT_LIMIT_TOKENS - 1, INPUT_LIMIT_TOKENS]
        assert spread_of(lengths).over_input_limit == 0

    def test_the_maximum_is_the_longest_text(self):
        assert spread_of([5, 1200, 7]).maximum == 1200

    def test_no_counts_is_an_error(self):
        with pytest.raises(ValueError):
            spread_of([])

    def test_a_spread_cannot_be_changed(self):
        spread = spread_of([1, 2, 3])
        with pytest.raises(ValidationError):
            spread.median = 9


class TestModelType:
    def test_reads_the_algorithm_from_the_tokenizer(self):
        assert model_type_of(word_level()) == "WordLevel"


class TestSerialisedHash:
    def test_is_the_sha256_of_the_serialised_form_as_utf8(self):
        tokenizer = word_level()
        expected = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
        assert serialised_sha256_of(tokenizer) == expected

    def test_the_same_tokenizer_built_twice_has_the_same_hash(self):
        assert serialised_sha256_of(word_level()) == serialised_sha256_of(word_level())

    def test_a_tokenizer_read_back_from_its_serialised_form_has_the_same_hash(self):
        tokenizer = word_level()
        again = tokenizers.Tokenizer.from_str(tokenizer.to_str())
        assert serialised_sha256_of(again) == serialised_sha256_of(tokenizer)

    def test_another_vocabulary_gives_another_hash(self):
        assert serialised_sha256_of(word_level(extra_word="fru")) != serialised_sha256_of(
            word_level()
        )

    def test_a_tokenizer_that_adds_special_tokens_has_another_hash(self):
        plain = serialised_sha256_of(word_level())
        assert serialised_sha256_of(word_level(with_special_tokens=True)) != plain

    def test_truncation_switched_on_gives_another_hash_until_it_is_switched_off(self):
        tokenizer = word_level()
        plain = serialised_sha256_of(tokenizer)
        tokenizer.enable_truncation(max_length=2)
        assert serialised_sha256_of(tokenizer) != plain
        tokenizer.no_truncation()
        assert serialised_sha256_of(tokenizer) == plain

    def test_padding_switched_on_gives_another_hash_until_it_is_switched_off(self):
        tokenizer = word_level()
        plain = serialised_sha256_of(tokenizer)
        tokenizer.enable_padding(length=8)
        assert serialised_sha256_of(tokenizer) != plain
        tokenizer.no_padding()
        assert serialised_sha256_of(tokenizer) == plain

    def test_a_saved_file_can_hash_differently_from_the_tokenizer_read_from_it(self, tmp_path):
        path = tmp_path / "tokenizer.json"
        word_level().save(str(path), pretty=True)
        read_back = tokenizers.Tokenizer.from_file(str(path))
        assert serialised_sha256_of(read_back) == serialised_sha256_of(word_level())
        assert serialised_sha256_of(read_back) != hashlib.sha256(path.read_bytes()).hexdigest()


def load_count_script():
    """Import search/pipeline/count_tokens.py, whose name is not a module name."""
    path = Path(__file__).resolve().parents[2] / "search" / "pipeline" / "count_tokens.py"
    spec = importlib.util.spec_from_file_location("count_tokens", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCountScript:
    """Where the script looks for the classifier's tokenizer."""

    script = load_count_script()

    @pytest.fixture
    def classifier_directory(self, tmp_path, monkeypatch):
        """Run the script's main with a given command line; return the directory it loads."""
        loaded: list[Path] = []

        class Stop(Exception):
            pass

        def fake_classifier(directory: Path):
            loaded.append(directory)
            return object(), "classifier"

        def stop(*arguments, **keywords):
            raise Stop

        monkeypatch.setattr(self.script, "load_classifier_tokenizer", fake_classifier)
        monkeypatch.setattr(self.script, "load_xlm_roberta_tokenizer", lambda: (object(), "xlm"))
        monkeypatch.setattr(self.script, "count_corpus", stop)

        def run_with(*options: str) -> Path:
            argv = ["count_tokens.py", "--data-root", str(tmp_path), *options]
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(Stop):
                self.script.main()
            return loaded[-1]

        return run_with

    def test_the_classifier_directory_defaults_to_the_data_setting(
        self, classifier_directory, tmp_path
    ):
        assert classifier_directory() == tmp_path / "data" / "models" / "party_classifier"

    def test_the_classifier_directory_can_be_given_on_the_command_line(
        self, classifier_directory, tmp_path
    ):
        other = tmp_path / "elsewhere"
        assert classifier_directory("--classifier-dir", str(other)) == other


class TestCountCorpus:
    """What the script records for each tokenizer, counted over a corpus of two speeches."""

    script = load_count_script()

    @pytest.fixture
    def corpus(self, tmp_path) -> Path:
        """Write a Parquet file of two speeches that holds the paragraphs column alone."""
        path = tmp_path / "anforanden.parquet"
        table = pa.table(
            {"paragraphs": [["herr talman", "tack"], ["herr talman tack tack"]]},
            schema=pa.schema([("paragraphs", pa.list_(pa.string()))]),
        )
        pq.write_table(table, path)
        return path

    def test_each_count_records_the_hash_of_its_tokenizer_in_a_field_of_its_own(self, corpus):
        tokenizer = word_level()
        report = self.script.count_corpus(corpus, {"word_level": (tokenizer, "in memory")}, {})
        (entry,) = report.tokenizers
        assert entry.serialised_sha256 == serialised_sha256_of(tokenizer)
        assert (entry.total_tokens, report.words, report.paragraphs) == (7, 7, 3)
        (written,) = json.loads(report.model_dump_json())["tokenizers"]
        assert written["serialised_sha256"] == serialised_sha256_of(tokenizer)
        assert written["source"] == "in memory"

    def test_two_tokenizers_with_the_same_counts_are_told_apart_by_their_hashes(self, corpus):
        plain, other = word_level(), word_level(extra_word="fru")
        report = self.script.count_corpus(
            corpus, {"plain": (plain, "in memory"), "other": (other, "in memory")}, {}
        )
        first, second = report.tokenizers
        assert (first.source, first.total_tokens) == (second.source, second.total_tokens)
        assert first.serialised_sha256 == serialised_sha256_of(plain)
        assert second.serialised_sha256 == serialised_sha256_of(other)
        assert first.serialised_sha256 != second.serialised_sha256

    def test_the_reason_a_tokenizer_is_unavailable_is_kept_in_the_report(self, corpus):
        reasons = {"xlm_roberta_sentencepiece": "OSError while loading it."}
        report = self.script.count_corpus(
            corpus, {"word_level": (word_level(), "in memory")}, reasons
        )
        assert report.unavailable == reasons


class TestUnavailableTokenizer:
    """What the script records when the XLM-RoBERTa tokenizer cannot be loaded."""

    script = load_count_script()

    @pytest.fixture
    def unloadable(self, monkeypatch) -> None:
        """Make ``Tokenizer.from_pretrained`` raise the error of a file that cannot be read."""

        class Unloadable:
            @staticmethod
            def from_pretrained(repository: str) -> None:
                raise OSError(f"{repository} cannot be read")

        monkeypatch.setattr(self.script, "Tokenizer", Unloadable)

    def test_the_reason_names_the_error_and_says_it_came_while_loading(self, unloadable):
        with pytest.raises(self.script.TokenizerUnavailable, match=r"^OSError while loading it\.$"):
            self.script.load_xlm_roberta_tokenizer()

    def test_the_count_is_given_the_reason_and_the_tokenizer_that_did_load(
        self, unloadable, tmp_path, monkeypatch
    ):
        counted: dict[str, object] = {}

        class Stop(Exception):
            pass

        def record(path: Path, loaded: dict[str, object], unavailable: dict[str, str]) -> None:
            counted.update(loaded=sorted(loaded), unavailable=dict(unavailable))
            raise Stop

        monkeypatch.setattr(
            self.script, "load_classifier_tokenizer", lambda directory: (word_level(), "in memory")
        )
        monkeypatch.setattr(self.script, "count_corpus", record)
        monkeypatch.setattr(sys, "argv", ["count_tokens.py", "--data-root", str(tmp_path)])
        with pytest.raises(Stop):
            self.script.main()
        assert counted["loaded"] == [self.script.CLASSIFIER_NAME]
        assert counted["unavailable"] == {self.script.XLM_ROBERTA_NAME: "OSError while loading it."}


class TestClassifierTokenizer:
    """How the classifier's tokenizer is read from a model directory."""

    script = load_count_script()

    @pytest.fixture
    def directory(self, tmp_path) -> Path:
        """A model directory holding a tokenizer that can be read."""
        word_level().save(str(tmp_path / "tokenizer.json"))
        return tmp_path

    def test_a_tokenizer_file_that_cannot_be_parsed_is_unavailable_with_the_reason(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("not a tokenizer", encoding="utf-8")
        with pytest.raises(self.script.TokenizerUnavailable, match=r"^\w+ while loading it\.$"):
            self.script.load_classifier_tokenizer(tmp_path)

    def test_the_model_named_in_the_config_is_part_of_the_source(self, directory):
        (directory / "config.json").write_text(
            json.dumps({"_name_or_path": "KB/bert-base-swedish-cased"}), encoding="utf-8"
        )
        _, source = self.script.load_classifier_tokenizer(directory)
        assert source.endswith(", the tokenizer of KB/bert-base-swedish-cased")

    @pytest.mark.parametrize("content", ["{not json", "[]", '{"_name_or_path": 5}', "{}"])
    def test_a_config_without_a_usable_model_name_leaves_the_name_out(self, directory, content):
        (directory / "config.json").write_text(content, encoding="utf-8")
        tokenizer, source = self.script.load_classifier_tokenizer(directory)
        assert source == f"local file {directory.name}/tokenizer.json"
        assert tokenizer.get_vocab_size() == len(VOCAB)

    def test_a_config_that_is_not_text_leaves_the_name_out(self, directory):
        (directory / "config.json").write_bytes(b"\xff\xfe\x00\x80")
        _, source = self.script.load_classifier_tokenizer(directory)
        assert source == f"local file {directory.name}/tokenizer.json"

    def test_the_count_records_a_damaged_tokenizer_file_as_unavailable(self, tmp_path, monkeypatch):
        classifier = tmp_path / "classifier"
        classifier.mkdir()
        (classifier / "tokenizer.json").write_text("{", encoding="utf-8")
        counted: dict[str, object] = {}

        class Stop(Exception):
            pass

        def record(path: Path, loaded: dict[str, object], unavailable: dict[str, str]) -> None:
            counted.update(loaded=sorted(loaded), unavailable=dict(unavailable))
            raise Stop

        monkeypatch.setattr(
            self.script, "load_xlm_roberta_tokenizer", lambda: (word_level(), "in memory")
        )
        monkeypatch.setattr(self.script, "count_corpus", record)
        monkeypatch.setattr(
            sys,
            "argv",
            ["count_tokens.py", "--data-root", str(tmp_path), "--classifier-dir", str(classifier)],
        )
        with pytest.raises(Stop):
            self.script.main()
        assert counted["loaded"] == [self.script.XLM_ROBERTA_NAME]
        assert list(counted["unavailable"]) == [self.script.CLASSIFIER_NAME]
        assert str(counted["unavailable"][self.script.CLASSIFIER_NAME]).endswith(
            " while loading it."
        )
