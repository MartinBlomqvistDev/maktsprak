"""Tests for search/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from search.config import DATA_ROOT_ENV, REPO_ROOT, load_settings

VALID_YAML = (
    "source:\n"
    "  dataset_url: https://example.org/anforande-{session}.json.zip\n"
    '  first_session: "2002/03"\n'
    '  last_session: "2003/04"\n'
    "corpus:\n"
    '  first_session: "2003/04"\n'
    "data:\n"
    "  raw_anforanden: data/raw_anforanden\n"
    "  search: data/search\n"
    "  legacy_corpus: data/parquet/speeches_full.parquet\n"
    "  party_classifier: data/models/party_classifier\n"
    "tracked:\n"
    "  reports: data/reports/search\n"
    "  questions: data/reports/search/questions\n"
    "passages:\n"
    "  target_words: 200\n"
    "  trial_targets: [120, 320]\n"
    "reconcile:\n"
    "  coverage_threshold: 0.99\n"
    "  edge_tokens: 8\n"
    "  min_sequence_share: 0.6\n"
    "  min_shared_tokens: 5\n"
    "  missing_tokens: 5\n"
    "  surplus_tokens: 20\n"
    "  listed_ids: 60\n"
    "  listed_deviations: 50\n"
    "  listed_missing: 25\n"
    "  listed_tokens: 30\n"
    "links:\n"
    "  seed: 2026\n"
    "  max_requests: 60\n"
    "questions:\n"
    "  seed: 11\n"
    "  pilot_seed: 12\n"
    "  review_seed: 13\n"
    "  sizes: {natural: 30, filtered: 10, stance: 5}\n"
    "  pilot_sizes: {natural: 2, stance: 1}\n"
    "  unanswerable: {off_topic: 5, unknown_speaker: 2, sd_before_2010: 2, after_index: 5}\n"
    "  unanswerable_asked: {off_topic: 6, unknown_speaker: 3, sd_before_2010: 3, after_index: 6}\n"
    "  pilot_unanswerable_asked: 2\n"
    "  rare_share: 0.001\n"
    "  max_passages_per_slot: 12\n"
    "  interpellation_sessions_per_period: 1\n"
    "  pilot_budget_usd: 1.0\n"
    "generation:\n"
    "  model: claude-opus-5-5\n"
    "  effort: low\n"
    "  max_tokens: 8000\n"
    "  batch_discount: 0.5\n"
    "  prices:\n"
    "    claude-opus-5-5: {input: 4.0, output: 20.0}\n"
    "judge:\n"
    "  model: claude-opus-5-5\n"
    "  effort: low\n"
    "  max_tokens: 4000\n"
    "  seed: 14\n"
    "baseline:\n"
    "  k1: [0.9, 1.2]\n"
    "  b: [0.5, 0.75]\n"
    "  resamples: 1000\n"
    "  bootstrap_seed: 15\n"
)


def _key_rows(key: str, section: str | None) -> tuple[list[str], list[bool]]:
    """The rows of the valid settings, and which of them hold ``  key:``, in *section* if given."""
    rows = VALID_YAML.splitlines(keepends=True)
    current = ""
    marks = []
    for row in rows:
        if not row.startswith(" "):
            current = row.split(":")[0]
        marks.append(row.startswith(f"  {key}:") and section in (None, current))
    assert sum(marks) == 1
    return rows, marks


def without_key(key: str, section: str | None = None) -> str:
    """The valid settings without the row ``  key:``, looked for in *section* if given."""
    rows, marks = _key_rows(key, section)
    return "".join(row for row, marked in zip(rows, marks, strict=True) if not marked)


def with_line_replaced(key: str, line: str, section: str | None = None) -> str:
    """The valid settings with the row ``  key:`` replaced by ``line``, in *section* if given."""
    rows, marks = _key_rows(key, section)
    return "".join(line if marked else row for row, marked in zip(rows, marks, strict=True))


def error_places(error: ValidationError) -> list[tuple[tuple[int | str, ...], str]]:
    """The place and the kind of every problem a validation error holds."""
    return [(item["loc"], item["type"]) for item in error.errors()]


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A directory standing in for a checkout that holds the bulk data."""
    return tmp_path.resolve()


class TestDataRoot:
    def test_bulk_data_resolves_against_the_data_root(self, data_root):
        settings = load_settings(data_root=data_root)
        assert settings.data.raw_anforanden == data_root / "data" / "raw_anforanden"
        assert settings.data.search == data_root / "data" / "search"
        assert settings.data.legacy_corpus == (
            data_root / "data" / "parquet" / "speeches_full.parquet"
        )
        assert settings.data.party_classifier == (
            data_root / "data" / "models" / "party_classifier"
        )

    def test_tracked_output_stays_in_the_repository(self, data_root):
        settings = load_settings(data_root=data_root)
        assert settings.tracked.reports == REPO_ROOT / "data" / "reports" / "search"

    def test_environment_sets_the_data_root(self, data_root, monkeypatch):
        monkeypatch.setenv(DATA_ROOT_ENV, str(data_root))
        assert load_settings().data.search == data_root / "data" / "search"

    def test_argument_wins_over_the_environment(self, data_root, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv(DATA_ROOT_ENV, str(other))
        assert load_settings(data_root=data_root).data.search == data_root / "data" / "search"

    def test_defaults_to_the_repository_root(self, monkeypatch):
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        assert load_settings().data.search == REPO_ROOT / "data" / "search"

    def test_a_missing_data_root_raises(self, data_root):
        with pytest.raises(FileNotFoundError):
            load_settings(data_root=data_root / "absent")


class TestValidation:
    def test_an_unknown_section_raises(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML + "surprise: 1\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(config_file=config, data_root=data_root)

    def test_an_unknown_path_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace(
            "  reports: data/reports/search\n",
            "  reports: data/reports/search\n  extra: data/extra\n",
        )
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(config_file=config, data_root=data_root)

    def test_a_missing_path_raises(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML.replace("  search: data/search\n", ""), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(config_file=config, data_root=data_root)

    def test_the_sessions_fetched_and_the_sessions_kept_are_read(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML, encoding="utf-8")
        settings = load_settings(config_file=config, data_root=data_root)
        assert (settings.source.first_session, settings.source.last_session) == (
            "2002/03",
            "2003/04",
        )
        assert settings.corpus.first_session == "2003/04"

    def test_the_shipped_corpus_starts_with_the_2003_04_session(self, data_root):
        settings = load_settings(data_root=data_root)
        assert settings.source.first_session == "2002/03"
        assert settings.corpus.first_session == "2003/04"

    def test_a_missing_corpus_section_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace('corpus:\n  first_session: "2003/04"\n', "")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("corpus",), "missing")]

    def test_an_unknown_corpus_setting_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace(
            'corpus:\n  first_session: "2003/04"\n',
            'corpus:\n  first_session: "2003/04"\n  last_session: "2026/27"\n',
        )
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("corpus", "last_session"), "extra_forbidden")]

    def test_the_passage_settings_are_read(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML, encoding="utf-8")
        settings = load_settings(config_file=config, data_root=data_root)
        assert settings.passages.target_words == 200
        assert settings.passages.trial_targets == (120, 320)

    def test_the_shipped_settings_pack_towards_200_words(self, data_root):
        assert load_settings(data_root=data_root).passages.target_words == 200

    def test_an_unknown_passage_setting_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace(
            "  trial_targets: [120, 320]\n", "  trial_targets: [120, 320]\n  overlap_words: 20\n"
        )
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("passages", "overlap_words"), "extra_forbidden")]

    def test_a_missing_passages_section_raises(self, data_root):
        config = data_root / "config.yaml"
        before, after = VALID_YAML.split("passages:")
        config.write_text(before + "reconcile:" + after.split("reconcile:")[1], encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("passages",), "missing")]

    @pytest.mark.parametrize(
        "key, line",
        [
            ("target_words", "  target_words: 0\n"),
            ("target_words", "  target_words: -5\n"),
            ("trial_targets", "  trial_targets: [120, 0]\n"),
        ],
    )
    def test_a_passage_size_below_one_word_raises(self, data_root, key, line):
        config = data_root / "config.yaml"
        config.write_text(with_line_replaced(key, line), encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        places = {place for place, _ in error_places(raised.value)}
        assert places and all(place[:2] == ("passages", key) for place in places)

    @pytest.mark.parametrize(
        "key, line",
        [
            ("target_words", "  target_words: 200\n"),
            ("trial_targets", "  trial_targets: [120, 320]\n"),
            ("target_words", "  target_words: 1\n"),
            ("trial_targets", "  trial_targets: [1]\n"),
        ],
    )
    def test_a_passage_size_of_at_least_one_word_loads(self, data_root, key, line):
        config = data_root / "config.yaml"
        config.write_text(with_line_replaced(key, line), encoding="utf-8")
        assert load_settings(config_file=config, data_root=data_root).passages.target_words >= 1

    def test_the_reconcile_settings_are_read(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML, encoding="utf-8")
        reconcile = load_settings(config_file=config, data_root=data_root).reconcile
        assert reconcile.coverage_threshold == 0.99
        assert (reconcile.edge_tokens, reconcile.min_sequence_share) == (8, 0.6)
        assert (reconcile.min_shared_tokens, reconcile.missing_tokens) == (5, 5)
        assert reconcile.surplus_tokens == 20
        assert (reconcile.listed_ids, reconcile.listed_deviations) == (60, 50)
        assert (reconcile.listed_missing, reconcile.listed_tokens) == (25, 30)

    def test_the_shipped_coverage_threshold_is_the_pre_registered_99_percent(self, data_root):
        assert load_settings(data_root=data_root).reconcile.coverage_threshold == 0.99

    def test_the_shipped_alignment_and_listing_settings(self, data_root):
        reconcile = load_settings(data_root=data_root).reconcile
        assert (reconcile.edge_tokens, reconcile.min_sequence_share) == (8, 0.6)
        assert (reconcile.min_shared_tokens, reconcile.missing_tokens) == (5, 5)
        assert reconcile.surplus_tokens == 20
        assert (reconcile.listed_ids, reconcile.listed_deviations) == (60, 50)
        assert (reconcile.listed_missing, reconcile.listed_tokens) == (25, 30)

    def test_a_missing_reconcile_section_raises(self, data_root):
        config = data_root / "config.yaml"
        before, after = VALID_YAML.split("reconcile:")
        config.write_text(before + "links:" + after.split("links:")[1], encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("reconcile",), "missing")]

    @pytest.mark.parametrize(
        "key",
        [
            "coverage_threshold",
            "edge_tokens",
            "min_sequence_share",
            "min_shared_tokens",
            "missing_tokens",
            "surplus_tokens",
            "listed_ids",
            "listed_deviations",
            "listed_missing",
            "listed_tokens",
        ],
    )
    def test_a_missing_reconcile_key_raises(self, data_root, key):
        config = data_root / "config.yaml"
        config.write_text(without_key(key), encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("reconcile", key), "missing")]

    def test_an_unknown_reconcile_key_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace(
            "  listed_tokens: 30\n", "  listed_tokens: 30\n  min_overlap: 3\n"
        )
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("reconcile", "min_overlap"), "extra_forbidden")]

    @pytest.mark.parametrize(
        "key, value",
        [
            ("coverage_threshold", "0"),
            ("coverage_threshold", "1.5"),
            ("coverage_threshold", "-0.1"),
            ("min_sequence_share", "0"),
            ("min_sequence_share", "1.01"),
            ("min_sequence_share", "-0.6"),
            ("edge_tokens", "0"),
            ("min_shared_tokens", "-1"),
            ("missing_tokens", "0"),
            ("surplus_tokens", "-20"),
            ("listed_ids", "0"),
            ("listed_deviations", "-50"),
            ("listed_missing", "0"),
            ("listed_tokens", "0"),
        ],
    )
    def test_a_reconcile_value_out_of_range_raises(self, data_root, key, value):
        config = data_root / "config.yaml"
        config.write_text(with_line_replaced(key, f"  {key}: {value}\n"), encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert [place for place, _ in error_places(raised.value)] == [("reconcile", key)]

    def test_a_share_of_exactly_one_is_allowed(self, data_root):
        config = data_root / "config.yaml"
        text = with_line_replaced("min_sequence_share", "  min_sequence_share: 1\n")
        config.write_text(text, encoding="utf-8")
        settings = load_settings(config_file=config, data_root=data_root)
        assert settings.reconcile.min_sequence_share == 1.0

    def test_the_link_settings_are_read(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML, encoding="utf-8")
        links = load_settings(config_file=config, data_root=data_root).links
        assert (links.seed, links.max_requests) == (2026, 60)

    def test_the_shipped_link_settings_are_seed_2026_and_60_requests(self, data_root):
        links = load_settings(data_root=data_root).links
        assert (links.seed, links.max_requests) == (2026, 60)

    def test_a_missing_links_section_raises(self, data_root):
        config = data_root / "config.yaml"
        before, after = VALID_YAML.split("links:")
        config.write_text(before + "questions:" + after.split("questions:")[1], encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("links",), "missing")]

    @pytest.mark.parametrize("key", ["seed", "max_requests"])
    def test_a_missing_link_key_raises(self, data_root, key):
        config = data_root / "config.yaml"
        config.write_text(without_key(key, "links"), encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("links", key), "missing")]

    def test_an_unknown_link_key_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace("  max_requests: 60\n", "  max_requests: 60\n  per_session: 2\n")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert error_places(raised.value) == [(("links", "per_session"), "extra_forbidden")]

    @pytest.mark.parametrize("value", ["0", "-3"])
    def test_a_request_limit_below_one_raises(self, data_root, value):
        config = data_root / "config.yaml"
        text = with_line_replaced("max_requests", f"  max_requests: {value}\n")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert [place for place, _ in error_places(raised.value)] == [("links", "max_requests")]

    def test_settings_cannot_be_mutated(self, data_root):
        settings = load_settings(data_root=data_root)
        with pytest.raises(ValidationError):
            settings.data.search = data_root  # type: ignore[misc]


class TestEvaluationSettings:
    def test_the_question_settings_are_read(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML, encoding="utf-8")
        questions = load_settings(config_file=config, data_root=data_root).questions
        assert (questions.seed, questions.pilot_seed, questions.review_seed) == (11, 12, 13)
        assert questions.sizes == {"natural": 30, "filtered": 10, "stance": 5}
        assert questions.rare_share == 0.001

    def test_the_shipped_sizes_are_the_pre_registered_ones(self, data_root):
        questions = load_settings(data_root=data_root).questions
        assert questions.sizes == {"natural": 300, "filtered": 100, "stance": 50}
        assert sum(questions.unanswerable.values()) == 150
        assert all(
            questions.unanswerable_asked[kind] >= count
            for kind, count in questions.unanswerable.items()
        )

    def test_the_tracked_questions_stay_in_the_repository(self, data_root):
        settings = load_settings(data_root=data_root)
        assert settings.tracked.questions == REPO_ROOT / "data" / "reports" / "search" / "questions"

    def test_a_count_below_one_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace("{natural: 30, filtered: 10, stance: 5}", "{natural: 0}")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert [place for place, _ in error_places(raised.value)] == [("questions", "sizes")]

    def test_an_unknown_set_in_the_sizes_raises(self, data_root):
        config = data_root / "config.yaml"
        text = VALID_YAML.replace("{natural: 30, filtered: 10, stance: 5}", "{essay: 3}")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(config_file=config, data_root=data_root)

    def test_a_generation_model_without_a_price_raises(self, data_root):
        config = data_root / "config.yaml"
        text = with_line_replaced("model", "  model: claude-haiku-5-5\n", "generation")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert "has no price" in str(raised.value)

    def test_a_judge_model_without_a_price_raises(self, data_root):
        config = data_root / "config.yaml"
        text = with_line_replaced("model", "  model: claude-haiku-5-5\n", "judge")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert "judge's model" in str(raised.value)

    def test_an_unknown_effort_raises(self, data_root):
        config = data_root / "config.yaml"
        text = with_line_replaced("effort", "  effort: extreme\n", "judge")
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert [place for place, _ in error_places(raised.value)] == [("judge", "effort")]

    @pytest.mark.parametrize(
        ("key", "line"),
        [("b", "  b: [0.5, 1.5]\n"), ("b", "  b: []\n"), ("k1", "  k1: [-0.1]\n")],
    )
    def test_a_bm25_value_out_of_bounds_raises(self, data_root, key, line):
        config = data_root / "config.yaml"
        config.write_text(with_line_replaced(key, line, "baseline"), encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            load_settings(config_file=config, data_root=data_root)
        assert [place for place, _ in error_places(raised.value)] == [("baseline", key)]

    def test_the_shipped_grid_holds_lucenes_defaults(self, data_root):
        baseline = load_settings(data_root=data_root).baseline
        assert 1.2 in baseline.k1 and 0.75 in baseline.b
        assert baseline.resamples == 10_000
