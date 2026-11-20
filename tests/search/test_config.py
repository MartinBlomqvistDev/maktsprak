"""Tests for search/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from search.config import DATA_ROOT_ENV, REPO_ROOT, load_settings

VALID_YAML = (
    "data:\n"
    "  raw_anforanden: data/raw_anforanden\n"
    "  search: data/search\n"
    "  legacy_corpus: data/parquet/speeches_full.parquet\n"
    "tracked:\n"
    "  reports: data/reports/search\n"
)


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
        config.write_text(VALID_YAML + "  extra: data/extra\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(config_file=config, data_root=data_root)

    def test_a_missing_path_raises(self, data_root):
        config = data_root / "config.yaml"
        config.write_text(VALID_YAML.replace("  search: data/search\n", ""), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(config_file=config, data_root=data_root)

    def test_settings_cannot_be_mutated(self, data_root):
        settings = load_settings(data_root=data_root)
        with pytest.raises(ValidationError):
            settings.data.search = data_root  # type: ignore[misc]
