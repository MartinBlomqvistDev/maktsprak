"""Tests for src/maktsprak_pipeline/config.py."""

from __future__ import annotations

from pathlib import Path

from src.maktsprak_pipeline.config import (
    DB_PORT,
    PARTY_LEADERS_IDS,
    PARTY_ORDER,
    PROJECT_ROOT,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RATE_LIMIT_WAIT_SECONDS,
    RIKSDAG_BASE_URL,
    VALID_PARTIES,
)


class TestPartyConstants:
    def test_party_order_is_subset_of_valid_parties(self):
        assert set(PARTY_ORDER) <= VALID_PARTIES

    def test_valid_parties_is_frozenset(self):
        assert isinstance(VALID_PARTIES, frozenset)

    def test_party_order_length(self):
        assert len(PARTY_ORDER) == 8

    def test_party_leaders_keys_match_valid_parties(self):
        assert set(PARTY_LEADERS_IDS.keys()) == VALID_PARTIES

    def test_party_leaders_values_are_lists_of_strings(self):
        for party, ids in PARTY_LEADERS_IDS.items():
            assert isinstance(ids, list), f"{party} IDs should be a list"
            assert all(isinstance(i, str) for i in ids), f"{party} IDs should be strings"


class TestPathConstants:
    def test_project_root_is_path(self):
        assert isinstance(PROJECT_ROOT, Path)

    def test_raw_data_path_is_path(self):
        assert isinstance(RAW_DATA_PATH, Path)

    def test_processed_data_path_is_path(self):
        assert isinstance(PROCESSED_DATA_PATH, Path)

    def test_paths_are_absolute(self):
        assert PROJECT_ROOT.is_absolute()
        assert RAW_DATA_PATH.is_absolute()
        assert PROCESSED_DATA_PATH.is_absolute()


class TestTypeCorrectness:
    def test_db_port_is_int(self):
        assert isinstance(DB_PORT, int)

    def test_rate_limit_wait_is_int(self):
        assert isinstance(RATE_LIMIT_WAIT_SECONDS, int)

    def test_riksdag_base_url_starts_with_https(self):
        assert RIKSDAG_BASE_URL.startswith("https://")
