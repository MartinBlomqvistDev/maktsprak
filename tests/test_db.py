"""Tests for the db sub-package, all Supabase calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestInsertSpeech:
    def test_returns_data_on_success(self):
        mock_resp = MagicMock()
        mock_resp.data = [{"id": "HD0972_1"}]

        with patch("src.maktsprak_pipeline.db.speeches.supabase_write") as mock_client:
            mock_client.table.return_value.upsert.return_value.execute.return_value = mock_resp

            from src.maktsprak_pipeline.db.speeches import insert_speech

            result = insert_speech({"id": "HD0972_1", "party": "S", "text": "Test"})
            assert result == [{"id": "HD0972_1"}]

    def test_raises_on_none_data(self):
        import pytest

        mock_resp = MagicMock()
        mock_resp.data = None

        with patch("src.maktsprak_pipeline.db.speeches.supabase_write") as mock_client:
            mock_client.table.return_value.upsert.return_value.execute.return_value = mock_resp

            from src.maktsprak_pipeline.db.speeches import insert_speech

            with pytest.raises(RuntimeError):
                insert_speech({"id": "BAD", "party": "X"})


class TestFetchSpeechesCount:
    def test_returns_integer(self):
        mock_resp = MagicMock()
        mock_resp.data = []
        mock_resp.count = 42

        with patch("src.maktsprak_pipeline.db.speeches.supabase") as mock_client:
            mock_client.table.return_value.select.return_value.execute.return_value = mock_resp

            from src.maktsprak_pipeline.db.speeches import fetch_speeches_count

            count = fetch_speeches_count()
            assert count == 42

    def test_returns_zero_when_count_is_none(self):
        mock_resp = MagicMock()
        mock_resp.data = []
        mock_resp.count = None

        with patch("src.maktsprak_pipeline.db.speeches.supabase") as mock_client:
            mock_client.table.return_value.select.return_value.execute.return_value = mock_resp

            from src.maktsprak_pipeline.db.speeches import fetch_speeches_count

            count = fetch_speeches_count()
            assert count == 0


class TestInsertTweet:
    def test_upsert_called_with_on_conflict(self):
        mock_resp = MagicMock()
        mock_resp.data = [{"tweet_id": "123"}]

        with patch("src.maktsprak_pipeline.db.tweets.supabase_write") as mock_client:
            upsert_mock = mock_client.table.return_value.upsert
            upsert_mock.return_value.execute.return_value = mock_resp

            from src.maktsprak_pipeline.db.tweets import insert_tweet

            insert_tweet({"tweet_id": "123", "text": "Test tweet"})

            upsert_mock.assert_called_once()
            call_kwargs = upsert_mock.call_args[1]
            assert call_kwargs.get("on_conflict") == "tweet_id"
