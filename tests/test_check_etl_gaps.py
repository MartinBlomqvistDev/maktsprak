"""Tests for scripts/check_etl_gaps.py, the daily check for protocols the ETL missed."""

from __future__ import annotations

import ast
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_etl_gaps", ROOT / "scripts" / "check_etl_gaps.py"
)
gaps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gaps)


def _speech(dok_id: str, day: str, party: str) -> dict[str, str]:
    """One entry of the Riksdag's speech list, with only the fields the check reads."""
    return {"dok_id": dok_id, "dok_datum": f"{day} 00:00:00", "parti": party}


def test_the_parties_match_the_pipeline():
    # Executes only the assignment, as tests/search/test_parties.py does,
    # because importing the pipeline's package needs Supabase credentials.
    config = ROOT / "src" / "maktsprak_pipeline" / "config.py"
    tree = ast.parse(config.read_text(encoding="utf-8"))
    assignment = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "VALID_PARTIES"
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=assignment, type_ignores=[]), str(config), "exec"), namespace)
    assert namespace["VALID_PARTIES"] == gaps.VALID_PARTIES


class TestCountPartySpeeches:
    def test_counts_party_members_only_from_the_start_date(self):
        items = [
            _speech("HE0913", "2026-11-03", "S"),
            _speech("HE0913", "2026-11-03", "M"),
            _speech("HE0913", "2026-11-03", "TALMANNEN"),
            _speech("HE0911", "2026-10-30", "ANDRE VICE TALMANNEN"),
            _speech("HE0910", "2026-10-21", "V"),
        ]
        counts = gaps.count_party_speeches(items, start=date(2026, 10, 28))
        assert counts == {"HE0913": (date(2026, 11, 3), 2)}


class TestOverdue:
    COUNTS = {"HE0913": (date(2026, 11, 3), 8), "HE0921": (date(2026, 11, 24), 36)}
    PUBLISHED = {"HE0913": date(2026, 11, 3), "HE0921": date(2026, 11, 24)}

    def test_due_once_more_than_eight_days_have_passed(self):
        assert gaps.overdue(self.COUNTS, self.PUBLISHED, today=date(2026, 11, 11)) == []
        assert gaps.overdue(self.COUNTS, self.PUBLISHED, today=date(2026, 11, 12)) == ["HE0913"]

    def test_a_protocol_missing_from_the_listing_is_dated_by_its_speeches(self):
        due = gaps.overdue(self.COUNTS, {}, today=date(2026, 12, 3))
        assert due == ["HE0913", "HE0921"]


class TestPartySpeeches:
    def test_the_batch_doubles_until_it_reaches_the_start(self, monkeypatch):
        recent = [_speech("HE0922", "2026-11-25", "S")]
        older = recent + [_speech("HE0910", "2026-10-21", "V")]
        sizes: list[str] = []

        def fake_get(url, headers=None):
            sizes.append(url.split("sz=")[1].split("&")[0])
            items = recent if len(sizes) == 1 else older
            return json.dumps({"anforandelista": {"anforande": items}}).encode()

        monkeypatch.setattr(gaps, "_get", fake_get)
        counts = gaps.party_speeches(start=date(2026, 10, 28))
        assert sizes == ["2000", "4000"]
        assert counts == {"HE0922": (date(2026, 11, 25), 1)}

    def test_gives_up_when_the_largest_batch_is_still_too_recent(self, monkeypatch):
        recent = json.dumps(
            {"anforandelista": {"anforande": [_speech("HE0922", "2026-11-25", "S")]}}
        )
        monkeypatch.setattr(gaps, "_get", lambda url, headers=None: recent.encode())
        with pytest.raises(RuntimeError, match="do not reach back"):
            gaps.party_speeches(start=date(2026, 10, 28))


class TestMain:
    @pytest.fixture
    def world(self, monkeypatch):
        """Replace every network call; return the set of stored protocols to edit."""
        counts = {"HE0913": (date(2026, 11, 3), 8), "HE0914": (date(2026, 11, 4), 20)}
        have = {"HE0913", "HE0914"}
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.setattr("sys.argv", ["check_etl_gaps.py"])
        monkeypatch.setattr(gaps, "party_speeches", lambda start: counts)
        monkeypatch.setattr(
            gaps, "published_protocols", lambda start, today: {p: d for p, (d, _) in counts.items()}
        )
        monkeypatch.setattr(gaps, "stored", lambda protocols, url, key: have & set(protocols))
        return have

    def test_passes_when_every_due_protocol_is_stored(self, world, capsys):
        assert gaps.main() == 0
        assert "Missing from speeches: 0." in capsys.readouterr().out

    def test_fails_and_names_the_backfill_when_one_is_missing(self, world, capsys):
        world.discard("HE0914")
        assert gaps.main() == 1
        out = capsys.readouterr().out
        assert "| HE0914 | 2026-11-04 | 20 | MISSING |" in out
        assert "::error::speeches lacks 1 protocol(s) with party speeches: HE0914." in out
        assert "backfill_speeches.py 2026-11-04 2026-11-04" in out
