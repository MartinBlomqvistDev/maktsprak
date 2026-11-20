"""Tests for search/parties.py."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from search.parties import PARTY_ORDER, PARTY_RENAMES, VALID_PARTIES, canonical_party

PIPELINE_CONFIG = Path(__file__).resolve().parents[2] / "src" / "maktsprak_pipeline" / "config.py"
SHARED_NAMES = ("VALID_PARTIES", "PARTY_RENAMES", "PARTY_ORDER")


def _assigned_name(node: ast.stmt) -> str | None:
    """Return the name a top-level statement assigns to, if it assigns to exactly one."""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _pipeline_constants() -> dict[str, object]:
    """Evaluate the pipeline's party constants without importing its config.

    Importing the module would load ``.env`` into the test process, and
    importing its package requires Supabase credentials. Only the assignments
    to the shared names are executed.
    """
    tree = ast.parse(PIPELINE_CONFIG.read_text(encoding="utf-8"))
    assignments = [node for node in tree.body if _assigned_name(node) in SHARED_NAMES]
    namespace: dict[str, object] = {}
    module = ast.Module(body=assignments, type_ignores=[])
    exec(compile(module, str(PIPELINE_CONFIG), "exec"), namespace)
    return {name: namespace[name] for name in SHARED_NAMES}


class TestParityWithPipeline:
    def test_constants_match_the_pipeline_config(self):
        pipeline = _pipeline_constants()
        assert pipeline["VALID_PARTIES"] == VALID_PARTIES
        assert pipeline["PARTY_RENAMES"] == dict(PARTY_RENAMES)
        assert pipeline["PARTY_ORDER"] == list(PARTY_ORDER)

    def test_display_order_covers_every_party_once(self):
        assert len(PARTY_ORDER) == len(set(PARTY_ORDER))
        assert set(PARTY_ORDER) == VALID_PARTIES

    def test_renames_cannot_be_mutated(self):
        with pytest.raises(TypeError):
            PARTY_RENAMES["NYD"] = "SD"  # type: ignore[index]


class TestCanonicalParty:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("S", "S"), ("m", "M"), ("kd", "KD"), ("FP", "L"), ("fp", "L"), (" SD ", "SD")],
    )
    def test_maps_published_abbreviations(self, raw, expected):
        assert canonical_party(raw) == expected

    @pytest.mark.parametrize("raw", ["", "-", "NYD", "TALMANNEN", "Socialdemokraterna"])
    def test_rejects_everything_outside_the_eight(self, raw):
        assert canonical_party(raw) is None
