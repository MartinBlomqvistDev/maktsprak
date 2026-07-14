"""Tone/rhetoric analytics: how a party speaks, not what it speaks about.

Every dimension plugs into the shared kernel (:mod:`.kernel`) — the statistics,
the suppression rule and the receipt sampling are written once and reused, so a
new dimension is one ``measure_fn`` plus one :class:`~.kernel.DimensionSpec`.

Dimension modules import and register themselves here; the registry
(:data:`~.kernel.TONE_DIMENSIONS`) is what ``build_site_data.py`` and the
validation script iterate.
"""

from __future__ import annotations

# Importing the dimension modules is what populates TONE_DIMENSIONS — the
# registry is the single wiring point that build_site_data.py and the validator
# iterate, so a dimension that is not imported here simply does not exist.
from . import inclusive, readability, vi_dom
from .kernel import (
    SENT_COL,
    TONE_DIMENSIONS,
    CellStats,
    DimensionSpec,
    Marker,
    ReceiptKind,
    aggregate_cells,
    compile_alternation,
    compile_pattern,
    context_for_span,
    ensure_sentence_spans,
    fightin_z,
    find_spans,
    load_pattern_table,
    raw_rate,
    register,
    sample_receipts,
    sentence_index,
    sentence_spans,
    sentence_split,
    sentences_with_a_hit,
    smoothed_rate,
    suppress_thin_cells,
)

__all__ = [
    "SENT_COL",
    "TONE_DIMENSIONS",
    "CellStats",
    "DimensionSpec",
    "Marker",
    "ReceiptKind",
    "aggregate_cells",
    "compile_alternation",
    "compile_pattern",
    "context_for_span",
    "ensure_sentence_spans",
    "fightin_z",
    "find_spans",
    "inclusive",
    "load_pattern_table",
    "raw_rate",
    "readability",
    "register",
    "sample_receipts",
    "sentence_index",
    "sentence_spans",
    "sentence_split",
    "sentences_with_a_hit",
    "smoothed_rate",
    "suppress_thin_cells",
    "vi_dom",
]
