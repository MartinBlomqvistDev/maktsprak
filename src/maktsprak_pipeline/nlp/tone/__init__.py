"""Tone/rhetoric analytics: how a party speaks, not what it speaks about.

Every dimension plugs into the shared kernel (:mod:`.kernel`) — the statistics,
the suppression rule and the receipt sampling are written once and reused, so a
new dimension is one ``measure_fn`` plus one :class:`~.kernel.DimensionSpec`.

Dimension modules import and register themselves here; the registry
(:data:`~.kernel.TONE_DIMENSIONS`) is what ``build_site_data.py`` and the
validation script iterate.
"""

from __future__ import annotations

from .kernel import (
    TONE_DIMENSIONS,
    CellStats,
    DimensionSpec,
    Marker,
    ReceiptKind,
    aggregate_cells,
    compile_pattern,
    context_for_span,
    fightin_z,
    find_spans,
    load_pattern_table,
    raw_rate,
    register,
    sample_receipts,
    sentence_spans,
    sentence_split,
    smoothed_rate,
    suppress_thin_cells,
)

__all__ = [
    "TONE_DIMENSIONS",
    "CellStats",
    "DimensionSpec",
    "Marker",
    "ReceiptKind",
    "aggregate_cells",
    "compile_pattern",
    "context_for_span",
    "fightin_z",
    "find_spans",
    "load_pattern_table",
    "raw_rate",
    "register",
    "sample_receipts",
    "sentence_spans",
    "sentence_split",
    "smoothed_rate",
    "suppress_thin_cells",
]
