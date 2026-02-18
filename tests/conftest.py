"""Shared pytest fixtures for the MaktspråkAI test suite."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sample Riksdag XML
# ---------------------------------------------------------------------------

SAMPLE_PROTOCOL_XML = """\
<dokumentlista traffar="1">
  <dokument>
    <dok_id>HD0972</dok_id>
    <datum>2025-11-20</datum>
    <filbilaga>
      <fil>
        <url>https://example.com/HD0972.pdf</url>
      </fil>
    </filbilaga>
  </dokument>
</dokumentlista>
"""

SAMPLE_SPEECH_TEXT = (
    "Anf. 1 Anna Andersson (S): Det är viktigt att vi värnar om välfärden. "
    "Vi måste satsa på skolan och sjukvården för alla medborgare. "
    "Anf. 2 Erik Eriksson (M): Vi vill sänka skatterna och öka friheten "
    "för företagare och privatpersoner i Sverige. "
    "Anf. 3 Karin Larsson (V): Klimatkrisen kräver omedelbar handling "
    "och en rättvis omställning för alla arbetare."
)


@pytest.fixture
def sample_xml_element() -> ET.Element:
    """Return a single ``<dokument>`` XML element for unit tests."""
    root = ET.fromstring(SAMPLE_PROTOCOL_XML)
    return root.find(".//dokument")


@pytest.fixture
def sample_speech_text() -> str:
    """Return a short multi-speaker Riksdag text for regex tests."""
    return SAMPLE_SPEECH_TEXT


@pytest.fixture
def sample_speeches_df() -> pd.DataFrame:
    """Return a minimal speeches DataFrame for NLP tests."""
    return pd.DataFrame(
        {
            "id": ["HD0972_1", "HD0972_2"],
            "party": ["S", "M"],
            "text": [
                "Vi måste satsa på välfärden och skolan.",
                "Sänk skatterna för att gynna företagande.",
            ],
            "protocol_date": pd.to_datetime(["2025-11-20", "2025-11-20"]),
        }
    )
