"""Transformation phase — parse Riksdag PDFs into structured speech records."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from tqdm import tqdm

from ..config import RAW_DATA_PATH, VALID_PARTIES
from ..logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Compiled regex for Riksdag speech segmentation
#
# Riksdag protocols use a consistent annotation format:
#     Anf. <number> <Speaker Name> (<PARTY>):
#         <speech text>
#     Anf. <next>...
#
# Capture groups:
#   1 — speaker name
#   2 — party abbreviation (1-2 uppercase letters including Swedish Å/Ä/Ö)
#   3 — speech body (up to next Anf. or end-of-string)
# ---------------------------------------------------------------------------
_SPEECH_RE = re.compile(
    r"Anf\.\s+\d+\s+(.*?)\s+\(([A-ZÅÄÖ]{1,2})\):(.*?)(?=Anf\.|\Z)",
    re.S,
)


def _process_doc(doc: ET.Element) -> list[dict[str, Any]]:
    """Parse a single Riksdag protocol XML element into speech records.

    Downloads the associated PDF if not already cached, extracts full text,
    and splits it into per-speaker speech segments.  Speakers from parties
    outside :data:`~config.VALID_PARTIES` are silently dropped.

    Args:
        doc: An XML ``<dokument>`` element from the Riksdag API response.

    Returns:
        List of speech dicts with keys: ``id``, ``protocol_id``,
        ``protocol_date``, ``speaker``, ``party``, ``text``, ``file_url``.
        Returns an empty list if the document has no PDF URL or if PDF
        extraction fails entirely.
    """
    protocol_id: str | None = doc.findtext("dok_id")
    protocol_date: str | None = doc.findtext("datum")
    file_url: str | None = doc.findtext("filbilaga/fil/url")

    if not file_url:
        return []

    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    pdf_path = RAW_DATA_PATH / f"{protocol_id}.pdf"

    # Download PDF only if not already cached from a previous run.
    if not pdf_path.exists():
        resp_pdf = requests.get(file_url, timeout=60)
        resp_pdf.raise_for_status()
        pdf_path.write_bytes(resp_pdf.content)

    try:
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text(x_tolerance=2) or ""
                # Rejoin hyphenated line-breaks, then collapse whitespace.
                raw = re.sub(r"-\n", "", raw)
                raw = re.sub(r"\s+", " ", raw)
                pages.append(raw)
        full_text = " ".join(pages)
    except Exception as exc:
        logger.warning(f"PDF extraction failed for {protocol_id}: {exc}")
        return []

    # Group consecutive speech segments by (speaker, party) so that a speaker
    # who holds the floor multiple times within one protocol is merged into a
    # single record.
    grouped: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for speaker, party, speech_body in _SPEECH_RE.findall(full_text):
        if party not in VALID_PARTIES:
            continue
        body = speech_body.strip()
        if body:
            grouped[(speaker.strip(), party)].append(body)

    records: list[dict[str, Any]] = []
    for idx, ((speaker, party), segments) in enumerate(grouped.items(), start=1):
        records.append(
            {
                "id": f"{protocol_id}_{idx}",
                "protocol_id": protocol_id,
                "protocol_date": protocol_date,
                "speaker": speaker,
                "party": party,
                "text": "\n\n".join(segments),
                "file_url": file_url,
            }
        )

    return records


def transform_riksdag(xml_file: str | None) -> list[dict[str, Any]]:
    """Convert a Riksdag protocol listing XML file into speech records.

    Iterates over every ``<dokument>`` element, downloads and parses each
    linked PDF, and returns the flat list of extracted speech dicts.

    Args:
        xml_file: Path to the XML file saved by
            :func:`~.extract.extract_riksdag_protocols`.

    Returns:
        All extracted speech dicts across all protocols in the file.
        Returns an empty list if ``xml_file`` is ``None``.
    """
    if not xml_file:
        return []

    root = ET.parse(xml_file).getroot()
    documents = root.findall(".//dokument")
    results: list[dict[str, Any]] = []

    for doc in tqdm(documents, desc="Parsing protocols"):
        results.extend(_process_doc(doc))

    logger.info(f"Transformation complete: {len(results)} speeches from {len(documents)} protocols.")
    return results
