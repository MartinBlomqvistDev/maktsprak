"""Transformation phase — parse Riksdag PDFs into structured speech records."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from tqdm import tqdm

from ..config import PARTY_RENAMES, RAW_DATA_PATH, VALID_PARTIES
from ..logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Compiled regex for Riksdag speech segmentation
#
# Riksdag protocols annotate every speech with a header of the form:
#     Anf. <number> <Speaker Name> (<PARTY>):            <- ordinary speech
#     Anf. <number> <Speaker Name> (<PARTY>) replik:     <- reply (~30 % of all)
#         <speech text>
#     Anf. <next>...
#
# The name and body groups are written so they can never cross another
# ``Anf. <digit>`` marker.  Without that guard, a party-less header
# (``Anf. 30 ELSA WIDDING (-):``, ``TALMANNEN:`` or a minister listed without a
# party) makes a naive ``.*?`` run on to the *next* ``(PARTY):`` and swallow the
# following speech, mis-attributing it.  The ``(?:\s+replik)?`` clause captures
# reply headers, whose colon follows ``replik`` rather than the party.
#
# Capture groups:
#   1 — speaker name
#   2 — party abbreviation (1-2 letters incl. Swedish Å/Ä/Ö).  Protocols up to
#       ~2009 lower-case the party — ``(m)``, ``(fp)`` — and switched to
#       upper-case ``(M)`` around 2010; both are accepted and normalised to
#       upper case in :func:`_canonical_party`.
#   3 — speech body (up to the next Anf. marker or end-of-string)
# ---------------------------------------------------------------------------
_SPEECH_RE = re.compile(
    r"Anf\.\s+\d+\s+"
    r"((?:(?!Anf\.\s+\d).)*?)"  # 1: speaker name (never crosses a marker)
    r"\s+\(([A-Za-zÅÄÖåäö]{1,2})\)"  # 2: party (either case; see note above)
    r"(?:\s+replik)?:"  # optional reply marker
    r"((?:(?!Anf\.\s+\d).)*)",  # 3: body (never crosses a marker)
    re.S,
)


def _canonical_party(raw_party: str) -> str | None:
    """Map a raw party abbreviation to its canonical form, or ``None`` if unknown.

    Upper-cases the abbreviation (pre-2010 protocols lower-case it), applies
    historical renames (e.g. ``FP`` -> ``L``, see :data:`~config.PARTY_RENAMES`)
    so one party keeps a single label across time, then validates against
    :data:`~config.VALID_PARTIES`.

    Args:
        raw_party: The party abbreviation as captured from the protocol header,
            in either case (``m``, ``FP`` …).

    Returns:
        The canonical party code, or ``None`` if it is not a valid party.
    """
    party = raw_party.upper()
    party = PARTY_RENAMES.get(party, party)
    return party if party in VALID_PARTIES else None

# ---------------------------------------------------------------------------
# Column-aware PDF text extraction
#
# Riksdag protocols are typeset in two columns whose gutter sits near, but not
# exactly at, the page midline.  ``page.extract_text`` reads straight across the
# full width, interleaving the columns and splicing the running header
# (``Prot. 2024/25:53``, the date, a horizontal rule and the debate title) into
# the middle of sentences.  We instead locate the gutter, extract each column
# separately, and drop repeated page furniture.
# ---------------------------------------------------------------------------
_MONTHS = "januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december"
_PROT_LINE_RE = re.compile(r"^Prot\.\s")
_DATE_LINE_RE = re.compile(rf"^\d{{1,2}}\s+(?:{_MONTHS})$", re.I)
_RULE_LINE_RE = re.compile(r"^[\s¯‾–—_-]+$")
_NUMBER_LINE_RE = re.compile(r"^\d{1,4}$")


def _find_column_gutter(
    page: pdfplumber.page.Page, words: list[dict[str, Any]]
) -> tuple[float, int]:
    """Find the vertical whitespace gutter in the page's central band.

    Scans candidate split positions between 40 % and 60 % of the page width and
    returns the one crossed by the fewest words.

    Args:
        page:  The pdfplumber page.
        words: Result of ``page.extract_words()`` (passed in to avoid recompute).

    Returns:
        Tuple of ``(split_x, straddle_count)`` where ``straddle_count`` is the
        number of words crossing ``split_x``.
    """
    lo, hi = page.width * 0.40, page.width * 0.60
    best_x, best_straddle = lo, len(words) + 1
    x = lo
    while x <= hi:
        straddle = sum(1 for w in words if w["x0"] < x < w["x1"])
        if straddle < best_straddle:
            best_x, best_straddle = x, straddle
        x += 2.0
    return best_x, best_straddle


def _extract_page_columns(page: pdfplumber.page.Page) -> list[list[str]]:
    """Extract text lines from a page, split into columns when two are present.

    Args:
        page: The pdfplumber page.

    Returns:
        One list of raw text lines per column (a single list for single-column
        pages), in natural reading order (left column, then right).
    """
    words = page.extract_words()
    two_column = False
    gutter = page.width / 2
    if words:
        gutter, straddle = _find_column_gutter(page, words)
        left = sum(1 for w in words if w["x1"] <= gutter)
        right = sum(1 for w in words if w["x0"] >= gutter)
        # A genuine two-column page has both sides populated and almost no words
        # crossing the gutter.
        two_column = left > 5 and right > 5 and straddle <= max(2, int(0.01 * len(words)))

    if two_column:
        boxes = [(0, 0, gutter, page.height), (gutter, 0, page.width, page.height)]
    else:
        boxes = [(0, 0, page.width, page.height)]

    return [
        (page.within_bbox(bbox).extract_text(x_tolerance=2) or "").split("\n") for bbox in boxes
    ]


def _detect_repeated_furniture(columns: list[list[str]]) -> set[str]:
    """Identify short lines that repeat across columns (running headers/titles).

    Running headers, dates and debate titles recur on nearly every column of a
    protocol; genuine speech lines do not.  Any short line appearing on at least
    20 % of columns is treated as furniture.

    Args:
        columns: Per-column line lists for the whole document.

    Returns:
        The set of line strings to strip everywhere.
    """
    freq: Counter[str] = Counter()
    for lines in columns:
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped.split()) <= 10:
                freq[stripped] += 1
    threshold = max(3, int(0.20 * max(1, len(columns))))
    return {line for line, count in freq.items() if count >= threshold}


def _is_furniture(line: str, repeated: set[str]) -> bool:
    """Return ``True`` if *line* is page furniture rather than speech text."""
    stripped = line.strip()
    if not stripped or stripped in repeated:
        return True
    return bool(
        _PROT_LINE_RE.match(stripped)
        or _DATE_LINE_RE.match(stripped)
        or _RULE_LINE_RE.match(stripped)
        or _NUMBER_LINE_RE.match(stripped)
    )


def _extract_protocol_text(pdf_path: Path) -> str:
    """Extract clean, de-columnised full text from a protocol PDF.

    Args:
        pdf_path: Path to the cached protocol PDF.

    Returns:
        The full protocol text as a single whitespace-normalised string with
        two-column layout resolved and page furniture removed.
    """
    columns: list[list[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            columns.extend(_extract_page_columns(page))

    repeated = _detect_repeated_furniture(columns)
    kept = [line for lines in columns for line in lines if not _is_furniture(line, repeated)]

    text = "\n".join(kept)
    text = re.sub(r"-\n", "", text)  # rejoin hyphenated line-breaks
    text = re.sub(r"\s+", " ", text)
    # Sweep any furniture that survived on single-column fallback pages, where it
    # is embedded inline rather than on its own line.
    text = re.sub(r"Prot\.\s*\d{4}/\d{2}:\d+", " ", text)
    text = re.sub(r"[¯‾]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _download_pdf(file_url: str, pdf_path: Path, retries: int = 5) -> None:
    """Download a protocol PDF, retrying transient connection failures.

    The Riksdag file host drops connections under rapid sequential requests
    (``ConnectionResetError`` / ``RemoteDisconnected``), especially during a
    large backfill.  Without retries those protocols are lost.  Retry with a
    short growing back-off (2, 4, 6, 8, 10 s) recovers them.

    Args:
        file_url: The protocol PDF URL.
        pdf_path: Where to write the downloaded bytes.
        retries:  Number of attempts before giving up.

    Raises:
        RuntimeError: If every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(file_url, timeout=60)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            return
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"PDF download failed after {retries} attempts: {file_url}") from last_exc


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
        _download_pdf(file_url, pdf_path)

    try:
        full_text = _extract_protocol_text(pdf_path)
    except Exception as exc:
        logger.warning(f"PDF extraction failed for {protocol_id}: {exc}")
        return []

    # Group consecutive speech segments by (speaker, party) so that a speaker
    # who holds the floor multiple times within one protocol is merged into a
    # single record.
    grouped: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for speaker, raw_party, speech_body in _SPEECH_RE.findall(full_text):
        party = _canonical_party(raw_party)
        if party is None:
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

    logger.info(
        f"Transformation complete: {len(results)} speeches from {len(documents)} protocols."
    )
    return results
