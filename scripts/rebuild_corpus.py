"""Rebuild the speech corpus from the cached source documents. Offline.

The corpus is derived data: every protocol Riksdagen published is already on
disk under ``data/raw`` (~3 200 PDFs). This script re-parses them all with the
current parser and writes ``speeches_full.parquet`` from scratch — no Supabase,
no network, no egress. That makes the archive **reproducible from source**
rather than an artefact whose provenance is a chain of past ETL runs.

Why it exists
-------------
The old record id was ``f"{protocol_id}_{idx}"``, an ``enumerate()`` counter
over first-appearance order — a property of the parser run, not of the speech.
When the parser fix changed which speeches were extracted, every later index
shifted, so ``HD098_60`` meant one speech before the fix and a different one
after. Re-ingesting wrote both, ids stopped being unique, and no join, dedup or
upsert could tell the two apart. Ids are now the natural key
(``protocol + speaker-slug + party``), which depends only on the document — but
every row written under the old scheme has to be regenerated, and Supabase only
retains recent years (it is a trimmed ETL landing zone), so the source documents
are the only complete truth.

Metadata (protocol date, source URL) is not in the PDFs. It is recovered from
the cached API responses (``data/raw/*.xml``) and, for older protocols the XML
sweep does not cover, from the existing archive — whose per-protocol metadata is
unaffected by the id bug.

Usage::

    python scripts/rebuild_corpus.py --out data/parquet/speeches_rebuilt.parquet
    python scripts/rebuild_corpus.py --limit 50        # quick smoke run
    python scripts/rebuild_corpus.py --jobs 8          # parallel
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from tqdm import tqdm

from src.maktsprak_pipeline.logger import get_logger
from src.maktsprak_pipeline.pipeline.transform import (
    _extract_protocol_text,
    records_from_text,
)

logger = get_logger()

RAW = Path("data/raw")
ARCHIVE = Path("data/parquet/speeches_full.parquet")


def load_metadata(archive: Path = ARCHIVE) -> dict[str, tuple[str, str]]:
    """``{protocol_id_lower: (date, file_url)}`` from every available source.

    The PDFs carry no reliable machine-readable date, so it is recovered from
    the cached API responses first and the existing archive second. The
    archive's *metadata* is trustworthy even though its rows are not: the id bug
    corrupted which speech a row referred to, never the protocol's own date.
    """
    meta: dict[str, tuple[str, str]] = {}

    for path in sorted(RAW.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            logger.warning(f"Unparseable XML {path.name}: {exc}")
            continue
        for doc in root.findall(".//dokument"):
            doc_id = doc.findtext("dok_id")
            date = doc.findtext("datum")
            if doc_id and date:
                url = doc.findtext("dokument_url_html") or doc.findtext("filbilaga") or ""
                meta[doc_id.lower()] = (date[:10], url)
    logger.info(f"Metadata from {len(list(RAW.glob('*.xml')))} XML file(s): {len(meta)} protocols.")

    if archive.exists():
        cols = ["protocol_id", "protocol_date", "file_url"]
        archived = pd.read_parquet(archive, columns=cols).drop_duplicates("protocol_id")
        before = len(meta)
        for row in archived.itertuples(index=False):
            meta.setdefault(
                row.protocol_id.lower(),
                (str(row.protocol_date)[:10], row.file_url or ""),
            )
        logger.info(f"Archive metadata added {len(meta) - before} protocols not in the XMLs.")

    return meta


def parse_one(args: tuple[Path, str, str]) -> list[dict]:
    """Parse one cached PDF into records. Module-level for ProcessPoolExecutor."""
    pdf_path, date, url = args
    protocol_id = pdf_path.stem
    try:
        text = _extract_protocol_text(pdf_path)
    except Exception as exc:  # a single unreadable PDF must not kill the run
        logger.warning(f"Extraction failed for {protocol_id}: {exc}")
        return []
    return records_from_text(text, protocol_id, date, url)


def rebuild(out: Path, limit: int | None, jobs: int) -> pd.DataFrame:
    """Re-parse every cached protocol with metadata and write *out*."""
    meta = load_metadata()
    pdfs = sorted(RAW.glob("*.pdf"))

    tasks: list[tuple[Path, str, str]] = []
    skipped: list[str] = []
    for pdf in pdfs:
        entry = meta.get(pdf.stem.lower())
        if entry is None:
            skipped.append(pdf.stem)
            continue
        tasks.append((pdf, entry[0], entry[1]))

    if skipped:
        logger.warning(
            f"{len(skipped)} PDF(s) have no metadata and are skipped "
            f"(likely non-debate documents): {skipped[:5]}..."
        )
    if limit:
        tasks = tasks[:limit]
    logger.info(f"Parsing {len(tasks)} protocols with {jobs} worker(s)...")

    records: list[dict] = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(parse_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="protocols"):
                records.extend(future.result())
    else:
        for task in tqdm(tasks, desc="protocols"):
            records.extend(parse_one(task))

    if not records:
        raise RuntimeError("No records parsed — refusing to write an empty archive.")

    df = pd.DataFrame(records)
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    df = df.dropna(subset=["protocol_date"])

    # The id is the natural key. If it is ever not unique the key is wrong, and
    # writing anyway would rebuild the exact bug this script exists to remove.
    duplicated = df["id"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "id"].head(5).tolist()
        raise RuntimeError(
            f"{int(duplicated.sum())} duplicate ids after rebuild — the natural key is not "
            f"unique, which is the bug this rebuild removes. Examples: {examples}"
        )

    df = df.sort_values(["protocol_date", "id"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd", index=False)

    size_mb = out.stat().st_size / 1_000_000
    logger.info(
        f"Wrote {len(df):,} speeches from {df['protocol_id'].nunique():,} protocols "
        f"to {out} ({size_mb:.0f} MB), "
        f"{df['protocol_date'].min():%Y-%m-%d}..{df['protocol_date'].max():%Y-%m-%d}"
    )
    logger.info(f"Rows per party: {df['party'].value_counts().to_dict()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the corpus from data/raw. Offline.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/parquet/speeches_rebuilt.parquet"),
        help="Destination Parquet (deliberately NOT the live archive by default).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only N protocols (smoke run).")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes.")
    args = parser.parse_args()

    df = rebuild(args.out, args.limit, args.jobs)
    print(f"REBUILD_DONE rows={len(df)} protocols={df['protocol_id'].nunique()}")


if __name__ == "__main__":
    main()
