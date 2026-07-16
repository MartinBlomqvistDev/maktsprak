"""Verify a rebuilt corpus against the old archive and the source PDFs.

Written *before* the rebuild finished, deliberately: a check designed after
seeing the numbers is a check designed to pass. Every assertion here is one the
rebuild could genuinely fail.

Three independent lines of evidence, because agreement between them is what
makes the result trustworthy — any one alone could be fooled:

1. **Internal** — is the archive self-consistent? Unique ids, no empties, plausible
   parties per era (SD cannot appear before 2010; FP must be folded into L).
2. **Against the old archive** — did content survive? The old rows are keyed by a
   broken id, so ids cannot be compared. Text can: the same speaker in the same
   protocol must still be there, saying the same thing.
3. **Against the source PDFs** — the ground truth. Re-extract a random protocol
   and confirm the archive's text is really in the document.

Usage::

    python scripts/verify_rebuild.py --new data/parquet/speeches_rebuilt.parquet
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.maktsprak_pipeline.pipeline.transform import _extract_protocol_text, records_from_text

OK = "  [ok]  "
FAIL = "  [FAIL]"

#: SD first entered the Riksdag in 2010; a speech attributed to them before
#: that is a parse or metadata error, not a fact.
SD_FIRST_YEAR = 2010


class Report:
    """Collects pass/fail so every check runs, rather than dying on the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"{OK if ok else FAIL} {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(label)
        return ok


def iteration_1_internal(new: pd.DataFrame, report: Report) -> None:
    """Is the archive self-consistent?"""
    print("\n" + "=" * 78 + "\nITERATION 1 — internal consistency\n" + "=" * 78)
    new["year"] = new["protocol_date"].dt.year

    report.check(
        new["id"].nunique() == len(new),
        "ids are unique",
        f"{new['id'].nunique():,} ids / {len(new):,} rows",
    )
    report.check(not new["text"].isna().any(), "no null texts")
    report.check((new["text"].str.strip() != "").all(), "no empty texts")
    report.check(not new["party"].isna().any(), "no null parties")
    report.check(
        new["id"].str.match(r"^[A-Za-z0-9]+_[a-z0-9-]+_[a-z]+$").all(),
        "ids match the natural-key shape",
        new["id"].iloc[0],
    )

    sd_early = new[(new["party"] == "SD") & (new["year"] < SD_FIRST_YEAR)]
    report.check(
        len(sd_early) == 0,
        f"no SD speeches before {SD_FIRST_YEAR}",
        f"{len(sd_early)} found" if len(sd_early) else "SD enters 2010, as it should",
    )
    report.check(
        "FP" not in set(new["party"]),
        "Folkpartiet folded into L (no raw FP)",
    )
    report.check(
        set(new["party"]) <= {"S", "M", "SD", "C", "V", "KD", "MP", "L"},
        "only the eight Riksdag parties",
        str(sorted(set(new["party"]))),
    )

    per_year = new.groupby("year").size()
    report.check(
        per_year.min() > 300,
        "every year has a plausible speech count",
        f"min {per_year.min()} ({per_year.idxmin()}), max {per_year.max()} ({per_year.idxmax()})",
    )


def iteration_2_vs_old(new: pd.DataFrame, old: pd.DataFrame, report: Report) -> None:
    """Did content survive the rebuild?

    Ids are not comparable (the old ones are the bug). Content is: for a given
    protocol, the same (speaker-ish, party) should be saying the same words.
    """
    print("\n" + "=" * 78 + "\nITERATION 2 — content vs the old archive\n" + "=" * 78)

    old_p, new_p = set(old["protocol_id"]), set(new["protocol_id"])
    report.check(
        old_p <= new_p,
        "every old protocol is still present",
        f"old {len(old_p):,} / new {len(new_p):,} (+{len(new_p - old_p)} new)",
    )

    print(f"\n  rows: old {len(old):,} -> new {len(new):,} ({len(new) - len(old):+,})")
    print(f"  speakers: old {old['speaker'].nunique():,} -> new {new['speaker'].nunique():,}")

    # Party mix must be broadly stable; a big shift means mis-attribution.
    old_mix = (old["party"].value_counts(normalize=True) * 100).round(1)
    new_mix = (new["party"].value_counts(normalize=True) * 100).round(1)
    mix = pd.DataFrame({"old_%": old_mix, "new_%": new_mix})
    mix["delta"] = (mix["new_%"] - mix["old_%"]).round(1)
    print("\n  party mix (%):")
    print("    " + mix.to_string().replace("\n", "\n    "))
    report.check(
        mix["delta"].abs().max() < 3.0,
        "party mix is stable",
        f"largest shift {mix['delta'].abs().max():.1f}pp",
    )

    # The real content test: is each old speech's text still somewhere in the
    # same protocol? Sampled, since comparing every pair is O(n*m).
    rng = random.Random(42)
    sample_ids = rng.sample(sorted(old_p), min(60, len(old_p)))
    checked = missing = 0
    for pid in sample_ids:
        new_text = " ".join(new.loc[new["protocol_id"] == pid, "text"])
        for text in old.loc[old["protocol_id"] == pid, "text"]:
            probe = " ".join(str(text).split())[:80]
            if len(probe) < 40:
                continue
            checked += 1
            if probe not in " ".join(new_text.split()):
                missing += 1
    report.check(
        missing == 0,
        "old speech text is still present in the rebuild",
        f"{checked - missing}/{checked} probes found across {len(sample_ids)} protocols",
    )


def iteration_3_vs_source(new: pd.DataFrame, report: Report) -> None:
    """The ground truth: is the archive's text actually in the PDF?"""
    print("\n" + "=" * 78 + "\nITERATION 3 — against the source documents\n" + "=" * 78)

    rng = random.Random(7)
    protocols = rng.sample(sorted(set(new["protocol_id"])), 5)
    all_ok = True
    for pid in protocols:
        pdf = Path("data/raw") / f"{pid}.pdf"
        if not pdf.exists():
            report.check(False, f"{pid}: source PDF present")
            all_ok = False
            continue
        rows = new[new["protocol_id"] == pid]
        fresh = records_from_text(_extract_protocol_text(pdf), pid, "2000-01-01", "")
        same_ids = {r["id"] for r in fresh} == set(rows["id"])
        same_text = sorted(r["text"] for r in fresh) == sorted(rows["text"])
        ok = same_ids and same_text
        all_ok &= ok
        print(
            f"{OK if ok else FAIL} {pid}: {len(rows)} rows re-parse identically"
            f"{'' if ok else '  ids_match=' + str(same_ids) + ' text_match=' + str(same_text)}"
        )
    report.check(all_ok, "re-parsing the source reproduces the archive byte-for-byte")

    # Determinism: the same PDF twice must give the same records.
    pid = protocols[0]
    pdf = Path("data/raw") / f"{pid}.pdf"
    a = records_from_text(_extract_protocol_text(pdf), pid, "2000-01-01", "")
    b = records_from_text(_extract_protocol_text(pdf), pid, "2000-01-01", "")
    report.check(a == b, "parsing is deterministic across runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a rebuilt corpus.")
    parser.add_argument("--new", type=Path, default=Path("data/parquet/speeches_rebuilt.parquet"))
    parser.add_argument("--old", type=Path, default=Path("data/parquet/speeches_full.parquet"))
    args = parser.parse_args()

    new = pd.read_parquet(args.new)
    new["protocol_date"] = pd.to_datetime(new["protocol_date"])
    old = pd.read_parquet(args.old)
    old["protocol_date"] = pd.to_datetime(old["protocol_date"])

    print(f"new: {len(new):,} rows  {new['protocol_date'].min():%Y-%m-%d}..{new['protocol_date'].max():%Y-%m-%d}")
    print(f"old: {len(old):,} rows  {old['protocol_date'].min():%Y-%m-%d}..{old['protocol_date'].max():%Y-%m-%d}")

    report = Report()
    iteration_1_internal(new, report)
    iteration_2_vs_old(new, old, report)
    iteration_3_vs_source(new, report)

    print("\n" + "=" * 78)
    if report.failures:
        print(f"VERDICT: {len(report.failures)} CHECK(S) FAILED — do not swap in")
        for failure in report.failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("VERDICT: all checks passed")


if __name__ == "__main__":
    main()
