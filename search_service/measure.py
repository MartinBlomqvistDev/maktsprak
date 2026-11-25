"""Measure the search service's response times from outside, the clock started at the question.

Each question is sent as the page sends it, and its clock starts just before
the request is made. The first line of the answer is the list of results,
which the page shows at once; the last line is the answer step's, once the
selector has chosen. Both times are recorded for every question, with what the
service says about how the question was searched and answered. The first
request is reported apart, since it may meet a cold start.

The questions are the natural questions of the development split, in the
order of their file; the test split is never sent.

Usage::

    python -m search_service.measure --target proxy --url https://HOST/api/fraga \\
        --mode answer --count 8 --questions data/reports/search/questions/natural.jsonl \\
        --out data/reports/search/service_latency_preview.json

With ``--target service`` the URL is the service's own, and the secret is read
from ``SEARCH_SHARED_SECRET``. A protected preview of the site is passed with
the bypass in ``VERCEL_AUTOMATION_BYPASS_SECRET``. Neither is written anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

SECRET_ENV = "SEARCH_SHARED_SECRET"
BYPASS_ENV = "VERCEL_AUTOMATION_BYPASS_SECRET"
#: The longest a request may take: a cold start, the wait for the vector and the selector.
TIMEOUT_S = 300.0


def dev_questions(path: Path, count: int) -> list[tuple[str, str]]:
    """The first *count* questions of the development split in a question set."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    chosen = [(row["question_id"], row["text"]) for row in rows if row.get("split") == "dev"]
    if len(chosen) < count:
        raise SystemExit(f"{path} holds {len(chosen)} questions of the development split.")
    return chosen[:count]


def ask(
    client: httpx.Client, target: str, url: str, mode: str, text: str, headers: dict[str, str]
) -> dict[str, Any]:
    """Send one question and time its lines."""
    if target == "proxy":
        endpoint, body = url, {"question": text, "filters": {}, "mode": mode}
    else:
        endpoint, body = f"{url.rstrip('/')}/{mode}", {"question": text, "filters": {}}
    started = time.perf_counter()
    first: float | None = None
    lines: list[dict[str, Any]] = []
    with client.stream("POST", endpoint, json=body, headers=headers) as response:
        status = response.status_code
        for line in response.iter_lines():
            if not line:
                continue
            if first is None:
                first = time.perf_counter() - started
            lines.append(json.loads(line))
    record: dict[str, Any] = {
        "status": status,
        "first_line_s": round(first, 3) if first is not None else None,
        "last_line_s": round(time.perf_counter() - started, 3),
        "lines": [line.get("type") for line in lines],
    }
    for line in lines:
        if line.get("type") == "search":
            record |= {
                "retrieval": line["retrieval"],
                "vector_wait_ms": line["vector_wait_ms"],
                "hits": len(line["hits"]),
            }
        elif line.get("type") == "answer":
            record |= {
                "selector": line["selector"],
                "fallback": line["fallback"],
                "quotes": len(line["quotes"]),
            }
        elif "error" in line:
            record["error"] = line["error"]
    return record


def quantile(values: list[float], share: float) -> float | None:
    """The nearest-rank quantile, or ``None`` without values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]


def summarize(records: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    """The median, the 95th percentile and the slowest of one time, over the answered requests."""
    values = [r[field] for r in records if r["status"] == 200 and r[field] is not None]
    return {
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
        "requests": len(values),
    }


def main(argv: list[str] | None = None) -> int:
    """Measure and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=("proxy", "service"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=("search", "answer"), required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--caller", default="measure", help="The caller the service is told of.")
    args = parser.parse_args(argv)

    headers: dict[str, str] = {}
    if args.target == "service":
        secret = os.environ.get(SECRET_ENV)
        if not secret:
            raise SystemExit(f"{SECRET_ENV} must be set to ask the service itself.")
        headers = {"x-search-key": secret, "x-caller": args.caller}
    elif os.environ.get(BYPASS_ENV):
        headers = {"x-vercel-protection-bypass": os.environ[BYPASS_ENV]}

    questions = dev_questions(args.questions, args.count)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    records = []
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for question_id, text in questions:
            record = ask(client, args.target, args.url, args.mode, text, headers)
            records.append({"question_id": question_id, **record})
            print(
                f"{question_id}: {record['status']} first {record['first_line_s']} s, "
                f"last {record['last_line_s']} s, {record.get('retrieval')}"
            )
    warm = records[1:]
    report = {
        "target": args.target,
        "host": urlparse(args.url).hostname,
        "mode": args.mode,
        "questions": args.questions.name,
        "started": started,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "first_request": records[0] if records else None,
        "warm": {
            "first_line_s": summarize(warm, "first_line_s"),
            "last_line_s": summarize(warm, "last_line_s"),
        },
        "bm25_alone": sum(1 for r in records if r.get("retrieval") == "bm25"),
        "requests": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
