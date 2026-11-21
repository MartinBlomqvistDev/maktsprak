"""Check that the corpus's deep links lead to their speeches.

Draws a sample of speeches that is reproducible from a seed and spread over the
sessions, requests each speech's address, and writes a tracked report of the one
response kept per address: the HTTP status, whether the page names the speech and
its speaker, and whether the opening words of every paragraph of the speech occur in
the page's text in order. Each page is also judged against the speech before and the
speech after in its protocol, each where the corpus holds one, and a page that passes
either of those judgements as well does not pass. The address is computed from the
speech's protocol and number with ``search.models.deep_link``; the ``url`` column of
the corpus is not read. The report records the build report's ``inputs_sha256`` and
number of speeches, so that the report names the corpus the sample was drawn from.
The corpus file is matched to its build report by its number of speeches alone.

Requests go to data.riksdagen.se only, two at a time at the most, at least a second
apart, and never more than the request limit, retries included. An address that
failed in a way that may not recur is requested again, up to ``--attempts`` times
in all. With the defaults, two
speeches per session and one more from each of the two latest sessions, the run makes 50
requests when none is retried. That fits in the limit that ``links.max_requests`` in
``search/config.yaml`` sets, 60 as shipped. No later step reads this script's report, so
the script is unnumbered.

Usage::

    python search/pipeline/check_deep_links.py
    python search/pipeline/check_deep_links.py --save-pages pages/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pyarrow.parquet as pq

from search.cli import add_data_root_argument, configure_logging, settings_from
from search.corpus.build import BUILD_REPORT_NAME, BuildReport
from search.corpus.links import (
    DeepLinkReport,
    HostThrottle,
    PageCheck,
    PageResponse,
    RequestBudget,
    SpeechRef,
    SpeechText,
    check_for,
    draw_sample,
    fetch_page,
    neighbours_of,
    report_of,
)
from search.corpus.store import SPEECHES_NAME, write_report

logger = logging.getLogger("check_deep_links")

REPORT_NAME = "deep_links.json"
MAX_CONCURRENCY = 2
MIN_INTERVAL_SECONDS = 1.0
REF_COLUMNS = ["anf_id", "dok_id", "anf_nummer", "rm", "date", "layout"]
TEXT_COLUMNS = ["anf_id", "anf_nummer", "speaker_raw", "paragraphs"]


def load_speech_refs(path: Path) -> list[SpeechRef]:
    """Read every speech's identity from the corpus, without its text or its url."""
    table = pq.read_table(path, columns=REF_COLUMNS)
    return [SpeechRef.model_validate(row) for row in table.to_pylist()]


def load_speech_texts(path: Path, dok_ids: Sequence[str]) -> dict[str, SpeechText]:
    """Read the number, speaker and paragraphs of every speech in the given protocols."""
    table = pq.read_table(
        path, columns=TEXT_COLUMNS + ["dok_id"], filters=[("dok_id", "in", dok_ids)]
    )
    texts: dict[str, SpeechText] = {}
    for row in table.to_pylist():
        texts[row["anf_id"]] = SpeechText(
            anf_id=row["anf_id"],
            anf_nummer=row["anf_nummer"],
            speaker=row["speaker_raw"],
            paragraphs=tuple(row["paragraphs"]),
        )
    return texts


async def fetch_pages(
    urls: Sequence[str],
    *,
    client: httpx.AsyncClient,
    throttle: HostThrottle,
    budget: RequestBudget,
    concurrency: int,
    attempts: int,
) -> dict[str, PageResponse]:
    """Fetch each distinct address, a few at a time, and return what came back."""
    gate = asyncio.Semaphore(concurrency)

    async def fetch(url: str) -> PageResponse:
        async with gate:
            response = await fetch_page(
                client, url, throttle=throttle, budget=budget, attempts=attempts
            )
        logger.info("%s: %s", url, response.error or f"HTTP {response.status}")
        return response

    pages = {url: asyncio.ensure_future(fetch(url)) for url in dict.fromkeys(urls)}
    return {url: await task for url, task in pages.items()}


def save_pages(
    directory: Path, sample: Sequence[SpeechRef], pages: dict[str, PageResponse]
) -> None:
    """Write each page that came back to *directory*, named after its speech."""
    directory.mkdir(parents=True, exist_ok=True)
    for speech in sample:
        html = pages[speech.url].html
        if html is not None:
            (directory / f"{speech.anf_id}.html").write_text(html, encoding="utf-8", newline="")


async def check_speeches(
    sample: Sequence[SpeechRef],
    texts: dict[str, SpeechText],
    *,
    client: httpx.AsyncClient,
    throttle: HostThrottle,
    budget: RequestBudget,
    concurrency: int,
    attempts: int,
    pages_directory: Path | None = None,
) -> list[PageCheck]:
    """Fetch each sampled speech's page and judge it against the speech and its neighbours."""
    responses = await fetch_pages(
        [speech.url for speech in sample],
        client=client,
        throttle=throttle,
        budget=budget,
        concurrency=concurrency,
        attempts=attempts,
    )
    if pages_directory is not None:
        save_pages(pages_directory, sample, responses)
    return [
        check_for(speech, texts[speech.anf_id], responses[speech.url], neighbours_of(speech, texts))
        for speech in sample
    ]


def load_build_report(path: Path) -> BuildReport:
    """Read the build report that describes the corpus.

    Args:
        path: The build report.

    Returns:
        The report, validated.

    Raises:
        SystemExit: If the report does not exist.
    """
    if not path.is_file():
        raise SystemExit(f"{path.name} does not exist. Run 02_build_anforanden.py first.")
    return BuildReport.model_validate_json(path.read_text(encoding="utf-8"))


async def run(
    args: argparse.Namespace,
    corpus: Path,
    build: BuildReport,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DeepLinkReport:
    """Draw the sample, check it and assemble the report.

    Args:
        args: The parsed command line.
        corpus: The corpus file.
        build: The build report of that corpus.
        transport: What the HTTP client sends its requests through; the
            network when ``None``.

    Returns:
        The report.

    Raises:
        SystemExit: If the corpus does not hold as many speeches as the build
            report says, so that the report would name another corpus.
    """
    speeches = load_speech_refs(corpus)
    if len(speeches) != build.speeches:
        raise SystemExit(
            f"{corpus.name} holds {len(speeches)} speeches and its build report "
            f"{build.speeches}. Run 02_build_anforanden.py again."
        )
    sample = draw_sample(speeches, args.seed, args.per_session, args.extra_recent_sessions)
    logger.info("%d speeches sampled over %d sessions", len(sample), len({s.rm for s in sample}))
    texts = load_speech_texts(corpus, sorted({speech.dok_id for speech in sample}))

    budget = RequestBudget(args.max_requests)
    throttle = HostThrottle(args.min_interval)
    async with httpx.AsyncClient(
        timeout=args.timeout, follow_redirects=False, transport=transport
    ) as client:
        checks = await check_speeches(
            sample,
            texts,
            client=client,
            throttle=throttle,
            budget=budget,
            concurrency=args.concurrency,
            attempts=args.attempts,
            pages_directory=args.save_pages,
        )
    return report_of(
        checks,
        inputs_sha256=build.inputs_sha256,
        corpus_speeches=build.speeches,
        seed=args.seed,
        per_session=args.per_session,
        extra_recent_sessions=args.extra_recent_sessions,
        budget=budget,
    )


def main() -> None:
    """Parse the command line, run the check and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_root_argument(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Decides which speeches are drawn. Defaults to links.seed in config.yaml.",
    )
    parser.add_argument(
        "--per-session", type=int, default=2, help="Speeches to draw from each session."
    )
    parser.add_argument(
        "--extra-recent-sessions",
        type=int,
        default=2,
        help="How many of the latest sessions give one more speech.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help=(
            "The most HTTP requests to make, retries included. "
            "Defaults to links.max_requests in config.yaml."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        choices=range(1, MAX_CONCURRENCY + 1),
        help="Requests in flight at once.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=MIN_INTERVAL_SECONDS,
        help=f"The least number of seconds between two requests, at least {MIN_INTERVAL_SECONDS}.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds before giving up.")
    parser.add_argument("--attempts", type=int, default=3, help="Tries per page.")
    parser.add_argument(
        "--save-pages",
        type=Path,
        default=None,
        help="A directory to write each fetched page to, one file per speech.",
    )
    args = parser.parse_args()
    if args.min_interval < MIN_INTERVAL_SECONDS:
        parser.error(f"--min-interval cannot be below {MIN_INTERVAL_SECONDS} seconds.")

    configure_logging()
    settings = settings_from(args)
    if args.seed is None:
        args.seed = settings.links.seed
    if args.max_requests is None:
        args.max_requests = settings.links.max_requests
    build = load_build_report(settings.tracked.reports / BUILD_REPORT_NAME)
    report = asyncio.run(run(args, settings.data.search / SPEECHES_NAME, build))
    write_report(report, settings.tracked.reports / REPORT_NAME)

    logger.info("%s", report.verdict)
    logger.info("%d of %d requests used", report.requests_made, report.request_limit)
    logger.info("Report written to %s", settings.tracked.reports / REPORT_NAME)


if __name__ == "__main__":
    main()
