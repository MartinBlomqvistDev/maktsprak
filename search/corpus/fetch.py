"""Fetching the Riksdag's speech datasets, one zip per parliamentary session.

The datasets are listed at
https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anforanden/
and served from ``data.riksdagen.se``. Every file that is fetched goes into a
tracked manifest with its size, its SHA-256 and the date it was fetched, so a
corpus can be traced to the exact bytes it was built from.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20
FIRST_SUPPORTED_YEAR = 2000


class FetchedFile(BaseModel):
    """One dataset file as it was fetched.

    Attributes:
        session: The parliamentary session, as the Riksdag writes it: ``2002/03``.
        file: The file name under the raw-data directory.
        url: Where the file was fetched from.
        bytes: The size of the file.
        sha256: The SHA-256 of the file, in hex.
        fetched: The date the file was fetched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: str
    file: str
    url: str
    bytes: int
    sha256: str
    fetched: date


class Manifest(BaseModel):
    """Every dataset file the corpus is built from, ordered by session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[FetchedFile, ...] = ()

    def by_session(self) -> dict[str, FetchedFile]:
        """Return the entries keyed by session."""
        return {entry.session: entry for entry in self.files}


def session_label(start_year: int) -> str:
    """Name the session that opens in the autumn of *start_year*.

    Args:
        start_year: The calendar year the session opens.

    Returns:
        The label the Riksdag uses, for example ``2002/03``.

    Raises:
        ValueError: For years before 2000, where the labels follow another form.
    """
    if start_year < FIRST_SUPPORTED_YEAR:
        raise ValueError(f"Sessions before {FIRST_SUPPORTED_YEAR}/01 are not supported.")
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def _start_year(session: str) -> int:
    """Return the year a session opens, or raise if the label is malformed."""
    start, _, end = session.partition("/")
    well_formed = len(start) == 4 and start.isdigit() and len(end) == 2 and end.isdigit()
    if not well_formed or session != session_label(int(start)):
        raise ValueError(f"Not a session label: {session!r}. Expected the form '2002/03'.")
    return int(start)


def sessions_between(first: str, last: str) -> list[str]:
    """List the sessions from *first* to *last*, both included.

    Args:
        first: The earliest session, for example ``2002/03``.
        last: The latest session.

    Returns:
        The session labels in chronological order.

    Raises:
        ValueError: If a label is malformed or *first* is later than *last*.
    """
    first_year, last_year = _start_year(first), _start_year(last)
    if first_year > last_year:
        raise ValueError(f"The first session {first} is later than the last session {last}.")
    return [session_label(year) for year in range(first_year, last_year + 1)]


def dataset_url(url_template: str, session: str) -> str:
    """Fill in the dataset address for one session.

    Args:
        url_template: The address with a ``{session}`` placeholder.
        session: The session label, for example ``2002/03``.

    Returns:
        The address of that session's dataset. The Riksdag names the file after
        the label without its slash: ``anforande-200203.json.zip``.
    """
    _start_year(session)
    return url_template.format(session=session.replace("/", ""))


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, in hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> Manifest:
    """Read the manifest, or return an empty one if it has not been written yet."""
    if not path.exists():
        return Manifest()
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Write the manifest as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def verify_raw_files(manifest: Manifest, raw_dir: Path) -> None:
    """Check that every file in the manifest is on disk with the recorded hash.

    Args:
        manifest: The manifest to check against.
        raw_dir: The directory the files were fetched to.

    Raises:
        FileNotFoundError: If a file is missing.
        ValueError: If a file's SHA-256 differs from the manifest.
    """
    for entry in manifest.files:
        path = raw_dir / entry.file
        if not path.exists():
            raise FileNotFoundError(f"{path} is in the manifest but not on disk.")
        if sha256_of(path) != entry.sha256:
            raise ValueError(f"{path} does not match the SHA-256 in the manifest.")


async def _download(client: httpx.AsyncClient, url: str, destination: Path) -> tuple[int, str]:
    """Stream *url* to *destination* and return its size and SHA-256."""
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            async for chunk in response.aiter_bytes(CHUNK_BYTES):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    partial.replace(destination)
    return size, digest.hexdigest()


def _is_transient(error: httpx.HTTPError) -> bool:
    """Tell a failure worth retrying from one that will not go away."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return (
            status == httpx.codes.TOO_MANY_REQUESTS or status >= httpx.codes.INTERNAL_SERVER_ERROR
        )
    return isinstance(error, httpx.TransportError)


async def _download_with_retries(
    client: httpx.AsyncClient,
    url: str,
    destination: Path,
    attempts: int,
    backoff_seconds: float,
) -> tuple[int, str]:
    """Download a file, retrying dropped connections and server errors.

    The Riksdag's file host drops connections under rapid requests, so a short
    growing pause between attempts recovers most failures.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await _download(client, url, destination)
        except httpx.HTTPError as error:
            if not _is_transient(error) or attempt == attempts:
                raise
            logger.warning("Attempt %d of %d failed for %s: %r", attempt, attempts, url, error)
            await asyncio.sleep(backoff_seconds * attempt)
    raise AssertionError("unreachable: the loop returns or raises")


async def fetch_datasets(
    sessions: Sequence[str],
    *,
    url_template: str,
    raw_dir: Path,
    known: Manifest,
    client: httpx.AsyncClient,
    force: bool = False,
    today: date | None = None,
    concurrency: int = 2,
    attempts: int = 5,
    backoff_seconds: float = 2.0,
) -> Manifest:
    """Fetch the datasets for *sessions* and return the updated manifest.

    A file that is on disk and matches its manifest entry is left alone, so a
    second run fetches nothing. A file on disk that the manifest does not vouch
    for stops the run: the corpus must never be built from bytes of unknown
    origin.

    Args:
        sessions: The session labels to fetch.
        url_template: The dataset address with a ``{session}`` placeholder.
        raw_dir: The directory to fetch to.
        known: The manifest as it stands before this run.
        client: The HTTP client to use.
        force: Fetch again even when the file is already on disk.
        today: The date to record, defaulting to the current date.
        concurrency: How many files to fetch at once.
        attempts: How many times to try each file.
        backoff_seconds: The pause after the first failed attempt; it grows
            linearly with each further attempt.

    Returns:
        The manifest with one entry per session, old and new, ordered by session.

    Raises:
        ValueError: If a file is on disk but not vouched for by the manifest.
        httpx.HTTPError: If a file cannot be fetched.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched_on = today or date.today()
    entries = known.by_session()
    gate = asyncio.Semaphore(concurrency)

    async def fetch_one(session: str) -> FetchedFile:
        url = dataset_url(url_template, session)
        destination = raw_dir / url.rsplit("/", 1)[-1]
        entry = entries.get(session)
        if destination.exists() and not force:
            if entry is None or sha256_of(destination) != entry.sha256:
                raise ValueError(
                    f"{destination} is on disk but does not match the manifest. "
                    f"Fetch it again with --force."
                )
            return entry
        async with gate:
            logger.info("Fetching %s", url)
            size, digest = await _download_with_retries(
                client, url, destination, attempts, backoff_seconds
            )
        return FetchedFile(
            session=session,
            file=destination.name,
            url=url,
            bytes=size,
            sha256=digest,
            fetched=fetched_on,
        )

    fetched = await asyncio.gather(*(fetch_one(session) for session in sessions))
    entries.update({entry.session: entry for entry in fetched})
    return Manifest(files=tuple(entries[session] for session in sorted(entries)))
