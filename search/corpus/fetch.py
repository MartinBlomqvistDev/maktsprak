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
import zipfile
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

from search.models import session_label, session_start_year

logger = logging.getLogger(__name__)

MANIFEST_NAME = "anforanden_manifest.json"
CHUNK_BYTES = 1 << 20


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

    @model_validator(mode="after")
    def _check_invariants(self) -> FetchedFile:
        session_start_year(self.session)
        if self.file in ("", ".", "..") or any(mark in self.file for mark in ("/", "\\", ":")):
            raise ValueError(f"{self.file!r} is not a bare file name.")
        if self.file != self.url.rsplit("/", 1)[-1]:
            raise ValueError(f"{self.file!r} is not the file at {self.url}.")
        if self.session.replace("/", "") not in self.file:
            raise ValueError(f"{self.file!r} is not named after session {self.session}.")
        return self


class Manifest(BaseModel):
    """Every dataset file the corpus is built from, ordered by session.

    A session and a file name each appear once, and the entries come in the
    order of their sessions, so a corpus is built from each file exactly once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[FetchedFile, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Manifest:
        sessions = [entry.session for entry in self.files]
        repeated_sessions = sorted({name for name in sessions if sessions.count(name) > 1})
        if repeated_sessions:
            raise ValueError(
                f"The manifest lists a session more than once: {', '.join(repeated_sessions)}."
            )
        files = [entry.file for entry in self.files]
        repeated_files = sorted({name for name in files if files.count(name) > 1})
        if repeated_files:
            raise ValueError(
                f"The manifest lists a file more than once: {', '.join(repeated_files)}."
            )
        if sessions != sorted(sessions):
            raise ValueError("The manifest entries must be ordered by session.")
        return self

    def by_session(self) -> dict[str, FetchedFile]:
        """Return the entries keyed by session."""
        return {entry.session: entry for entry in self.files}


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
    first_year, last_year = session_start_year(first), session_start_year(last)
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
    session_start_year(session)
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
    """Write the manifest as indented JSON with Unix line endings.

    The manifest vouches for every dataset file, so it is written beside its
    destination and moved into place. A write that raises leaves the previous
    manifest untouched and no partial one behind. A process killed during the
    write can leave the ``.part`` file beside the manifest.

    Args:
        manifest: The manifest to write.
        path: Where to write it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        partial.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


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


def _require_zip(path: Path, url: str) -> None:
    """Raise if *path* is not a zip whose member list can be read.

    A response with status 200 can carry a body that is not the dataset, such
    as an HTML page, and a download that stops short leaves a zip without its
    member list. The check refuses both.

    Args:
        path: The file that was fetched.
        url: Where it was fetched from, for the error message.

    Raises:
        ValueError: If the file cannot be read as a zip.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            archive.namelist()
    except zipfile.BadZipFile as error:
        raise ValueError(f"The body fetched from {url} is not a readable zip: {error}") from error


async def _download(client: httpx.AsyncClient, url: str, destination: Path) -> tuple[int, str]:
    """Stream *url* to *destination* and return its size and SHA-256.

    The bytes go to a partial file first and move into place only once they read
    as a zip. Whatever stops the download, the partial file is removed and the
    file already at *destination*, if any, is left as it was.
    """
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                async for chunk in response.aiter_bytes(CHUNK_BYTES):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        _require_zip(partial, url)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
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
    """Download a file, retrying transport errors, rate limits and server errors.

    A dropped connection, a timeout, a 429 or a 5xx answer may not recur, so the
    download is tried again after a pause that grows with each attempt. Any
    other failure is raised at once.

    Args:
        client: The HTTP client to use.
        url: The address of the file.
        destination: Where the file is moved once it reads as a zip.
        attempts: How many times to try.
        backoff_seconds: The pause after the first failed attempt; it grows
            linearly with each further attempt.

    Returns:
        The size of the file in bytes, and its SHA-256 in hex.

    Raises:
        httpx.HTTPError: If a failure is not one that is retried, or the last
            attempt fails.
        ValueError: If the body is not a readable zip.
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
    on_fetched: Callable[[Manifest], None] | None = None,
) -> Manifest:
    """Fetch the datasets for *sessions* and return the updated manifest.

    A file that is on disk and matches its manifest entry is left alone, so a
    second run fetches nothing. A file on disk that the manifest does not vouch
    for fails its session: the corpus must never be built from bytes of unknown
    origin.

    A session named twice is fetched once. Every label is checked before the
    first request. When one session fails the others still finish, and the
    first error is raised afterwards. *on_fetched* is called with the manifest
    each time a file has been moved into place, with no wait in between, so the
    caller can keep the manifest in step with the disk even if the run stops.
    When *on_fetched* raises, a file that was not on disk before is removed
    again, as no manifest vouches for it, and no further request is made. A file
    fetched again with *force* has already replaced the old one by then, so the
    run that follows needs ``--force`` once more.

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
        on_fetched: Called with the manifest as it stands after each file that
            was fetched, for example to write it to disk. If it raises, the
            error is raised once the other sessions have finished.

    Returns:
        The manifest with one entry per session, old and new, ordered by session.

    Raises:
        ValueError: If a session label is malformed, if a file is on disk but
            not vouched for by the manifest, or if a body is not a zip.
        httpx.HTTPError: If a file cannot be fetched.
    """
    wanted = list(dict.fromkeys(sessions))
    urls = {session: dataset_url(url_template, session) for session in wanted}
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched_on = today or date.today()
    entries = known.by_session()
    gate = asyncio.Semaphore(concurrency)
    stopped = False

    def current() -> Manifest:
        return Manifest(files=tuple(entries[session] for session in sorted(entries)))

    async def fetch_one(session: str) -> None:
        nonlocal stopped
        url = urls[session]
        destination = raw_dir / url.rsplit("/", 1)[-1]
        entry = entries.get(session)
        if destination.exists() and not force:
            if entry is None or sha256_of(destination) != entry.sha256:
                raise ValueError(
                    f"{destination} is on disk but does not match the manifest. "
                    f"Fetch it again with --force."
                )
            return
        replaces_file = destination.exists()
        async with gate:
            if stopped:
                return
            logger.info("Fetching %s", url)
            size, digest = await _download_with_retries(
                client, url, destination, attempts, backoff_seconds
            )
        try:
            entries[session] = FetchedFile(
                session=session,
                file=destination.name,
                url=url,
                bytes=size,
                sha256=digest,
                fetched=fetched_on,
            )
            if on_fetched is not None:
                on_fetched(current())
        except BaseException:
            stopped = True
            if entry is None:
                entries.pop(session, None)
            else:
                entries[session] = entry
            if not replaces_file:
                destination.unlink(missing_ok=True)
            raise

    outcomes = await asyncio.gather(
        *(fetch_one(session) for session in wanted), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return current()
