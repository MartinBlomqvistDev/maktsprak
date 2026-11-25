"""Fetch the search service's package into the image, at build time and never at start.

Run by the Dockerfile's ``fetch`` stage, which holds nothing but this script and
the Hugging Face client. The package is downloaded from its dataset at one
pinned revision, and installed only if its manifest has the pinned hash and
every file the manifest names has its size and hash. Nothing else from the
dataset, such as its card, enters the image. A failure stops the build, so an
image never holds a package other than the one it pins.

The Hub limits the rate of downloads, so a failed download is tried again a few
times before the build gives up, as ``inference_service/bake_model.py`` does.

Usage::

    python bake_index.py --repo OWNER/NAME --revision SHA --manifest-sha256 HEX --dest /opt/index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

MANIFEST_NAME = "package.json"
ATTEMPTS = 5
#: What the Dockerfile pins until the package is published.
UNPUBLISHED = "unpublished"
_CHUNK = 1 << 22


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo: str, revision: str, into: Path) -> Path:
    """Download the dataset at *revision* into *into*, trying again after a failure."""
    from huggingface_hub import snapshot_download

    for attempt in range(1, ATTEMPTS + 1):
        try:
            return Path(
                snapshot_download(
                    repo_id=repo, repo_type="dataset", revision=revision, local_dir=into
                )
            )
        except Exception as error:
            if attempt == ATTEMPTS:
                raise
            backoff = 2**attempt * 5
            print(f"attempt {attempt}/{ATTEMPTS} failed ({error}); retrying in {backoff} s")
            time.sleep(backoff)
    raise AssertionError("unreachable")


def install(source: Path, dest: Path, manifest_sha256: str) -> int:
    """Check the package in *source* and move its files to *dest*.

    Args:
        source: The downloaded dataset.
        dest: Where the package goes; it must not exist.
        manifest_sha256: The hash the manifest must have.

    Returns:
        How many files were installed beside the manifest.

    Raises:
        ValueError: If the manifest's hash, or a file's size or hash, differs.
    """
    raw = (source / MANIFEST_NAME).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest_sha256:
        raise ValueError(f"The manifest's hash is {digest}, not the {manifest_sha256} pinned.")
    files = json.loads(raw)["files"]
    for relative, entry in files.items():
        path = source / relative
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{relative} is missing or has another size.")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{relative} has another hash.")
    for relative in files:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source / relative, target)
    (dest / MANIFEST_NAME).write_bytes(raw)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    """Fetch, check and install the package; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="The dataset, as OWNER/NAME.")
    parser.add_argument("--revision", required=True, help="The dataset's commit.")
    parser.add_argument("--manifest-sha256", required=True, help="The manifest's pinned hash.")
    parser.add_argument("--dest", type=Path, required=True, help="Where the package goes.")
    args = parser.parse_args(argv)
    if UNPUBLISHED in (args.revision, args.manifest_sha256):
        print("The package's revision and manifest hash must be pinned before the image is built.")
        return 1
    if args.dest.exists():
        print(f"{args.dest} exists; the package goes into a new directory.")
        return 1
    with tempfile.TemporaryDirectory() as staging:
        try:
            source = fetch(args.repo, args.revision, Path(staging))
            count = install(source, args.dest, args.manifest_sha256)
        except Exception as error:
            print(f"could not install the package: {error}")
            return 1
    print(f"installed {count} files of {args.repo}@{args.revision} into {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
