"""Settings for the search package, read from ``config.yaml``.

Paths resolve against two roots, because the two kinds of file have different
lifetimes. Bulk data (downloads, Parquet, indexes) is gitignored and lives
under the *data root*. Tracked output (results, manifests) lives in the
repository the code runs from, so a result is committed on the branch that
produced it.

The data root is the repository root unless an argument or the
``MAKTSPRAK_DATA_ROOT`` environment variable says otherwise. In a git worktree,
point it at the main checkout, where the corpus is.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_FILE: Path = Path(__file__).resolve().with_name("config.yaml")
DATA_ROOT_ENV: str = "MAKTSPRAK_DATA_ROOT"


class _Frozen(BaseModel):
    """Base for settings: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DataPaths(_Frozen):
    """Gitignored bulk data, resolved against the data root.

    Attributes:
        raw_anforanden: The Riksdag's speech datasets as downloaded.
        search: Parquet files, vectors and indexes built from them.
        legacy_corpus: The corpus parsed from the protocol PDFs, one record
            per speaker and protocol.
    """

    raw_anforanden: Path
    search: Path
    legacy_corpus: Path


class TrackedPaths(_Frozen):
    """Tracked output, resolved against the repository root.

    Attributes:
        reports: Results and manifests as JSON.
    """

    reports: Path


class Settings(_Frozen):
    """Everything ``config.yaml`` holds, with paths made absolute."""

    data: DataPaths
    tracked: TrackedPaths


def resolve_data_root(data_root: Path | None = None) -> Path:
    """Pick the data root: the argument, then the environment, then the repository.

    Args:
        data_root: An explicit data root, or ``None`` to fall back.

    Returns:
        The data root as an absolute path.

    Raises:
        FileNotFoundError: If the chosen root is not an existing directory.
    """
    if data_root is None:
        from_env = os.environ.get(DATA_ROOT_ENV)
        data_root = Path(from_env) if from_env else REPO_ROOT
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root {data_root} is not a directory.")
    return data_root.resolve()


def load_settings(config_file: Path = CONFIG_FILE, data_root: Path | None = None) -> Settings:
    """Read the settings file and resolve its paths.

    Args:
        config_file: The YAML file to read.
        data_root: An explicit data root, see :func:`resolve_data_root`.

    Returns:
        The validated, immutable settings.

    Raises:
        FileNotFoundError: If the data root does not exist.
        pydantic.ValidationError: If a key is missing, unknown or mistyped.
    """
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = resolve_data_root(data_root)
    resolved = dict(raw)
    resolved["data"] = {name: root / value for name, value in raw.get("data", {}).items()}
    resolved["tracked"] = {
        name: REPO_ROOT / value for name, value in raw.get("tracked", {}).items()
    }
    return Settings.model_validate(resolved)
