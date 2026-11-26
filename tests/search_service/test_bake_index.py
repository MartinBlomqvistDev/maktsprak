"""Tests for search_service/bake_index.py, on a package of the kit's corpus."""

from __future__ import annotations

import shutil

import pytest

from search_service import bake_index
from search_service.package import MANIFEST_NAME, package_files, read_manifest, verify_package
from tests.search_service.package_kit import build_kit_package
from tests.search_service.spike_kit import write_spike_corpus


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = write_spike_corpus(tmp_path_factory.mktemp("kit"))
    return build_kit_package(corpus, tmp_path_factory.mktemp("built"))


@pytest.fixture
def downloaded(built, tmp_path):
    """The package as a download of its dataset holds it, with the card beside it."""
    source = tmp_path / "download"
    shutil.copytree(built[0], source)
    (source / "README.md").write_text("the dataset's card", encoding="utf-8")
    return source


def test_the_package_is_installed_as_its_manifest_names_it(built, downloaded, tmp_path):
    _, digest = read_manifest(built[0])
    count = bake_index.install(downloaded, tmp_path / "opt", digest)
    assert count == len(package_files(built[0]))
    assert not (tmp_path / "opt" / "README.md").exists()
    verify_package(tmp_path / "opt", manifest_sha256=digest)


def test_another_manifest_or_a_changed_file_stops_the_build(built, downloaded, tmp_path):
    _, digest = read_manifest(built[0])
    with pytest.raises(ValueError, match="pinned"):
        bake_index.install(downloaded, tmp_path / "a", "0" * 64)
    path = downloaded / "speech_rows.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="another hash"):
        bake_index.install(downloaded, tmp_path / "b", digest)


def test_an_unpinned_package_is_not_fetched(tmp_path, capsys):
    code = bake_index.main(
        ["--repo", "o/n", "--revision", "unpublished", "--manifest-sha256", "0" * 64,
         "--dest", str(tmp_path / "opt")]
    )  # fmt: skip
    assert code == 1 and "pinned" in capsys.readouterr().out
    assert MANIFEST_NAME == bake_index.MANIFEST_NAME
