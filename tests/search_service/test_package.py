"""Tests for search_service/package.py, on the kit's corpus."""

from __future__ import annotations

import shutil

import numpy as np
import pyarrow.parquet as pq
import pytest

from search.models import text_sha256
from search_service.package import (
    MANIFEST_NAME,
    STORE_DIR,
    SpeechTable,
    StalePackageError,
    package_files,
    read_manifest,
    verify_package,
    write_speech_table,
)
from search_service.spike.blob import StringBlob, write_strings
from search_service.spike.inputs import day_of, read_passage_columns
from tests.search_service.package_kit import ANSWER, EMBEDDING, RETRIEVAL, build_kit_package
from tests.search_service.spike_kit import write_spike_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_spike_corpus(tmp_path_factory.mktemp("kit"))


@pytest.fixture(scope="module")
def built(corpus, tmp_path_factory):
    return build_kit_package(corpus, tmp_path_factory.mktemp("built"))


def speeches_of(files):
    rows = pq.read_table(files.speeches, columns=["anf_id", "paragraphs", "text_sha256"])
    return {row["anf_id"]: row for row in rows.to_pylist()}


class TestBuild:
    def test_the_manifest_lists_every_other_file_with_its_size_and_hash(self, built):
        directory, manifest, *_ = built
        assert set(manifest.files) == set(package_files(directory))
        assert MANIFEST_NAME not in manifest.files
        assert f"{STORE_DIR}/store.json" in manifest.files
        verify_package(directory)

    def test_the_counts_and_dates_are_the_corpus(self, built):
        _, manifest, ids, _, files = built
        days = read_passage_columns(files.passages).days
        assert manifest.passages == len(ids)
        assert manifest.speeches == len({passage.rpartition(":")[0] for passage in ids})
        assert (manifest.first_date, manifest.last_date) == (
            day_of(int(days.min())),
            day_of(int(days.max())),
        )

    def test_the_settings_and_the_sources_are_recorded(self, built):
        _, manifest, *_ = built
        assert (manifest.retrieval, manifest.embedding, manifest.answer) == (
            RETRIEVAL,
            EMBEDDING,
            ANSWER,
        )
        assert manifest.sources.answer_run_sha256 == "d" * 64
        assert len(manifest.sources.vectors_sha256) == 64

    def test_the_store_holds_the_real_vectors_and_no_questions(self, built):
        directory, _, _, matrix, _ = built
        np.testing.assert_array_equal(
            np.load(directory / STORE_DIR / "vectors-8-float32.npy"), matrix
        )
        assert not (directory / STORE_DIR / "query_vectors.npy").exists()

    def test_a_package_is_never_built_over_another(self, corpus, built):
        directory, *_ = built
        with pytest.raises(FileExistsError):
            build_kit_package(corpus, directory.parent)


class TestSpeeches:
    def test_every_speech_is_its_passages_text_with_the_corpus_hash(self, built):
        directory, manifest, _, _, files = built
        table = SpeechTable(directory, StringBlob.open(directory / STORE_DIR / "text"))
        corpus = speeches_of(files)
        assert table.rows == manifest.speeches
        for row in range(table.rows):
            speech = table.speech(row)
            assert speech.paragraphs == tuple(corpus[speech.anf_id]["paragraphs"])
            assert speech.text_sha256 == text_sha256(speech.paragraphs)

    def test_each_passage_points_at_its_own_speech(self, built):
        directory, _, ids, _, _ = built
        table = SpeechTable(directory, StringBlob.open(directory / STORE_DIR / "text"))
        for position, passage in enumerate(ids):
            assert table.fields(table.row_of(position))["anf_id"] == passage.rpartition(":")[0]

    def test_the_fields_the_page_shows_are_kept(self, built):
        directory, *_ = built
        table = SpeechTable(directory, StringBlob.open(directory / STORE_DIR / "text"))
        fields = table.fields(0)
        for name in ("rm", "dok_nummer", "is_reply", "speaker_raw", "title", "url", "date"):
            assert name in fields

    def test_a_store_text_that_is_not_the_corpus_is_refused(self, built, tmp_path):
        directory, _, _, _, files = built
        texts = StringBlob.open(directory / STORE_DIR / "text")
        changed = [texts[position] for position in range(len(texts))]
        changed[1] = changed[1] + " tillagt"
        write_strings(tmp_path / "text", changed)
        with pytest.raises(StalePackageError, match="differs"):
            write_speech_table(
                files.speeches, files.passages, StringBlob.open(tmp_path / "text"), tmp_path
            )


class TestVerify:
    @pytest.fixture
    def copy(self, built, tmp_path):
        directory, *_ = built
        target = tmp_path / "copy"
        shutil.copytree(directory, target)
        return target

    def test_the_pinned_hash_of_the_manifest_must_match(self, copy):
        _, digest = read_manifest(copy)
        verify_package(copy, manifest_sha256=digest)
        with pytest.raises(StalePackageError, match="pinned"):
            verify_package(copy, manifest_sha256="0" * 64)

    def test_a_file_of_another_size_is_refused_even_without_hashes(self, copy):
        path = copy / "speech_rows.npy"
        path.write_bytes(path.read_bytes() + b"\0")
        with pytest.raises(StalePackageError, match="another size"):
            verify_package(copy, hashes=False)

    def test_a_changed_byte_is_found_by_the_hashes(self, copy):
        path = copy / STORE_DIR / "text.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(bytes(data))
        verify_package(copy, hashes=False)
        with pytest.raises(StalePackageError, match="another hash"):
            verify_package(copy)

    def test_a_missing_or_extra_file_is_refused(self, copy):
        (copy / "speech_rows.npy").unlink()
        (copy / "README.md").write_text("x", encoding="utf-8")
        with pytest.raises(StalePackageError, match="missing") as caught:
            verify_package(copy, hashes=False)
        assert "README.md is not in the manifest" in str(caught.value)
