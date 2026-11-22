"""Strings kept as one UTF-8 file with the offset where each starts, read one at a time."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def blob_paths(stem: Path) -> tuple[Path, Path]:
    """The data file and the offsets file of the strings stored under *stem*."""
    return stem.with_name(stem.name + ".bin"), stem.with_name(stem.name + ".offsets.npy")


def write_strings(stem: Path, strings: Iterable[str]) -> int:
    """Write *strings* one after the other, with the offset of each.

    Args:
        stem: The path both files are named after; ``text`` gives ``text.bin``
            and ``text.offsets.npy``.
        strings: The strings, in order.

    Returns:
        How many strings were written.
    """
    data_path, offsets_path = blob_paths(stem)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = array("q", [0])
    with data_path.open("wb") as out:
        for string in strings:
            encoded = string.encode("utf-8")
            out.write(encoded)
            offsets.append(offsets[-1] + len(encoded))
    np.save(offsets_path, np.frombuffer(offsets, dtype=np.int64))
    return len(offsets) - 1


class StringBlob:
    """Strings written by :func:`write_strings`, memory-mapped and decoded one at a time.

    Attributes:
        data: The UTF-8 bytes of every string.
        offsets: Where each string starts in ``data``, and the end of the last.
    """

    def __init__(self, data: np.ndarray, offsets: np.ndarray) -> None:
        if len(offsets) == 0 or int(offsets[-1]) != len(data):
            raise ValueError("The offsets do not fit the data.")
        self.data = data
        self.offsets = offsets

    @classmethod
    def open(cls, stem: Path) -> StringBlob:
        """Open the strings stored under *stem*.

        Args:
            stem: The path given to :func:`write_strings`.

        Returns:
            The strings.
        """
        data_path, offsets_path = blob_paths(stem)
        offsets = np.load(offsets_path)
        if data_path.stat().st_size == 0:
            data = np.zeros(0, dtype=np.uint8)
        else:
            data = np.memmap(data_path, dtype=np.uint8, mode="r")
        return cls(data, offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> str:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.data[start:end].tobytes().decode("utf-8")
