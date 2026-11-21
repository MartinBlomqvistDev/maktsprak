"""The passages' fields as columns, for sampling, filtering and mapping hits to speeches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from search.corpus.passages import PERIOD_FIRST_YEARS, period_label


@dataclass(frozen=True)
class PassageTable:
    """One row per passage, in the order of ``passages.parquet``, which is index order.

    Attributes:
        passage_ids: The key of each passage.
        anf_ids: The speech each passage belongs to.
        parties: The party of each passage's speech.
        years: The year of each passage's speech.
        words: The words in each passage.
    """

    passage_ids: list[str]
    anf_ids: list[str]
    parties: np.ndarray
    years: np.ndarray
    words: np.ndarray
    _positions: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        sizes = {len(self.anf_ids), len(self.parties), len(self.years), len(self.words)}
        if sizes != {len(self.passage_ids)}:
            raise ValueError("The columns of the passage table differ in length.")
        self._positions.update((key, number) for number, key in enumerate(self.passage_ids))
        if len(self._positions) != len(self.passage_ids):
            raise ValueError("A passage id occurs twice in the passage table.")

    def __len__(self) -> int:
        return len(self.passage_ids)

    def position(self, passage_id: str) -> int:
        """Return the index position of a passage.

        Raises:
            KeyError: If the table has no such passage.
        """
        return self._positions[passage_id]

    @property
    def latest_year(self) -> int:
        """The latest year among the speeches, which names the last period."""
        return int(self.years.max())

    def period_labels(self) -> list[str]:
        """The labels of the periods, as the breakdowns name them."""
        return [period_label(first, self.latest_year) for first in PERIOD_FIRST_YEARS]

    def period_of_year(self, year: int) -> str:
        """The label of the period a year falls in.

        Raises:
            ValueError: If the year lies before the first period.
        """
        if year < PERIOD_FIRST_YEARS[0]:
            raise ValueError(f"{year} lies before the first period.")
        first = max(start for start in PERIOD_FIRST_YEARS if start <= year)
        return period_label(first, self.latest_year)

    def period_mask(self, label: str) -> np.ndarray:
        """Mark the passages whose speech falls in the period named *label*.

        Raises:
            ValueError: If *label* names no period.
        """
        labels = self.period_labels()
        if label not in labels:
            raise ValueError(f"{label!r} is not a period; the periods are {labels}.")
        number = labels.index(label)
        first = PERIOD_FIRST_YEARS[number]
        later = PERIOD_FIRST_YEARS[number + 1 :]
        mask = self.years >= first
        if later:
            mask &= self.years < later[0]
        return mask

    def party_mask(self, party: str) -> np.ndarray:
        """Mark the passages whose speech was given by a member of *party*."""
        return self.parties == party


def load_passage_table(passages_path: Path) -> PassageTable:
    """Read the passages' keys, speeches, parties, years and sizes.

    Args:
        passages_path: ``passages.parquet``.

    Returns:
        The table.
    """
    table = pq.read_table(passages_path, columns=["passage_id", "anf_id", "party", "date", "words"])
    dates = table.column("date").to_pylist()
    return PassageTable(
        passage_ids=table.column("passage_id").to_pylist(),
        anf_ids=table.column("anf_id").to_pylist(),
        parties=np.array(table.column("party").to_pylist(), dtype="<U2"),
        years=np.array([day.year for day in dates], dtype=np.int16),
        words=np.array(table.column("words").to_pylist(), dtype=np.int32),
    )
