"""Bootstrap intervals over questions.

The pre-registration fixes a paired bootstrap over questions, 10 000
resamples and 95% percentile intervals. Questions are resampled with
replacement; for a paired comparison the same resampled questions are used for
both systems, so the interval is for the difference itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

RESAMPLES = 10_000
LEVEL = 0.95


def _percentiles(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_interval(
    values: Sequence[float], seed: int, resamples: int = RESAMPLES, level: float = LEVEL
) -> tuple[float, float]:
    """A percentile interval for the mean of per-question values.

    Args:
        values: One value per question.
        seed: Decides the resamples.
        resamples: How many resamples.
        level: The interval's coverage.

    Returns:
        The lower and upper bound.

    Raises:
        ValueError: If there are no values.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("A bootstrap needs at least one value.")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, data.size, size=(resamples, data.size))
    return _percentiles(data[picks].mean(axis=1), level)


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    seed: int,
    resamples: int = RESAMPLES,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """The mean difference of two systems over the same questions, with its interval.

    Args:
        first: The first system's value per question.
        second: The second system's, question by question.
        seed: Decides the resamples.
        resamples: How many resamples.
        level: The interval's coverage.

    Returns:
        The mean of ``first - second``, and the interval's lower and upper bound.

    Raises:
        ValueError: If the two differ in length or are empty.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("A paired bootstrap needs two equally long, non-empty lists.")
    differences = a - b
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, differences.size, size=(resamples, differences.size))
    low, high = _percentiles(differences[picks].mean(axis=1), level)
    return float(differences.mean()), low, high
