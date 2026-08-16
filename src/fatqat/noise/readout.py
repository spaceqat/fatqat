"""Immutable classical readout-confusion declarations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, init=False, eq=False)
class ReadoutConfusion:
    """Classical report confusion applied after physical measurement.

    ``matrix[reported, true]`` is the conditional probability of reporting a
    digit given the physical measurement outcome. The matrix is copied and
    frozen so this declaration is safe to share with a backend snapshot.
    """

    matrix: np.ndarray

    def __init__(self, matrix: np.ndarray) -> None:
        value = np.array(matrix, dtype=float, copy=True)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(
                f"confusion matrix must be square, got shape {value.shape}"
            )
        if value.shape[0] < 2:
            raise ValueError(
                "confusion matrix side length must be >= 2, " f"got {value.shape[0]}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("confusion matrix entries must be finite")
        if np.any(value < 0) or np.any(value > 1):
            raise ValueError("confusion matrix entries must be in [0, 1]")
        if not np.allclose(value.sum(axis=0), 1.0):
            raise ValueError(
                "confusion matrix must be column-stochastic: each column "
                "C[:, j] = P(report | true j) must sum to 1"
            )
        value.flags.writeable = False
        object.__setattr__(self, "matrix", value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReadoutConfusion):
            return NotImplemented
        return np.array_equal(self.matrix, other.matrix)

    def __hash__(self) -> int:
        return hash((self.matrix.shape, self.matrix.dtype.str, self.matrix.tobytes()))
