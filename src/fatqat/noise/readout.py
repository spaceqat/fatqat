"""Immutable classical readout-confusion declarations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, init=False, eq=False)
class ReadoutConfusion:
    """Classical report confusion applied after physical measurement.

    ``matrix[reported, true]`` is the conditional probability of reporting a
    digit given the physical measurement outcome. Each column must therefore
    sum to one. Readout confusion changes the reported classical digit, not the
    physical post-measurement state.

    Args:
        matrix: Square column-stochastic matrix with entries in ``[0, 1]``.
            Its side length must match the measured digit dimension supported
            by the selected backend.

    Raises:
        ValueError: If ``matrix`` is not finite, square, at least ``2 x 2``,
            bounded by ``[0, 1]``, and column-stochastic.

    Examples:
        Register asymmetric qubit readout confusion for one device operand:

        >>> import numpy as np
        >>> import fatqat as fq
        >>> confusion = fq.noise.ReadoutConfusion(
        ...     np.array([[0.98, 0.04], [0.02, 0.96]])
        ... )
        >>> noise = fq.NoiseModel()
        >>> noise.add(confusion, targets="q0")

    Attributes:
        matrix: Read-only copy of the normalized input matrix.
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
