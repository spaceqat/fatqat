"""Classical readout-confusion noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, init=False, eq=False)
class ReadoutConfusion:
    """Classical confusion between true and reported measurement digits.

    Entry ``matrix[reported, true]`` is the conditional probability of
    reporting one digit given the true physical outcome. The matrix must be
    square with side length at least 2, contain only finite values in
    ``[0, 1]``, and be column-stochastic: each fixed-true column must sum to 1
    within NumPy's default numerical tolerance.

    FATQAT converts the input to float and stores its own read-only copy.
    Changing the input array later has no effect. Instances compare and hash
    by their stored matrix content.

    Confusion is applied after physical measurement. It changes the classical
    digit used for counts and feedforward, but not the true collapse outcome or
    post-measurement quantum state. A backend checks the side length against
    its reported classical digit dimension when a concrete measurement is
    prepared. In NoiseModel, readout confusion always applies at measurement.

    Args:
        matrix: Float-convertible array-like value containing the square
            column-stochastic confusion matrix.

    Raises:
        TypeError: If matrix cannot be converted to a numeric array.
        ValueError: If conversion fails or the resulting matrix is not finite,
            square, at least ``2 x 2``, bounded by ``[0, 1]``, and
            column-stochastic.

    Examples:
        Register asymmetric qubit readout confusion for every measurement:

        >>> import numpy as np
        >>> import fatqat as fq
        >>> confusion = fq.noise.ReadoutConfusion(
        ...     np.array([[0.98, 0.04], [0.02, 0.96]])
        ... )
        >>> noise = fq.NoiseModel()
        >>> noise.add(confusion)

    Attributes:
        matrix: Read-only float copy of the input matrix.
    """

    matrix: np.ndarray

    def __init__(self, matrix: ArrayLike) -> None:
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
