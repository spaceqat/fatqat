"""Compiled classical readout confusion after physical measurement."""

import numpy as np
from numba import njit

from ...numerics.numba_kernels._measurement import _inverse_cdf_pick


@njit(cache=True)
def _report_digit_kernel(
    conf_flat: np.ndarray, ptr: int, dim: int, true_digit: int, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Resample one reported readout digit through a confusion matrix.

    ``conf_flat[ptr:]`` holds a row-major ``(dim, dim)`` column-stochastic
    matrix ``C[i, j] = P(report i | true j)``, so column ``true_digit`` is the
    reporting distribution for this outcome and is already normalized - it is
    handed to `_inverse_cdf_pick` unscaled, exactly as the NumPy twin hands
    ``confusion[:, true_digit]`` to ``rng.choice``.

    Only the reported classical value is affected; the caller's collapsed
    state keeps the true outcome.
    """
    column = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        column[i] = conf_flat[ptr + i * dim + true_digit]
    return _inverse_cdf_pick(column, u)
