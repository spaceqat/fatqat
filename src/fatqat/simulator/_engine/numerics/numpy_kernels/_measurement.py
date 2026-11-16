"""NumPy probabilities, physical projection, and categorical sampling."""

from collections.abc import Sequence

import numpy as np


def _strides(dims: Sequence[int]) -> list[int]:
    """Little-endian place values: ``stride[q] = prod(dims[:q])``."""
    strides = [1] * len(dims)
    for q in range(1, len(dims)):
        strides[q] = strides[q - 1] * dims[q - 1]
    return strides


def _measured_keep_mask(
    idx: int, subsystems: Sequence[int], dims: Sequence[int], size: int
) -> np.ndarray:
    """Boolean mask over flat basis states matching ``idx``'s measured digits.

    The projector core both collapse kernels share: an entry is kept when every
    measured subsystem's little-endian digit equals the corresponding digit of
    the sampled index ``idx``.
    """
    subsystems = list(subsystems)
    if len(set(subsystems)) == len(dims):
        keep = np.zeros(size, dtype=bool)
        keep[idx] = True
        return keep
    strides = _strides(dims)
    basis = np.arange(size)
    # One (N, m) broadcast instead of m separate O(N) passes: fold every measured
    # subsystem's stride and modulus into a single vectorized divide/mod/compare.
    stride_arr = np.array([strides[q] for q in subsystems])
    dim_arr = np.array([dims[q] for q in subsystems])
    digits = (basis[:, None] // stride_arr) % dim_arr
    idx_digits = (idx // stride_arr) % dim_arr
    return np.all(digits == idx_digits, axis=1)


def _statevector_probabilities(state: np.ndarray) -> np.ndarray:
    """Return normalized basis probabilities, preserving an all-zero input."""
    probabilities = np.abs(state) ** 2
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _project_statevector(
    state: np.ndarray,
    idx: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
) -> np.ndarray:
    """Return a normalized physical projection without modifying the input."""
    keep = _measured_keep_mask(idx, measured_subsystems, dims, len(state))
    new = state.copy()
    new[~keep] = 0.0
    norm = np.linalg.norm(new)
    return new / norm if norm > 0 else new


def _density_matrix_probabilities(rho: np.ndarray) -> np.ndarray:
    """Return normalized diagonal probabilities with round-off clipped."""
    # A valid rho's diagonal is real and non-negative; clip tiny round-off so
    # the values stay a valid sampling distribution.
    probabilities = np.clip(np.real(np.diagonal(rho)), 0.0, None)
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _project_density_matrix(
    rho: np.ndarray,
    idx: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
) -> np.ndarray:
    """Return a normalized physical projection without modifying the input."""
    keep = _measured_keep_mask(idx, measured_subsystems, dims, rho.shape[0])
    new = rho * keep[:, None] * keep[None, :]
    trace = np.real(np.trace(new))
    return new / trace if trace > 0 else new


def _sample_basis_indices(
    probabilities: np.ndarray, rng: np.random.Generator, shots: int | None
) -> int | np.ndarray:
    """Sample using NumPy's categorical algorithm and the supplied stream."""
    return rng.choice(len(probabilities), size=shots, p=probabilities)
