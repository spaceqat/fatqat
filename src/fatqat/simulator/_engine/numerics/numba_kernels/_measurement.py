"""Physical measurement and reset-shift primitives for Numba states."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np
from numba import njit

from ._layouts import _build_measurement_layout

# --- shared sampling ---


@njit(cache=True)
def _inverse_cdf_pick(
    probabilities: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Pick an index from ``probabilities`` by inverse CDF at uniform ``u``.

    Accumulates the cdf, normalizes it by its last entry, and returns the
    first entry strictly greater than ``u`` - the exact two steps
    ``rng.choice(n, p=...)`` performs internally, so one ``rng.random()`` fed
    here consumes the caller's stream identically. A zero-probability entry
    has a zero-width interval and is therefore never returned. Callers pass
    the same ``p`` their NumPy counterpart passes to ``rng.choice``, already
    normalized or not, so the arithmetic matches term for term.

    The normalization makes ``cdf[-1]`` exactly 1.0 and ``u < 1.0``, so the
    search cannot run off the end; the clamp holds that invariant explicitly
    because callers use the result to index memory.

    The cdf is never materialized: it is accumulated twice instead, once for
    its total and once to find the crossing. Same additions and same division
    per entry, and a cdf is non-decreasing, so this is bit-identical to
    bisecting the array.

    Basis-index sampling below uses a materialized CDF. Keep the two algorithms
    separate here to preserve their existing arithmetic and seeded behavior.
    """
    n = probabilities.shape[0]
    total = 0.0
    for i in range(n):
        total += probabilities[i]

    running = 0.0
    for i in range(n):
        running += probabilities[i]
        if running / total > u:
            return i
    return n - 1


@njit(cache=True)
def _normalized_cdf(
    probabilities: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Cumulative distribution normalized by its last element (``cdf /= cdf[-1]``).

    Built with the same sequential cumulative sum NumPy's ``cumsum`` uses, so a
    ``searchsorted`` against it matches ``rng.choice``'s internal inverse-CDF.
    """
    size = probabilities.shape[0]
    cdf = np.empty(size, dtype=np.float64)
    running = 0.0
    for i in range(size):
        running += probabilities[i]
        cdf[i] = running
    last = cdf[size - 1]
    for i in range(size):
        cdf[i] = cdf[i] / last
    return cdf


@njit(cache=True)
def _searchsorted_right(
    cdf: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Index of the first ``cdf`` entry strictly greater than ``u``.

    Equivalent to ``cdf.searchsorted(u, side="right")`` for a non-decreasing
    ``cdf`` - the inverse-CDF step of categorical sampling.
    """
    lo = 0
    hi = cdf.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _sample_index_kernel(
    probabilities: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Draw one flat basis index from ``probabilities`` given uniform ``u``."""
    return _searchsorted_right(_normalized_cdf(probabilities), u)


@njit(cache=True)
def _sample_indices_kernel(
    probabilities: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Draw one flat index per uniform in ``uniforms`` (built cdf reused)."""
    cdf = _normalized_cdf(probabilities)
    out = np.empty(uniforms.shape[0], dtype=np.int64)
    for j in range(uniforms.shape[0]):
        out[j] = _searchsorted_right(cdf, uniforms[j])
    return out


# --- statevector ---


@njit(cache=True)
def _sv_probabilities_kernel(
    state: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Computational-basis probabilities of a flat statevector.

    Mirrors ``np.abs(state) ** 2`` normalized by its sum: ``abs`` of a complex
    value is ``sqrt(re^2 + im^2)``, squared back to the Born probability.
    """
    size = state.shape[0]
    probabilities = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        magnitude = abs(state[i])
        probabilities[i] = magnitude * magnitude
        total += probabilities[i]
    if total > 0.0:
        for i in range(size):
            probabilities[i] = probabilities[i] / total
    return probabilities


@njit(cache=True)
def _project_statevector_strided(
    state: np.ndarray,
    measured_strides: np.ndarray,
    measured_dims: np.ndarray,
    index: int,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Project onto the measured digits of ``index`` and renormalize.

    Keeps every basis state whose measured subsystem digits match those of the
    sampled ``index``, zeros the rest, then divides by the surviving norm -
    exactly ``np.linalg.norm``-normalized projective collapse.
    """
    size = state.shape[0]
    m = measured_strides.shape[0]
    index_digits = np.empty(m, dtype=np.int64)
    for j in range(m):
        index_digits[j] = (index // measured_strides[j]) % measured_dims[j]

    out = np.zeros(size, dtype=np.complex128)
    norm_sq = 0.0
    for flat in range(size):
        keep = True
        for j in range(m):
            if (flat // measured_strides[j]) % measured_dims[j] != index_digits[j]:
                keep = False
                break
        if keep:
            amplitude = state[flat]
            out[flat] = amplitude
            norm_sq += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    if norm_sq > 0.0:
        norm = sqrt(norm_sq)
        for flat in range(size):
            out[flat] = out[flat] / norm
    return out


@njit(cache=True, inline="always")
def _sv_reset_kernel(
    state: np.ndarray,
    reset_strides: np.ndarray,
    reset_dims: np.ndarray,
    u: float,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Sample one branch, project, then shift the reset subsystems to zero.

    The caller supplies one uniform draw. Direct digit shifts avoid constructing
    local shift matrices or routing reset through generic gate application.
    """
    index = _sample_index_kernel(_sv_probabilities_kernel(state), u)
    state = _project_statevector_strided(state, reset_strides, reset_dims, index)
    for j in range(reset_strides.shape[0]):
        sub_stride = reset_strides[j]
        sub_dim = reset_dims[j]
        outcome = (index // sub_stride) % sub_dim
        if outcome != 0:
            # Projection leaves only this digit populated. Move it to zero in
            # the owned projected buffer instead of rotating a full new state.
            shift = outcome * sub_stride
            for base in range(0, state.shape[0], sub_stride * sub_dim):
                for offset in range(sub_stride):
                    source = base + shift + offset
                    state[base + offset] = state[source]
                    state[source] = 0.0
    return state


# --- density matrix ---


@njit(cache=True)
def _dm_probabilities_kernel(
    rho: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Computational-basis probabilities from the diagonal of ``rho``.

    Mirrors ``clip(real(diag(rho)), 0, None)`` normalized by its sum: a valid
    density matrix has a real, non-negative diagonal, so tiny negative round-off
    is clamped before normalizing into a sampling distribution.
    """
    size = rho.shape[0]
    probabilities = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        value = rho[i, i].real
        value = max(value, 0.0)
        probabilities[i] = value
        total += value
    if total > 0.0:
        for i in range(size):
            probabilities[i] = probabilities[i] / total
    return probabilities


@njit(cache=True)
def _project_density_matrix_strided(
    rho: np.ndarray,
    measured_strides: np.ndarray,
    measured_dims: np.ndarray,
    index: int,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Project ``rho`` onto the measured digits of ``index`` and renormalize.

    Keeps the block of basis states whose measured subsystem digits match those
    of the sampled ``index`` (both row and column must match), zeros the rest,
    then divides by the surviving trace - exactly the trace-normalized
    projective collapse ``rho * keep[:, None] * keep[None, :] / Tr``.
    """
    size = rho.shape[0]
    m = measured_strides.shape[0]
    index_digits = np.empty(m, dtype=np.int64)
    for j in range(m):
        index_digits[j] = (index // measured_strides[j]) % measured_dims[j]

    keep = np.empty(size, dtype=np.bool_)
    for i in range(size):
        match = True
        for j in range(m):
            if (i // measured_strides[j]) % measured_dims[j] != index_digits[j]:
                match = False
                break
        keep[i] = match

    trace = 0.0
    for i in range(size):
        if keep[i]:
            trace += rho[i, i].real
    # Mirror ``new / trace if trace > 0 else new``: on a zero-trace branch leave
    # the (already all-zero) kept block unscaled instead of dividing by zero.
    scale = 1.0 / trace if trace > 0.0 else 1.0

    out = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if keep[i]:
            for j in range(size):
                if keep[j]:
                    out[i, j] = rho[i, j] * scale
    return out


# --- Runtime-facing measurement functions ---


def _project_statevector(
    state: np.ndarray, index: int, indices: Sequence[int], dims: Sequence[int]
) -> np.ndarray:
    strides, measured_dims = _build_measurement_layout(dims, indices)
    return _project_statevector_strided(
        np.ascontiguousarray(state), strides, measured_dims, index
    )


def _project_density_matrix(
    state: np.ndarray, index: int, indices: Sequence[int], dims: Sequence[int]
) -> np.ndarray:
    strides, measured_dims = _build_measurement_layout(dims, indices)
    return _project_density_matrix_strided(
        np.ascontiguousarray(state), strides, measured_dims, index
    )


def _sample_basis_indices(
    probabilities: np.ndarray, rng: np.random.Generator, shots: int | None
) -> int | np.ndarray:
    """Draw uniforms from the workflow's stream and invert their CDF in Numba."""
    if shots is None:
        return int(_sample_index_kernel(probabilities, float(rng.random())))
    return _sample_indices_kernel(probabilities, rng.random(shots))


def _reset_statevector(
    state: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Reset one sampled branch using the fused Numba projection/shift kernel."""
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    strides, measured_dims = _build_measurement_layout(dims, indices)
    return _sv_reset_kernel(
        np.ascontiguousarray(state), strides, measured_dims, float(rng.random())
    )
